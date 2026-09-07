//! **The emitted tier follows a relocating window** (#1312) — the pin for the `"win"` global.
//!
//! A cooperative run's window backing can now grow (`Region::Growable`), and growing it reallocates
//! the buffer, which may **move** it inside the host's linear memory. An emitted frame receives its
//! window base as the `win` parameter, so a grow that happens *during* a cross-tier bounce leaves
//! every subsequent access in that frame addressing the **old**, freed base — silently reading and
//! writing the wrong memory, which is exactly the class of bug invariant #2 forbids.
//!
//! The emit therefore treats `win` as re-readable state rather than a fixed parameter:
//!  - every emitted function **publishes** its `win` to the exported `"win"` global on entry, so a
//!    host with a fixed window never has to write it (the publish keeps it correct by itself), and
//!  - every call that can run guest code is followed by a **reload** of local 0 from that global,
//!    so a host that *did* relocate the window is obeyed from the next instruction onward.
//!
//! This test plays the relocating host directly: the `env.call_interp` stub moves the whole window
//! to a second base, republishes it, and returns. The emitted leaf's post-bounce store must land at
//! the new base and **not** at the old one — the assertion that fails without the reload.

use temen_wasm_jit::compile_module_tierup_b2;
use wasmi::{
    Caller, Engine, FuncRef, Global, Linker, Memory, MemoryType, Module as WModule, Store, Table,
    TableType, Val,
};

/// The window's first home in the mirrored linear memory, and the second one it moves to. Far apart
/// so a stale-base store lands in the old window rather than incidentally in the new one.
const OLD_WIN: u32 = 0x4_0000;
const NEW_WIN: u32 = 0x14_0000;
/// The mirrored window's length (`memory 16`).
const WIN_LEN: usize = 1 << 16;
const ENV_PTR: u32 = 1024;
/// The shared reserved dispatch table's size, as the coop driver sizes it for a tiny guest.
const TABLE_LOG2: u32 = 4;

/// Where the leaf stores, window-relative: once before the bounce, once after. Both must clear the
/// #1094 unconditional NULL guard (`[0, 16 KiB)` faults on any guest access) and the relocated args
/// region above it, so they sit past 32 KiB — the `coop_tierup_driver` harness picks its slot the
/// same way.
const ADDR_BEFORE: i64 = 32768 + 2048;
const ADDR_AFTER: i64 = 32768 + 2048 + 8;
const VAL_BEFORE: i64 = 0xABCD;
const VAL_AFTER: i64 = 0xBEEF;
/// What the interpreted helper "returns" through the scratch — carried through to the result so the
/// bounce is proven to have actually run.
const BOUNCE_K: i64 = 7_000_000;

/// func 0: an interp-driven entry (never run here). func 1: the **emitted leaf** under test — store,
/// bounce cross-tier, store again, then read both back. func 2: interpreter-resident (it holds a
/// `call.cap`, so the emitter cannot emit it), which is what makes the `call 2` a cross-tier bounce.
const SRC: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vr = call 1 (vz)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  va1 = i64.const 34816
  vb = i64.const 43981
  i64.store va1 vb
  vh = call 2 (v0)
  va2 = i64.const 34824
  vaf = i64.const 48879
  i64.store va2 vaf
  vl1 = i64.load va1
  vl2 = i64.load va2
  vsum = i64.add vl1 vl2
  vtot = i64.add vsum vh
  return vtot
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vas = i32.const 0
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  return vr
  }
}
"#;

/// The relocating host's state: the handles the bounce needs (the emitted module *imports* memory
/// and the table, so the stub cannot reach them through `Caller::get_export`) plus what it observed.
#[derive(Default)]
struct Bounce {
    mem: Option<Memory>,
    /// The emitted module's exported `"win"` global — filled in after instantiation.
    win_global: Option<Global>,
    /// The `"win"` global as the emitted frame left it — must be the base the host passed in,
    /// proving the entry publish happened (this host never wrote the global before the bounce).
    win_at_bounce: i32,
    calls: u32,
}

#[test]
fn an_emitted_frame_follows_the_window_across_a_relocating_bounce() {
    let m = temen_text::parse_module(SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    // The **B2** emit — the shape the cooperative driver runs (`coop_emit_for`): a shared reserved
    // dispatch table and the #888 widened cross-tier set, so an in-subset leaf that calls a
    // cap-bearing helper stays emitted with its bounce intact. That bounce is the whole point here.
    let (wasm, eligible) = compile_module_tierup_b2(&m, false, TABLE_LOG2).expect("emit");
    assert!(
        eligible[1] && !eligible[2],
        "the leaf emits; the cap-bearing helper stays interpreter-resident (so `call 2` bounces): {eligible:?}"
    );

    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("emitted wasm must validate");
    let mut store: Store<Bounce> = Store::new(&engine, Bounce::default());
    let pages = ((NEW_WIN as usize + WIN_LEN) as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    store.data_mut().mem = Some(memory);
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();

    let mut linker: Linker<Bounce> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let tsize = 1u32 << TABLE_LOG2;
    let table = Table::new(
        &mut store,
        TableType::new(wasmi::core::ValType::FuncRef, tsize, Some(tsize)),
        Val::FuncRef(FuncRef::null()),
    )
    .unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, Bounce>, _code: i32| {})
        .unwrap();
    // The relocating host: service the bounce, then **move the window** and republish its base —
    // precisely what a `Region::Growable` backing does when a guest allocator's `vm_map` grows it
    // past the current buffer and the allocator hands back a different address.
    linker
        .func_wrap(
            "env",
            "call_interp",
            |mut c: Caller<'_, Bounce>, target: i32, args_ptr: i32| {
                assert_eq!(target, 2, "only the cap-bearing helper bounces");
                let mem = c.data().mem.expect("memory registered before the run");
                let win_global = c
                    .data()
                    .win_global
                    .expect("the emitted module exports the live-win global");
                let seen = match win_global.get(&c) {
                    Val::I32(v) => v,
                    other => panic!("`win` must be an i32 global, got {other:?}"),
                };
                // Relocate: copy the whole window to its new home, leaving the old bytes in place so
                // a stale-base store is *visible* rather than silently landing in the same buffer.
                let mut buf = vec![0u8; WIN_LEN];
                mem.read(&c, seen as usize, &mut buf).unwrap();
                mem.write(&mut c, NEW_WIN as usize, &buf).unwrap();
                win_global.set(&mut c, Val::I32(NEW_WIN as i32)).unwrap();
                // The helper's i64 result rides back in the cross-tier scratch (slot 0). The
                // emitted code passes `args_ptr` already offset to the scratch, so slot 0 is at 0.
                mem.write(&mut c, args_ptr as usize, &BOUNCE_K.to_le_bytes())
                    .unwrap();
                let d = c.data_mut();
                d.win_at_bounce = seen;
                d.calls += 1;
            },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let win_global = instance
        .get_global(&store, "win")
        .expect("the emitted module exports the live-win global");
    store.data_mut().win_global = Some(win_global);
    let f1 = instance.get_func(&store, "f1").expect("f1 exported");
    let mut results = [Val::I64(0)];
    f1.call(
        &mut store,
        &[
            Val::I32(OLD_WIN as i32),
            Val::I32(ENV_PTR as i32),
            Val::I64(0),
        ],
        &mut results,
    )
    .expect("the emitted leaf runs to completion");

    assert_eq!(store.data().calls, 1, "the bounce ran exactly once");
    assert_eq!(
        store.data().win_at_bounce,
        OLD_WIN as i32,
        "the entry publish put THIS frame's `win` in the global (no host write preceded it)"
    );

    // The post-bounce store landed at the NEW base...
    let read = |m: &Memory, s: &Store<Bounce>, at: u32| -> i64 {
        let mut b = [0u8; 8];
        m.read(s, at as usize, &mut b).unwrap();
        i64::from_le_bytes(b)
    };
    assert_eq!(
        read(&memory, &store, NEW_WIN + ADDR_AFTER as u32),
        VAL_AFTER,
        "the store after the bounce must land at the relocated window base"
    );
    // ...and NOT at the old one. This is the assertion the reload exists for: without it the frame
    // keeps its stale `win` parameter and writes into the abandoned buffer.
    assert_eq!(
        read(&memory, &store, OLD_WIN + ADDR_AFTER as u32),
        0,
        "nothing may be written through the stale base after a relocation"
    );
    // The pre-bounce store stays where it was written and survives the move (the host copied it).
    assert_eq!(
        read(&memory, &store, OLD_WIN + ADDR_BEFORE as u32),
        VAL_BEFORE
    );
    assert_eq!(
        read(&memory, &store, NEW_WIN + ADDR_BEFORE as u32),
        VAL_BEFORE
    );

    // Both loads after the bounce read through the new base, so the leaf's own result agrees.
    let got = match results[0] {
        Val::I64(v) => v,
        ref other => panic!("i64 result expected, got {other:?}"),
    };
    assert_eq!(
        got,
        VAL_BEFORE + VAL_AFTER + BOUNCE_K,
        "the leaf reads both stores back through the relocated window"
    );
}
