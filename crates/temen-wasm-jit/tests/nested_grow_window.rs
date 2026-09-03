//! **#1123 slice 4a — a §14 confined child GROWS its carve, byte-identical on the wasm-JIT tier and
//! the interpreter.** The parent grants the child a carve (`memory 16` = 64 KiB) larger than the
//! child's declared/committed window (`memory 15` = 32 KiB); the child `vm_map`-grows into the granted
//! tail (op 0 — an outlined `env.call_interp` leaf) and then accesses the freshly-committed region.
//! Because a nested child's `& MASK` clamps only to the full `DEFAULT_RESERVED_LOG2` (=40) reservation,
//! the *only* live confinement bound is the exported `"mapped"` global — so the emitted access must
//! admit exactly what the interpreter admits once the driver advances `"mapped"` past the grow, and
//! fault where an un-grown `"mapped"` would.
//!
//! The oracle is [`bytecode::Vcpu::new_confined_child_grow`] — the growable-confined-child primitive
//! (the parent grants the AddressSpace over the carve, `mapped` starts at the declared window and grows
//! into it). The emitted side services the child's `map` bounce by advancing `"mapped"` (the #717
//! driver contract, applied to a nested child's grow). This pins the confinement property the growable
//! op-13 nim phases rest on, and is the harness slice 4b/4c (fuzz + browser backing-grow) extend.

use std::sync::Arc;
use temen_interp::{bytecode, Region, Trap, Value};
use temen_wasm_jit::{compile_module_nested, outline_nested_cap_calls, TRAP_MEMORY_FAULT};
use wasmi::{Caller, Engine, Global, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: i32 = 0x1_0000; // the child carve base in the wasmi linear memory
const ENV_PTR: i32 = 1024; // dispatcher-fuel cell (below the child window)
const DECL: u8 = 15; // child declared window: `memory 15` = 32 KiB — initial `"mapped"`
const CARVE: u8 = 16; // parent-granted carve: 64 KiB — the child grows into `[32 KiB, 64 KiB)`
const GROWN: u64 = 1 << CARVE; // grown high-water = the whole carve
const STORE_AT: u64 = 40960; // 32 KiB + 8 KiB — inside the grown region, past the declared window

/// The child entry `(inst, as) -> i64`. When `grow_len` is `Some(len)`, it `map`s
/// `[32 KiB, 32 KiB + len)` through its AddressSpace handle (op 0 — an outlined leaf bounce) before
/// storing the sentinel at `store_at`; otherwise it stores straight away (which, with `"mapped"` still
/// at the 32 KiB declared prefix, is out of bounds for any `store_at >= 32 KiB`). Either way it loads
/// `store_at` back and returns it.
fn grow_child_src(grow_len: Option<u64>, store_at: u64) -> String {
    let map = match grow_len {
        Some(len) => format!(
            "  vas = i32.wrap_i64 v1\n  voff = i64.const 32768\n  vlen = i64.const {len}\n  vprot = i32.const 3\n  vm = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)\n"
        ),
        None => String::new(),
    };
    format!(
        r#"memory {DECL}
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
{map}  vsent = i64.const 424242
  vsa = i64.const {store_at}
  i64.store vsa vsent
  vld = i64.load vsa
  return vld
  }}
}}
"#
    )
}

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Val(i64),
    Trap,
}

/// The interpreter oracle: run `child` as a **growable** confined child — the parent grants a
/// `1<<CARVE` carve, the committed window starts at the declared `1<<DECL` and grows on `vm_map`. The
/// backing covers the whole carve. Returns the entry's result or a trap.
fn interp_confined(child: &temen_ir::Module) -> Outcome {
    let prog = bytecode::VcpuProgram::compile(child).expect("compile");
    let carve = 1usize << CARVE;
    let layout = std::alloc::Layout::from_size_align(carve, 8).unwrap();
    // SAFETY: non-zero 8-aligned layout; owned here until freed after the vCPU (and its `Mem`) drop.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `carve` valid bytes, exclusively this child's window, freed only after the vCPU.
    let back = Arc::new(unsafe { Region::shared(base, carve as u64) });
    let out = {
        let mut vcpu = bytecode::Vcpu::new_confined_child_grow(
            &prog,
            0,
            0,
            Arc::clone(&back),
            DECL,
            CARVE,
            u64::MAX,
        )
        .expect("growable confined child builds");
        match vcpu.run() {
            bytecode::VcpuEvent::Done(v) => Outcome::Val(match v.first() {
                Some(Value::I64(x)) => *x,
                _ => panic!("child returns one i64"),
            }),
            bytecode::VcpuEvent::Trapped(Trap::MemoryFault) => Outcome::Trap,
            bytecode::VcpuEvent::Trapped(t) => panic!("unexpected child trap: {t:?}"),
            _ => panic!("unexpected confined-child event (expected Done/Trapped)"),
        }
    };
    drop(back);
    // SAFETY: same layout; the vCPU and its region view are dropped, so no borrow outlives this.
    unsafe { std::alloc::dealloc(base, layout) };
    out
}

/// Per-`Store` state for the emitted run: the outlined module (so the `env.call_interp` handler can
/// read the `map` args' shape), the captured `"mapped"` global (advanced from inside the handler), the
/// memory, and the emitted-trap sink + bounce count.
struct Driver {
    m: temen_ir::Module,
    mem: Option<Memory>,
    mapped_global: Option<Global>,
    trap: i32,
    bounces: u32,
}

/// The emitted run: compile `child` **mask-only** (`compile_module_nested`) and run its `f0` under
/// wasmi, servicing the outlined `map` wrapper by advancing `"mapped"` to the grown high-water — the
/// nested twin of `tierup_grow_window`'s `sync`. Returns the outcome and the bounce count (non-vacuity).
fn emitted_nested(child: &temen_ir::Module) -> (Outcome, u32) {
    let mut m = child.clone();
    outline_nested_cap_calls(&mut m);
    let wasm = compile_module_nested(&m, false).expect("map-only nested child emits mask-only");

    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("nested wasm validates");
    let mut store: Store<Driver> = Store::new(
        &engine,
        Driver {
            m,
            mem: None,
            mapped_global: None,
            trap: 0,
            bounces: 0,
        },
    );
    // Physically cover `[WIN_BASE, WIN_BASE + GROWN)` so a non-faulting grown access lands in real
    // memory (a bounds bug then diverges instead of reading adjacent bytes).
    let need = WIN_BASE as usize + GROWN as usize;
    let pages = (need as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    store.data_mut().mem = Some(memory);

    let mut linker: Linker<Driver> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, Driver>, code: i32| {
            c.data_mut().trap = code;
        })
        .unwrap();
    // The outlined `map` wrapper bounce: read `(off, len)` from the env scratch, advance `"mapped"` to
    // `off + len` (the freshly-committed high-water), write the wrapper's `0` result back. The #717
    // driver contract applied to a nested child's grow — what the browser Worker does when a child's
    // `vm_map` advances its live window. (The parent-granted carve caps how far the child may grow; the
    // interpreter oracle enforces that via the AddressSpace grant, so a faithful servicer never
    // advances past the carve — here `off+len == GROWN`.)
    let mem = memory;
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, Driver>, func: i32, args_ptr: i32| {
                let op = match &caller.data().m.funcs[func as usize].blocks[0].insts[0] {
                    temen_ir::Inst::CapCall { op, .. } => *op,
                    other => panic!("wrapper body must be a cap-call, got {other:?}"),
                };
                assert_eq!(
                    op, 0,
                    "the only bounced leaf here is ADDRESS_SPACE.map (op 0)"
                );
                let data = mem.data(&caller);
                let slot = |i: usize| {
                    let o = args_ptr as usize + i * 8;
                    u64::from_le_bytes(data[o..o + 8].try_into().unwrap())
                };
                let off = slot(1);
                let len = slot(2);
                let mg = caller.data().mapped_global.unwrap();
                let cur = mg.get(&caller).i64().unwrap() as u64;
                // Mirror the interpreter's `map`: the committed high-water rounds **up to the host
                // page** (16 KiB macOS / 4 KiB Linux), so the live `mapped` a faithful driver reads back
                // is page-rounded — not the raw `off+len`. `GROWN` (the carve) caps it (page-aligned).
                let page = temen_interp::host_page_size();
                let grown = cur.max((off + len).next_multiple_of(page)).min(GROWN);
                mg.set(&mut caller, Val::I64(grown as i64)).unwrap();
                caller.data_mut().bounces += 1;
                // `map` returns 0 on success — write it into the result slot (slot 0).
                let out = mem.data_mut(&mut caller);
                let o = args_ptr as usize;
                out[o..o + 8].copy_from_slice(&0u64.to_le_bytes());
            },
        )
        .unwrap();
    define_unused(&mut linker);

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let mapped = instance
        .get_global(&store, "mapped")
        .expect("nested module exports the live-mapped global");
    store.data_mut().mapped_global = Some(mapped);
    // Seed `"mapped"` to the declared prefix (the child's initial committed window), the per-call
    // driver contract — growth past it must come from a serviced `map`, not the emit-time default.
    mapped.set(&mut store, Val::I64(1i64 << DECL)).unwrap();

    let f0 = instance.get_func(&store, "f0").expect("f0 exported");
    // f0(win, env, inst_handle, as_handle) — the handles are opaque to the servicer (dummy values).
    let params = [
        Val::I32(WIN_BASE),
        Val::I32(ENV_PTR),
        Val::I64(1),
        Val::I64(2),
    ];
    let mut results = [Val::I64(0)];
    let outcome = match f0.call(&mut store, &params, &mut results) {
        Ok(()) => Outcome::Val(results[0].i64().expect("i64")),
        Err(_) => Outcome::Trap,
    };
    assert!(
        outcome != Outcome::Trap || store.data().trap == TRAP_MEMORY_FAULT,
        "a trap here must be the emitted MemoryFault, got trap code {}",
        store.data().trap
    );
    (outcome, store.data().bounces)
}

/// Define the remaining §14 nested imports as unreachable (never called by a pure map-grow child).
fn define_unused(linker: &mut Linker<Driver>) {
    linker
        .func_wrap(
            "env",
            "instantiate",
            |_: Caller<'_, Driver>, _w: i32, _i: i32, _e: i64, _o: i64, _s: i64, _q: i64| -> i32 {
                unreachable!("no instantiate")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "join",
            |_: Caller<'_, Driver>, _i: i32, _c: i32| -> i64 { unreachable!("no join") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_spawn",
            |_: Caller<'_, Driver>, _f: i32, _sp: i64, _a: i64| -> i32 {
                unreachable!("no thread")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_join",
            |_: Caller<'_, Driver>, _h: i32| -> i64 { unreachable!("no thread") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_wait",
            |_: Caller<'_, Driver>, _w: i32, _a: i64, _e: i64, _t: i64, _is64: i32| -> i32 {
                unreachable!("no futex")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_notify",
            |_: Caller<'_, Driver>, _w: i32, _a: i64, _c: i32| -> i32 { unreachable!("no futex") },
        )
        .unwrap();
}

/// The grow case: the child `map`s `[32 KiB, 64 KiB)` then stores+loads the sentinel at `STORE_AT` — an
/// access into the freshly-grown region. The emitted tier admits it (its bound tracks the serviced
/// `"mapped"`) exactly as the interpreter growable-confined-child oracle does, byte-identical.
#[test]
fn nested_child_grow_access_matches_interpreter() {
    let child = parse(&grow_child_src(Some(32768), STORE_AT));
    let (emitted, bounces) = emitted_nested(&child);
    assert!(bounces >= 1, "the map wrapper must bounce, saw {bounces}");
    assert_eq!(
        interp_confined(&child),
        Outcome::Val(424242),
        "interp oracle: the growable confined child grows into the granted carve and reads back its sentinel"
    );
    assert_eq!(
        emitted, Outcome::Val(424242),
        "the emitted nested child confines to the live `mapped`; the grown access matches the interpreter"
    );
}

/// The negative: the same access **without** the grow. With `"mapped"` still at the 32 KiB declared
/// prefix, a store at `STORE_AT` is out of bounds — it must fault on both tiers (the bound is live, not
/// a blanket admit).
#[test]
fn nested_child_ungrown_access_faults_on_both_tiers() {
    let child = parse(&grow_child_src(None, STORE_AT));
    let (emitted, bounces) = emitted_nested(&child);
    assert_eq!(bounces, 0, "no map, so no bounce");
    assert_eq!(
        interp_confined(&child),
        Outcome::Trap,
        "interp oracle: a store past the un-grown `mapped` faults"
    );
    assert_eq!(
        emitted,
        Outcome::Trap,
        "the emitted nested child faults on the ungrown access exactly as the interpreter does"
    );
}

// ---- fuzz the grown-carve confinement as its own unit (AGENTS.md / INVARIANTS §2) -------------------
//
// A growing confined child is the confinement boundary in motion, so its masking lowering is fuzzed
// directly: across random grow extents and store addresses, the emitted nested child's outcome must
// equal the interpreter growable-confined-child oracle — the emitted access is admitted exactly where
// the interpreter admits it (inside `[guard, live mapped)`) and faults everywhere else. Deterministic
// (SplitMix64, no dev-deps — the escape TCB stays dependency-free).

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[test]
fn grown_carve_confinement_matches_interpreter_over_random_configs() {
    let mut rng = Rng(0x1123_5104_C0FF_EE00);
    for _ in 0..400 {
        // Grow `[32 KiB, 32 KiB + len)` for a page-aligned `len` in `[0, 32 KiB]` (up to the carve), or
        // no grow at all — so `mapped` lands anywhere in `[declared, carve]`.
        let grow_len = match rng.next() % 4 {
            0 => None,
            _ => Some((rng.next() % 9) * 4096), // 0, 4 KiB, .. 32 KiB
        };
        // A store anywhere in the carve `[0, 64 KiB)`, 8-aligned: below the guard, in the declared
        // prefix, in the grown tail, or past the live high-water — every side of both bounds.
        let store_at = (rng.next() % (GROWN / 8)) * 8;

        let child = parse(&grow_child_src(grow_len, store_at));
        let (emitted, _) = emitted_nested(&child);
        let interp = interp_confined(&child);
        assert_eq!(
            emitted, interp,
            "grown-carve confinement diverged: grow_len={grow_len:?} store_at={store_at} \
             emitted={emitted:?} interp={interp:?}"
        );
        // Whenever the access is admitted it returned the sentinel; a fault is the only other outcome —
        // the emitted nested child never reads adjacent bytes past its live window.
        if let Outcome::Val(v) = emitted {
            assert_eq!(
                v, 424242,
                "an admitted access must read back exactly what it stored"
            );
        }
    }
}
