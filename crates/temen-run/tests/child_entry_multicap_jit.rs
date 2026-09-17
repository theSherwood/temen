//! **Multi-record op-13 grant marshaling on the emitted tier** (#1221, NIM.md §3c).
//!
//! The front-end drivers hand a §14 op-13 child a **four**-record grant list — `{fs, stdout, exit,
//! exec}` — laid out in the parent's guarded window: records at `guard + 1024..`, cap-names at
//! `guard + 2048..`. `nifler_child_asset.rs` guards that layout on the **tree-walker**. On the
//! **JIT** the only per-PR coverage is `rust_guest_op13`, which writes exactly **one** record, so the
//! multi-record offset arithmetic — the part that actually regressed at #1094 — has been running
//! unguarded on the emitted tier.
//!
//! `nifler_child_jit.rs` does exercise it, with three records and the real nifler, but it is
//! `#[ignore]`d: Cranelift-compiling nifler's 100+ funcs takes ~250 s in debug. Its ignore note
//! points at a fast `child_entry_io_jit` repro as the substitute that keeps the class covered — **that
//! test does not exist anywhere in the tree**, so the substitution was never actually made. This is
//! it, written to be the cheap one: two hand-written text-IR modules, no asset, no toolchain, a child
//! small enough that the JIT compiles it in well under a second.
//!
//! The sensitivity comes from *where* the used caps sit. The grant list is ordered
//! `[extra, exit, stdout, fs]`, so the two caps the child actually exercises are the **last two**
//! records — index 2 and index 3. A parent that miscomputes the 16-byte record stride, or the
//! name-offset/length packing in a record's first word, resolves them to the wrong handle or to
//! nothing at all, and the child fails to write. `extra` is offered and ignored, mirroring the spare
//! record in `nifler_child_asset.rs`'s four-cap parent.

use std::ffi::c_void;
use std::sync::Arc;

use temen_interp::{ForkedProc, Host, HostProc, HostProcFork, StreamRole};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};

/// What the child writes through the granted `fs`, and through the granted `stdout`.
const FILE_BODY: &str = "jit-multicap";
const STREAM_BODY: &str = "ok";

/// The child: resolve two of the four granted names and use both. `self.resolve` is a name lookup in
/// the child's own powerbox, so it only succeeds if the parent's record for that name was marshaled
/// to the right offset with the right handle.
const CHILD: &str = r#"
memory 16
data 16384 "fs"
data 16392 "stdout"
data 16400 "out.bin"
data 16416 "jit-multicap"
data 16432 "ok"
func (i64) -> (i64) {
block 0 (vstarter: i64) {
  vfp = i64.const 16384
  vfl = i64.const 2
  vfs = self.resolve vfp vfl
  vpath = i64.const 16400
  vplen = i64.const 7
  vflags = i64.const 26
  vzero = i64.const 0
  vfd = call.cap 13 0 (i64, i64, i64, i64) -> (i64) vfs (vpath, vplen, vflags, vzero)
  vbuf = i64.const 16416
  vblen = i64.const 12
  vn = call.cap 13 2 (i64, i64, i64, i64) -> (i64) vfs (vfd, vbuf, vblen, vzero)
  vc = call.cap 13 4 (i64, i64, i64, i64) -> (i64) vfs (vfd, vzero, vzero, vzero)
  vsp = i64.const 16392
  vsl = i64.const 6
  vs = self.resolve vsp vsl
  vob = i64.const 16432
  vol = i64.const 2
  vw = call.cap 0 1 (i64, i64) -> (i64) vs (vob, vol)
  return vn
  }
}
"#;

/// The four-record guarded parent. Records at `guard + 1024`, 16 bytes each — word 0 packs the
/// cap-name as `name_off | (name_len << 32)`, word 1 the handle; names at `guard + 2048`. Both sit
/// **above** the #1094 NULL guard `[0, POWERBOX_NULL_GUARD)`, the layout the front-end drivers use.
fn parent_src(child_sl: u32, carve_off: u64) -> String {
    let parent_sl = child_sl + 1;
    let guard = temen_ir::POWERBOX_NULL_GUARD;
    let rec_base = guard + 1024;
    let name_base = guard + 2048;
    // (name, byte offset of the name, parameter index holding the handle) — in record order.
    let caps = [
        ("extra", 0u64, 2u32),
        ("exit", 8, 3),
        ("stdout", 16, 4),
        ("fs", 24, 5),
    ];
    let mut data = String::new();
    let mut body = String::new();
    for (i, (name, name_rel, param)) in caps.iter().enumerate() {
        let name_off = name_base + name_rel;
        let rec_off = rec_base + (i as u64) * 16;
        let w0 = name_off | ((name.len() as u64) << 32);
        data.push_str(&format!("data {name_off} \"{name}\"\n"));
        body.push_str(&format!(
            "  w{i} = i64.const {w0}\n  o{i} = i64.const {rec_off}\n  i64.store o{i} w{i}\n\
             \x20 h{i} = i64.extend_i32_u v{param}\n  p{i} = i64.const {}\n  i64.store p{i} h{i}\n",
            rec_off + 8
        ));
    }
    format!(
        r#"memory {parent_sl}
{data}func (i32, i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i32) {{
{body}  vmh = i64.extend_i32_u v1
  vgptr = i64.const {rec_base}
  vgn = i64.const 4
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {child_sl}
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#
    )
}

/// The production granted-spawn hook table, as `nifler_child_jit` and `rust_guest_op13` install it.
fn grant_hooks(host: *mut Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(temen_run::CapCtx::Raw(host))
}

/// A four-record grant list marshaled by a parent running on **emitted code**, with the two caps the
/// child uses sitting at records 2 and 3 so the stride and name packing both have to be right.
#[test]
fn multi_record_grant_list_marshals_on_the_jit() {
    let child = temen_text::parse_module(CHILD).expect("parse child");
    temen_verify::verify_module(&child).expect("child verifies");
    let child_sl = 16u32;
    let carve_off = 1u64 << child_sl;
    let parent = temen_text::parse_module(&parent_src(child_sl, carve_off)).expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(vec![], vec![]);
    let factory = Arc::new(factory);

    let mut host = Host::new();
    let fs_init: HostProc = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| ForkedProc::shared((*f)()))
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork);
    let sink = host.shared_stdout();
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();
    // The spare: a second stream, offered in record 0 and never resolved by the child.
    let extra_h = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, 1u64 << (child_sl + 1));
    let modh = host.grant_module(&child);

    let args = [
        inst as i64,
        modh as i64,
        extra_h as i64,
        exit_h as i64,
        stdout_h as i64,
        fs_h as i64,
    ];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(&mut host as *mut Host)),
    )
    .expect("jit run");
    let joined = match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        JitOutcome::Exited(c) => c as i64,
        ref o => panic!("jit ended abnormally: {o:?}"),
    };
    assert_eq!(
        joined,
        FILE_BODY.len() as i64,
        "the child joined back with the byte count its granted `fs` write returned"
    );

    // Record 3 (`fs`): the file the child wrote through the re-granted memfs.
    let (files, _dirs) = handle.seed();
    let emitted = files
        .into_iter()
        .find(|(k, _)| k == "out.bin")
        .map(|(_, b)| b)
        .expect("the child wrote no out.bin — record 3 (`fs`) did not resolve on the JIT");
    assert_eq!(emitted, FILE_BODY.as_bytes());

    // Record 2 (`stdout`): the bytes the child put on the re-granted stream.
    let streamed = sink.lock().unwrap().clone();
    assert_eq!(
        String::from_utf8_lossy(&streamed),
        STREAM_BODY,
        "the child wrote nothing to `stdout` — record 2 did not resolve on the JIT"
    );
}
