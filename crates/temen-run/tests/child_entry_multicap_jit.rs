//! **Multi-record grant marshaling on the emitted tier** (#1221, NIM.md §3c).
//!
//! The front-end drivers hand a §14 child a **four**-record grant list — `{fs, stdout, exit, exec}` —
//! through [`temen_run::conductor`]. `nifler_child_asset.rs` covers that on the **tree-walker**;
//! `nifler_child_jit.rs` covers it on the **JIT** with the real nifler, but it is `#[ignore]`d:
//! Cranelift-compiling nifler's 100+ funcs takes ~250 s in debug. This is the cheap per-PR stand-in:
//! two small text-IR modules, no asset, no toolchain, a child the JIT compiles in well under a second.
//!
//! The sensitivity comes from *where* the used caps sit. The grant list is ordered
//! `[extra, exit, stdout, fs]`, so the two caps the child actually exercises are the **last two**
//! records — index 2 and index 3. A conductor or spawn that miscomputes the 16-byte record stride, or
//! the name-offset/length packing in a record's first word, resolves them to the wrong handle or to
//! nothing at all, and the child fails to write. `extra` is offered and ignored, mirroring the spare
//! record in `nifler_child_asset.rs`'s four-cap list.

use std::ffi::c_void;
use std::sync::Arc;

use temen_interp::{ForkedProc, Host, HostProcFork, StreamRole};
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
    let parent = temen_run::conductor(&["extra", "exit", "stdout", "fs"], &[]);

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(vec![], vec![]);
    let factory = Arc::new(factory);

    let mut host = Host::new();
    let (fs_init, fs_init_state) = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| {
            let (h, s) = (*f)();
            ForkedProc::shared(h, s)
        })
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork, fs_init_state);
    let sink = host.shared_stdout();
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();
    // The spare: a second stream, offered in record 0 and never resolved by the child.
    let extra_h = host.grant_stream(StreamRole::Out);
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);

    let args = [
        inst as i64,
        modh as i64,
        budget as i64,
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
