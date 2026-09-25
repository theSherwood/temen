//! #1646 — a personality **caller request** must be answered identically however the guest reached
//! the op, and on whichever engine runs it.
//!
//! `fork`, `execve` and a blocking `wait4` are not ordinary host calls: the op fires a request into
//! the #799 door and returns a placeholder, and the *drive loop* acts on it. Which means every call
//! form and every engine needs the same answer — and twice now they have not had one:
//!
//! * **#1621** — `call.import`/`call.dyn` drained the request and dropped it, so an import-routed
//!   `fork`/`execve` could not work at all. Invisible to every chibicc guest (those arrive as
//!   `call.cap`/`call.sym`) and fatal to a no-C nim module, whose personality ops are imports.
//! * **#1635** — after #1621 the reap bench was *still* decided per arm: `call.cap`/`call.sym`
//!   parked, the import routes answered `-ECHILD` at once, so `execShellCmd` reported failure for
//!   a command that had in fact run.
//!
//! So the assertion here is not "each row is reasonable" but "**every row is the same**". A new
//! call form that quietly acquires its own answer fails this file.
//!
//! The grant axis is the other half of #1635: the same personality reached through **one** powerbox
//! entry (#1645, what temen-run's nim lane now builds) and through **an entry per op** (what it
//! built before). `Host::fork_powerbox` runs the fork factory once per entry, so the second shape
//! asks for one twin once per op — and every op must still land on one process with one working
//! door (the #1644 net).
//!
//! **Not covered here (1): the interrupt row.** #1647 made `-EINTR` on an interrupted park one
//! decision for all four forms, but exercising it needs a signal delivered *while* the caller is
//! parked. The deterministic idiom for that is `c_posix.rs`'s
//! `c_a_caught_signal_interrupts_a_blocked_capability_read_with_eintr`: a spawned guest thread
//! raises in a loop until the parked caller takes the interrupt, so it retries rather than races.
//! Reproducing it here means a hand-written IR guest with `thread.spawn`, atomics and the signal
//! setup; #1647 tracks it. A timing-based approximation would be a `kind:flaky-ci` issue waiting
//! to happen, so there isn't one.
//!
//! **Not covered here (2):** `call.cap` and `call.import.dyn`, whose handle operand is real rather than
//! vestigial — a `_start`-shaped powerbox run has no way to hand a guest a live `HOST_PROC` handle,
//! and building one by hand would bypass the scheduler these requests need. `call.cap` is covered
//! end-to-end instead by `crates/temen/tests/c_posix.rs`, where chibicc's real shell forks, execs
//! and reaps through it; `call.sym` here is its near neighbour (the same inline arm shape).
//! #1646 tracks closing the two remaining rows.
//!
//! **Pipe parks (#1826)** are the same question for a pipe's blocking read: a read of an empty pipe
//! whose writer is still open must wait for the writer, on every route and every engine. The
//! import routes of the tree-walker used to answer it `0`, a false EOF, and the JIT served no pipe
//! at all. Two rows: the embedder's pipe with its ends bound as imports (the read reached through
//! each form), and a pipe the guest mints itself and reads with a direct `call.cap` on its handle —
//! the shape a nim or chibicc program's `pipe`/`read` take through the tag protocol.

use std::sync::{Arc, Mutex};

use temen_run::{
    instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig, SharedHostProc, Value,
};
use temen_text::parse_module;

/// The command `/bin/c`: a child-entry `(i64) -> (i64)` returning 77. The point is *whether* it
/// runs, not what it does.
const COMMAND: &str = "memory 17\n\
func (i64) -> (i64) {\n\
block 0 (v0: i64) {\n\
  v1 = i64.const 77\n\
  return v1\n\
  }\n\
}\n";

/// #1768 — a command exec cannot start: its entry `(i32) -> (i32)` is no shape a §14 child enters by
/// (`child_entry_ok`), so the ENGINE refuses it after the personality has already resolved the path
/// and raised the request — the refusal only the engine can make.
const UNSTARTABLE: &str = "memory 17\n\
func (i32) -> (i32) {\n\
block 0 (v0: i32) {\n\
  return v0\n\
  }\n\
}\n";

/// #1768 — a command that reports the argv it was exec'd with: its entry returns the `argc` word at
/// `module_args_base` (`16512`), where the exec's commit puts the new image's args blob.
const ARGC_COMMAND: &str = "memory 17\n\
func (i64) -> (i64) {\n\
block 0 (v0: i64) {\n\
  v1 = i64.const 16512\n\
  v2 = i32.load v1\n\
  v3 = i64.extend_i32_u v2\n\
  return v3\n\
  }\n\
}\n";

/// How the guest spells a personality call. The two forms produce byte-identical modules apart from
/// the spelling — `vdummy` is emitted in both so even the value numbering matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// Indexed `call.import <slot> (args)` — a no-C nim module's route.
    Import,
    /// `call.sym <slot> v<handle> (args)` — a chibicc/link-form guest's route. The handle operand
    /// is vestigial under the reference policy.
    Sym,
}

impl Form {
    fn call(self, slot: u32, args: &str) -> String {
        match self {
            Form::Import => format!("call.import {slot} ({args})"),
            Form::Sym => format!("call.sym {slot} vdummy ({args})"),
        }
    }
    fn all() -> [Form; 2] {
        [Form::Import, Form::Sym]
    }
}

/// How the personality occupies the powerbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Grant {
    /// One entry serving every op (#1645) — what `nim_posix_imports` builds.
    OneEntry,
    /// An entry per op over one shared fork factory — the pre-#1645 shape, kept as a row because
    /// the #1644 net is what still makes it behave.
    EntryPerOp,
}

impl Grant {
    fn all() -> [Grant; 2] {
        [Grant::OneEntry, Grant::EntryPerOp]
    }
}

/// What the guest does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Body {
    /// `execve("/bin/c", NULL, NULL)` then `exit(9)`. Reaching `exit` means the request did not
    /// replace the image.
    Exec,
    /// #1768 — `execve("/bin/c", ["x"], NULL)` against a command the engine refuses, then check the
    /// caller is untouched: the op answered `-EINVAL`, and neither the args region the new image would
    /// have read nor the personality's argv changed. `exit(9)` iff all three hold; `exit(1..3)` names
    /// the first that did not.
    ExecRefusedUntouched,
    /// #1768 — `execve("/bin/c", ["x", "x", "x"], NULL)` into [`ARGC_COMMAND`]: the image-replace must
    /// deliver the argv the op staged, so the command returns `3`.
    ExecDeliversArgv,
    /// `fork()`; the child `execve`s and falls back to `exit(9)`; the parent `wait4`s and exits
    /// with the reaped `WEXITSTATUS`. Nim's `execShellCmd` is exactly this shape.
    ForkExecReap,
    /// #763 — `execve("/bin/c", NULL, NULL)`, then `exit` with the errno it answered, negated: a
    /// refusal's *which* is part of the answer every row must agree on.
    ExecErrno,
    /// #1826 — a pipe between fork twins, its ops through the form: the embedder's pipe, its ends
    /// bound as imports. See [`pipe_guest`].
    PipeImported,
    /// #1826 — the same with a pipe the guest mints (`CAP_SELF_PIPE`) and drives with direct
    /// `call.cap`s on the handles the mint wrote.
    PipeMinted,
    /// #1826 — the write side: a writer that fills the pipe waits for its reader to drain it. See
    /// [`backpressure_guest`].
    PipeBackpressure,
}

/// What is at `/bin/c` when the guest execs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cmd {
    /// Nothing: `-ENOENT`.
    Absent,
    /// A registered command: the embedder granted the module and registered the path.
    Registered,
    /// #763 — a program the process built: the module's encoding is a file at the path, and the
    /// process holds a `ModuleLoader`, which promotes it at the exec.
    Built,
    /// The same file, but the process holds no `ModuleLoader`: not executable to it (`-EACCES`).
    BuiltNoLoader,
    /// A file that is not a program: `-EACCES`, as for any file without the exec bit.
    Plain,
    /// A file whose header is a module's but whose body does not decode: `-ENOEXEC`.
    Corrupt,
}

/// The four personality slots every guest declares, in a fixed order so the call indices are
/// stable across bodies.
const IMPORTS: &str = "import 0 \"execve\" (i64, i64, i64) -> (i64)\n\
import 1 \"exit\" (i32) -> ()\n\
import 2 \"fork\" () -> (i64)\n\
import 3 \"wait4\" (i64, i64, i64, i64) -> (i64)\n\
import 4 \"argc\" () -> (i64)\n";

/// #1826 — a pipe between fork twins. The child spins, so its parent is most likely already
/// waiting, then writes `"GO!"` and exits 7. The parent closes its own write end, so the child's
/// copy is the pipe's only writer, and reads twice: the first read must wait for the child's bytes
/// (answering `0` would be a false EOF: the writer is still open), and the second must see EOF once
/// the child's exit released its end. It then reaps the child and exits
/// `10 * n1 + n2 + (b0 - 'G') + (WEXITSTATUS - 7)`: `30` when every step held.
///
/// `minted`: the guest mints the pipe (`call.cap 4294967295 16`, which writes the read end's handle
/// at `44000` and the write end's at `44004`) and drives it with direct `call.cap`s. Otherwise the
/// pipe is the embedder's, its read, write and close bound as imports 5–7 and reached through the
/// form. Either way `fork`, `wait4` and `exit` go through the form.
fn pipe_guest(form: Form, minted: bool) -> String {
    let head = format!(
        "memory 17 shadow 65536 69632\n\n{IMPORTS}\
         import 5 \"pipe_read\" (i64, i64) -> (i64)\n\
         import 6 \"pipe_write\" (i64, i64) -> (i64)\n\
         import 7 \"pipe_close\" () -> (i64)\n\n\
         data 41000 \"\\xab\\xcd\\x00\\x00\"\n\
         data 46000 \"GO!\"\n\n"
    );
    // The ends' handles, reloaded in each block that uses them (values are block-local).
    let ends = "  vfr = i64.const 44000\n  vrh = i32.load vfr\n  vfw = i64.const 44004\n  vwh = i32.load vfw\n";
    let (read, write, close) = if minted {
        (
            "call.cap 0 0 (i64, i64) -> (i64) vrh (vbuf, veight)".to_string(),
            "call.cap 0 1 (i64, i64) -> (i64) vwh (vmsg, vthree)".to_string(),
            "call.cap 0 2 () -> (i64) vwh ()".to_string(),
        )
    } else {
        (
            form.call(5, "vbuf, veight"),
            form.call(6, "vmsg, vthree"),
            form.call(7, ""),
        )
    };
    let mint = if minted {
        "call.cap 4294967295 16 (i64) -> (i64) vdummy (vfds)"
    } else {
        "i64.const 0"
    };
    format!(
        "{head}func () -> () {{\n\
         block 0 () {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vfds = i64.const 44000\n\
         \x20 vm = {mint}\n\
         \x20 vz = i64.const 0\n\
         \x20 vok = i64.eq vm vz\n\
         \x20 br_if vok 1() 5()\n\
         \x20 }}\n\
         block 1 () {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vpid = {fork}\n\
         \x20 vz = i64.const 0\n\
         \x20 veq = i64.eq vpid vz\n\
         \x20 br_if veq 2(vz) 4(vpid)\n\
         \x20 }}\n\
         block 2 (vi: i64) {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vone = i64.const 1\n\
         \x20 vi2 = i64.add vi vone\n\
         \x20 vlim = i64.const 100000\n\
         \x20 vlt = i64.lt_s vi2 vlim\n\
         \x20 br_if vlt 2(vi2) 3()\n\
         \x20 }}\n\
         block 3 () {{\n\
         \x20 vdummy = i32.const 0\n\
         {ends}\
         \x20 vmsg = i64.const 46000\n\
         \x20 vthree = i64.const 3\n\
         \x20 vw = {write}\n\
         \x20 vseven = i32.const 7\n\
         \x20 {exit7}\n\
         \x20 unreachable\n\
         \x20 }}\n\
         block 4 (vkid: i64) {{\n\
         \x20 vdummy = i32.const 0\n\
         {ends}\
         \x20 vc = {close}\n\
         \x20 vbuf = i64.const 45000\n\
         \x20 veight = i64.const 8\n\
         \x20 vn1 = {read}\n\
         \x20 vn2 = {read}\n\
         \x20 vst = i64.const 41000\n\
         \x20 vz2 = i64.const 0\n\
         \x20 vwt = {wait4}\n\
         \x20 vten = i64.const 10\n\
         \x20 vt = i64.mul vn1 vten\n\
         \x20 vn = i64.add vt vn2\n\
         \x20 vn32 = i32.wrap_i64 vn\n\
         \x20 vb0 = i32.load8_u vbuf\n\
         \x20 vg = i32.const 71\n\
         \x20 vdb = i32.sub vb0 vg\n\
         \x20 vhi = i64.const 41001\n\
         \x20 vcode = i32.load8_u vhi\n\
         \x20 vseven = i32.const 7\n\
         \x20 vdc = i32.sub vcode vseven\n\
         \x20 vs1 = i32.add vn32 vdb\n\
         \x20 vs2 = i32.add vs1 vdc\n\
         \x20 {exit_s2}\n\
         \x20 unreachable\n\
         \x20 }}\n\
         block 5 () {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vfail = i32.const 90\n\
         \x20 {exit_fail}\n\
         \x20 unreachable\n\
         \x20 }}\n\
         }}\n\
         export 0 func \"_start\" 0\n",
        fork = form.call(2, ""),
        exit7 = form.call(1, "vseven"),
        wait4 = form.call(3, "vkid, vst, vz2, vz2"),
        exit_s2 = form.call(1, "vs2"),
        exit_fail = form.call(1, "vfail"),
    )
}

/// #1826 — backpressure. The child writes 96 KiB into a minted pipe, 4 KiB a call; the pipe holds
/// 64 KiB, so its seventeenth write must wait for the parent to drain it (a write that answered `0`
/// instead would stop the child at 64 KiB). The parent closes its write end and spins — long
/// enough for the child to fill the pipe first even on the JIT, where the twin's thread must start
/// — then reads until EOF, reaps the child and exits `KiB read + (WEXITSTATUS - 7)`: `96` when every
/// step held. The child exits 7 once all of it is written, 8 if a write failed.
fn backpressure_guest(form: Form) -> String {
    let head = format!(
        "memory 17 shadow 65536 69632\n\n{IMPORTS}\n\
         data 41000 \"\\xab\\xcd\\x00\\x00\"\n\n"
    );
    format!(
        "{head}func () -> () {{\n\
         block 0 () {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vfds = i64.const 44000\n\
         \x20 vm = call.cap 4294967295 16 (i64) -> (i64) vdummy (vfds)\n\
         \x20 vz = i64.const 0\n\
         \x20 vok = i64.eq vm vz\n\
         \x20 br_if vok 1() 7()\n\
         \x20 }}\n\
         block 1 () {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vpid = {fork}\n\
         \x20 vz = i64.const 0\n\
         \x20 veq = i64.eq vpid vz\n\
         \x20 br_if veq 2(vz) 4(vpid)\n\
         \x20 }}\n\
         block 2 (vsent: i64) {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vfw = i64.const 44004\n\
         \x20 vwh = i32.load vfw\n\
         \x20 vtotal = i64.const 98304\n\
         \x20 vleft = i64.sub vtotal vsent\n\
         \x20 vchunk = i64.const 4096\n\
         \x20 vsmall = i64.lt_s vleft vchunk\n\
         \x20 vlen = select vsmall vleft vchunk\n\
         \x20 vbuf = i64.const 47000\n\
         \x20 vn = call.cap 0 1 (i64, i64) -> (i64) vwh (vbuf, vlen)\n\
         \x20 vz = i64.const 0\n\
         \x20 vgood = i64.gt_s vn vz\n\
         \x20 br_if vgood 3(vsent, vn) 6()\n\
         \x20 }}\n\
         block 3 (vs: i64, vn: i64) {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vs2 = i64.add vs vn\n\
         \x20 vtotal = i64.const 98304\n\
         \x20 vmore = i64.lt_s vs2 vtotal\n\
         \x20 br_if vmore 2(vs2) 5()\n\
         \x20 }}\n\
         block 4 (vkid: i64) {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vfw = i64.const 44004\n\
         \x20 vwh = i32.load vfw\n\
         \x20 vc = call.cap 0 2 () -> (i64) vwh ()\n\
         \x20 vz = i64.const 0\n\
         \x20 br 8(vkid, vz)\n\
         \x20 }}\n\
         block 5 () {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vseven = i32.const 7\n\
         \x20 {exit7}\n\
         \x20 unreachable\n\
         \x20 }}\n\
         block 6 () {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 veight = i32.const 8\n\
         \x20 {exit8}\n\
         \x20 unreachable\n\
         \x20 }}\n\
         block 7 () {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vfail = i32.const 90\n\
         \x20 {exit_fail}\n\
         \x20 unreachable\n\
         \x20 }}\n\
         block 8 (vkid: i64, vi: i64) {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vone = i64.const 1\n\
         \x20 vi2 = i64.add vi vone\n\
         \x20 vlim = i64.const 2000000\n\
         \x20 vlt = i64.lt_s vi2 vlim\n\
         \x20 vz = i64.const 0\n\
         \x20 br_if vlt 8(vkid, vi2) 9(vkid, vz)\n\
         \x20 }}\n\
         block 9 (vkid: i64, vgot: i64) {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vfr = i64.const 44000\n\
         \x20 vrh = i32.load vfr\n\
         \x20 vbuf = i64.const 52000\n\
         \x20 vcap = i64.const 8192\n\
         \x20 vn = call.cap 0 0 (i64, i64) -> (i64) vrh (vbuf, vcap)\n\
         \x20 vz = i64.const 0\n\
         \x20 vmore = i64.gt_s vn vz\n\
         \x20 vgot2 = i64.add vgot vn\n\
         \x20 br_if vmore 9(vkid, vgot2) 10(vkid, vgot, vn)\n\
         \x20 }}\n\
         block 10 (vkid: i64, vgot: i64, vlast: i64) {{\n\
         \x20 vdummy = i32.const 0\n\
         \x20 vst = i64.const 41000\n\
         \x20 vz2 = i64.const 0\n\
         \x20 vwt = {wait4}\n\
         \x20 vk = i64.const 1024\n\
         \x20 vkib = i64.div_s vgot vk\n\
         \x20 vkib32 = i32.wrap_i64 vkib\n\
         \x20 vlast32 = i32.wrap_i64 vlast\n\
         \x20 vhi = i64.const 41001\n\
         \x20 vcode = i32.load8_u vhi\n\
         \x20 vseven = i32.const 7\n\
         \x20 vdc = i32.sub vcode vseven\n\
         \x20 vs1 = i32.add vkib32 vlast32\n\
         \x20 vs2 = i32.add vs1 vdc\n\
         \x20 {exit_s2}\n\
         \x20 unreachable\n\
         \x20 }}\n\
         }}\n\
         export 0 func \"_start\" 0\n",
        fork = form.call(2, ""),
        exit7 = form.call(1, "vseven"),
        exit8 = form.call(1, "veight"),
        exit_fail = form.call(1, "vfail"),
        wait4 = form.call(3, "vkid, vst, vz2, vz2"),
        exit_s2 = form.call(1, "vs2"),
    )
}

fn guest(form: Form, body: Body) -> String {
    // The shadow arena is where the Cranelift JIT unwinds a forking caller to (FORK.md §9.5); its
    // placement is the module's (INVARIANTS.md #16), clear of the data below. The interpreters
    // fork without one, so it changes nothing there.
    let head = format!(
        "memory 17 shadow 65536 69632\n\n{IMPORTS}\n\
         data 40000 \"/bin/c\\x00\"\n\
         data 41000 \"\\xab\\xcd\\x00\\x00\"\n\n"
    );
    // `41000` starts as a marker, so "wait4 never wrote the status" is distinguishable from
    // "wait4 reported 0" — the difference between a dropped reap bench and a reaped exit.
    let exec = format!(
        "  vp = i64.const 40000\n\
         \x20 vz1 = i64.const 0\n\
         \x20 vr = {}\n\
         \x20 vnine = i32.const 9\n\
         \x20 {}\n\
         \x20 unreachable\n",
        form.call(0, "vp, vz1, vz1"),
        form.call(1, "vnine"),
    );
    match body {
        // `43000` is argv `["x", NULL]` (`42000` is "x"); `16512` is `module_args_base`, where the
        // blob an exec commits lands — its first word is the new image's argc.
        Body::ExecRefusedUntouched => format!(
            "{head}data 42000 \"x\\x00\"\n\
             data 43000 \"\\x10\\xa4\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\n\
             func () -> () {{\n\
             block 0 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vab = i64.const 16512\n\
             \x20 va0 = i32.load vab\n\
             \x20 vn0 = {}\n\
             \x20 vp = i64.const 40000\n\
             \x20 vargv = i64.const 43000\n\
             \x20 vz = i64.const 0\n\
             \x20 vr = {}\n\
             \x20 va1 = i32.load vab\n\
             \x20 vn1 = {}\n\
             \x20 vinval = i64.const -22\n\
             \x20 vok = i64.eq vr vinval\n\
             \x20 br_if vok 1(va0, va1, vn0, vn1) 4()\n\
             \x20 }}\n\
             block 1 (wa0: i32, wa1: i32, wn0: i64, wn1: i64) {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vok = i32.eq wa0 wa1\n\
             \x20 br_if vok 2(wn0, wn1) 5()\n\
             \x20 }}\n\
             block 2 (xn0: i64, xn1: i64) {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vok = i64.eq xn0 xn1\n\
             \x20 br_if vok 3() 6()\n\
             \x20 }}\n\
             block 3 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vc = i32.const 9\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             block 4 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vc = i32.const 1\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             block 5 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vc = i32.const 2\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             block 6 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vc = i32.const 3\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             }}\n\
             export 0 func \"_start\" 0\n",
            form.call(4, ""),
            form.call(0, "vp, vargv, vz"),
            form.call(4, ""),
            form.call(1, "vc"),
            form.call(1, "vc"),
            form.call(1, "vc"),
            form.call(1, "vc"),
        ),
        // `43000` is argv `["x", "x", "x", NULL]` (`42000` is "x").
        Body::ExecDeliversArgv => format!(
            "{head}data 42000 \"x\\x00\"\n\
             data 43000 \"\\x10\\xa4\\x00\\x00\\x00\\x00\\x00\\x00\\x10\\xa4\\x00\\x00\\x00\\x00\\x00\\x00\\x10\\xa4\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\n\
             func () -> () {{\n\
             block 0 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vp = i64.const 40000\n\
             \x20 vargv = i64.const 43000\n\
             \x20 vz = i64.const 0\n\
             \x20 vr = {}\n\
             \x20 vnine = i32.const 9\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             }}\n\
             export 0 func \"_start\" 0\n",
            form.call(0, "vp, vargv, vz"),
            form.call(1, "vnine"),
        ),
        Body::ExecErrno => format!(
            "{head}func () -> () {{\n\
             block 0 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vp = i64.const 40000\n\
             \x20 vz = i64.const 0\n\
             \x20 vr = {}\n\
             \x20 vneg = i64.sub vz vr\n\
             \x20 vc = i32.wrap_i64 vneg\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             }}\n\
             export 0 func \"_start\" 0\n",
            form.call(0, "vp, vz, vz"),
            form.call(1, "vc"),
        ),
        Body::Exec => format!(
            "{head}func () -> () {{\n\
             block 0 () {{\n\
             \x20 vdummy = i32.const 0\n\
             {exec}  }}\n\
             }}\n\
             export 0 func \"_start\" 0\n"
        ),
        Body::ForkExecReap => format!(
            "{head}func () -> () {{\n\
             block 0 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vpid = {}\n\
             \x20 vz = i64.const 0\n\
             \x20 veq = i64.eq vpid vz\n\
             \x20 br_if veq 1() 2(vpid)\n\
             \x20 }}\n\
             block 1 () {{\n\
             \x20 vdummy = i32.const 0\n\
             {exec}  }}\n\
             block 2 (vkid: i64) {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vst = i64.const 41000\n\
             \x20 vz2 = i64.const 0\n\
             \x20 vw = {}\n\
             \x20 vhi = i64.const 41001\n\
             \x20 vcode = i32.load8_u vhi\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             }}\n\
             export 0 func \"_start\" 0\n",
            form.call(2, ""),
            form.call(3, "vkid, vst, vz2, vz2"),
            form.call(1, "vcode"),
        ),
        Body::PipeImported => pipe_guest(form, false),
        Body::PipeMinted => pipe_guest(form, true),
        Body::PipeBackpressure => backpressure_guest(form),
    }
}

/// Run one cell of the table. `cmd` decides what `/bin/c` is, which is the difference between an
/// exec that replaces the image and one that is refused.
fn run(form: Form, grant: Grant, body: Body, backend: Backend, cmd: Cmd) -> Outcome {
    let caller = parse_module(&guest(form, body)).expect("parse caller");
    let command = parse_module(match body {
        Body::ExecRefusedUntouched => UNSTARTABLE,
        Body::ExecDeliversArgv => ARGC_COMMAND,
        _ => COMMAND,
    })
    .expect("parse command");
    let cmd_wl = command.memory.expect("command window").size_log2;

    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let image = temen_encode::encode_module(&command);
    match cmd {
        Cmd::Built | Cmd::BuiltNoLoader => posix.write_file("/bin/c", &image),
        Cmd::Plain => posix.write_file("/bin/c", b"echo not a program\n"),
        // The header and the declared window a module's, the rest cut short.
        Cmd::Corrupt => posix.write_file("/bin/c", &image[..image.len() / 2]),
        Cmd::Absent | Cmd::Registered => {}
    }
    let make: Arc<dyn Fn() -> temen_interp::HostProc + Send + Sync> = Arc::new(make);
    // #1826 — the embedder's pipe ([`Body::PipeImported`]): minted once per run by whichever of its
    // three imports binds first, each import bound to its end's op.
    let pipe: Arc<Mutex<Option<(i32, i32)>>> = Arc::default();
    let pipe_end = |write: bool, op: u32| {
        let pipe = Arc::clone(&pipe);
        HostCap::custom(temen_interp::cap_id::STREAM, op, move |h, _| {
            let (w, r) = *pipe
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_or_insert_with(|| h.grant_pipe());
            if write {
                w
            } else {
                r
            }
        })
    };
    let slot = SharedHostProc::new(
        {
            let make = Arc::clone(&make);
            move || (*make)()
        },
        temen_posix::cap_fork_factory(&posix),
    );
    let per_op_fork = temen_posix::cap_fork_factory(&posix);
    let cap = |op: u32| match grant {
        Grant::OneEntry => HostCap::host_proc_shared(op, &slot),
        Grant::EntryPerOp => {
            let make = Arc::clone(&make);
            HostCap::host_proc_forkable(op, move || (*make)(), Arc::clone(&per_op_fork))
        }
    };
    let imports = Imports::new()
        .provide("execve", cap(temen_posix::OP_EXECVE))
        .provide("exit", HostCap::exit())
        .provide("fork", cap(temen_posix::OP_FORK))
        .provide("wait4", cap(temen_posix::OP_WAIT4))
        .provide("argc", cap(temen_posix::OP_ARGC))
        .provide("pipe_read", pipe_end(false, 0))
        .provide("pipe_write", pipe_end(true, 1))
        .provide("pipe_close", pipe_end(true, 2));
    let inst = instantiate_with_imports(caller, imports).expect("instantiate");

    // The process doors, as every process-shaped lane installs them: without the signal source
    // there is no #799 caller-request door at all and the ops cannot even ask.
    let p = posix.clone();
    let slot_for_setup = slot.clone();
    let mut setup = move |host: &mut temen_interp::Host| {
        let handle = slot_for_setup.install(host);
        let (door, armed) = temen_posix::cap_signal_source(&p);
        host.set_signal_source(door, armed);
        host.push_exec_remap_hook(temen_posix::cap_exec_remap_hook(&p));
        let (names, sigs) = temen_posix::cap_vtable();
        host.set_host_proc_vtable(handle, names, sigs);
        if cmd == Cmd::Registered {
            let h = host.grant_module(&command);
            p.register_executable("/bin/c", h, cmd_wl);
        }
        if matches!(cmd, Cmd::Built | Cmd::Corrupt) {
            temen_run::grant_module_loader(host);
        }
    };
    inst.run_with_caps_and_host(backend, &RunConfig::default(), &[], Some(&mut setup))
        .expect("run")
        .outcome
}

/// Every (form, grant, engine) cell of one behaviour must produce `want`. The message names the
/// cell, because "which row disagreed" is the whole diagnostic.
fn assert_parity(body: Body, cmd: Cmd, want: Outcome) {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        for grant in Grant::all() {
            for form in Form::all() {
                let got = run(form, grant, body, backend, cmd);
                assert_eq!(
                    got, want,
                    "{body:?} ({cmd:?}) disagreed at {form:?} / {grant:?} / {backend:?}"
                );
            }
        }
    }
}

/// #1621 — `execve` replaces the image whichever way the op was reached. Before that fix the
/// import row exited 9 (request discarded) while the `call.sym` row returned 77.
#[test]
fn execve_replaces_the_image_identically_on_every_route() {
    assert_parity(
        Body::Exec,
        Cmd::Registered,
        Outcome::Returned(vec![Value::I64(77)]),
    );
}

/// POSIX: `execve` returns only on failure. An unregistered path must leave the caller running with
/// a probeable errno on every route — a route that *replaced* the image here would be handing out
/// an image nobody granted.
#[test]
fn a_refused_execve_leaves_the_caller_running_on_every_route() {
    assert_parity(Body::Exec, Cmd::Absent, Outcome::Exited(9));
}

/// #1768 — the refusal only the engine can make (the command's entry is no shape exec can start),
/// reached after the personality resolved the path and raised the request. POSIX: a failed `execve`
/// returns to an unchanged caller — the argv the op staged must not have reached the caller's args
/// region or the personality's argv. Before the fix both were overwritten by the time the engine
/// refused (`exit(2)`: the args region's argc had become 1).
#[test]
fn an_execve_the_engine_refuses_leaves_the_caller_untouched_on_every_route() {
    assert_parity(
        Body::ExecRefusedUntouched,
        Cmd::Registered,
        Outcome::Exited(9),
    );
}

/// #1768 — the new image reads the argv its `execve` passed. On the JIT this is a different road
/// from the interpreters' (the image runs in a fresh window seeded from the commit, not the caller's
/// window reused in place), so it is a row of its own.
#[test]
fn an_execd_image_reads_the_argv_it_was_given_on_every_route() {
    assert_parity(
        Body::ExecDeliversArgv,
        Cmd::Registered,
        Outcome::Returned(vec![Value::I64(3)]),
    );
}

/// #1635 — nim's `execShellCmd` shape. The twin must get its own process and its own door (so its
/// `execve` fires into its own cell rather than the parent's), and the parent's blocking `wait4`
/// must **bench** rather than answer `-ECHILD`. Before the fix this row was `Exited(205)` (the
/// untouched status marker — the reap was never serviced) on the import routes.
#[test]
fn a_fork_twin_execs_and_the_parent_reaps_identically_on_every_route() {
    assert_parity(Body::ForkExecReap, Cmd::Registered, Outcome::Exited(77));
}

/// #763 — PROCESS.md's `cc x.c && ./a.out`: a program the process built is a file holding a
/// module's encoding, and the process's `ModuleLoader` promotes it at the exec. It replaces the
/// image as a registered command does, however the op was reached and on every engine.
#[test]
fn a_built_program_execs_like_a_registered_command_on_every_route() {
    assert_parity(
        Body::Exec,
        Cmd::Built,
        Outcome::Returned(vec![Value::I64(77)]),
    );
}

/// #763 — nimony's compile-time evaluation: a fork twin execs the program the build wrote and the
/// parent reaps it. The twin holds the loader its parent does (the fork clones the powerbox).
#[test]
fn a_fork_twin_execs_a_built_program_and_the_parent_reaps_it_on_every_route() {
    assert_parity(Body::ForkExecReap, Cmd::Built, Outcome::Exited(77));
}

/// Every refusal answers its own errno, identically on every route: nothing at the path is
/// `-ENOENT`; a file that is not a program, or a program the process may not load, `-EACCES`; a
/// file with a module's header whose body does not decode, `-ENOEXEC`.
#[test]
fn a_refused_execve_answers_its_errno_on_every_route() {
    for (cmd, errno) in [
        (Cmd::Absent, 2),
        (Cmd::Plain, 13),
        (Cmd::BuiltNoLoader, 13),
        (Cmd::Corrupt, 8),
    ] {
        assert_parity(Body::ExecErrno, cmd, Outcome::Exited(errno));
    }
}

/// #1826 — a pipe between fork twins, its read, write and close reached through each call form. The
/// parent's first read finds the pipe empty with the child's write end still open, so it must wait
/// for the child's bytes. Before the fix the tree-walker's import route answered that read `0`, a
/// false EOF (the bytecode engine parked there all along), and the JIT served no pipe at all: this
/// row was `Exited(0)` there (every read `0`).
#[test]
fn a_pipe_read_waits_for_a_twins_write_on_every_route() {
    assert_parity(Body::PipeImported, Cmd::Absent, Outcome::Exited(30));
}

/// #1826 — the pipe a program mints for itself (`pipe(2)` through nim's or chibicc's libc), read with
/// a direct `call.cap` on its handle: the mint and its parks are served by every engine's process
/// runs. The JIT answered the mint `-EINVAL` before, as a tier with no pipe parks must (`Exited(90)`).
#[test]
fn a_minted_pipe_carries_a_twins_output_on_every_route() {
    assert_parity(Body::PipeMinted, Cmd::Absent, Outcome::Exited(30));
}

/// #1826 — the write side: a writer that fills the pipe waits for the reader to drain it, on every
/// route and engine. A write that answered `0` instead of waiting would stop the child at 64 KiB
/// (`Exited(65)`: 64 KiB read, and the child's failed-write exit 8).
#[test]
fn a_pipe_write_waits_for_its_reader_to_drain_on_every_route() {
    assert_parity(Body::PipeBackpressure, Cmd::Absent, Outcome::Exited(96));
}
