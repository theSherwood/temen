//! #1896 — a process tree's **leaf processes** run whole on the emitted tier. The cooperative driver
//! keeps the tree: it forks, execs and waits on the interpreter, and an exec'd image that cannot park
//! tiers up at its entry. The host runs it and delivers how it ended. This test stands in for the
//! browser host without any wasm, as `coop_tierup.rs` does: it serves the tier-up by bouncing the
//! entry ([`CoopRun::bounce`], the nested interpretation an emitted image's own cap calls bounce
//! into), and checks that the tree ends exactly as it does interpreted.

use std::sync::{Arc, Mutex};

use temen_interp::bytecode::{CoopEvent, CoopRun, LeafOffer, TierUpConfig};
use temen_interp::{BoundImport, Host, Trap};

/// `/bin/leaf`: writes `leaf` to `out.txt`, then exits 5. Its ops are file ops, so it cannot park.
const LEAF: &str = "memory 17\n\
import 0 \"__px_open\" (i64, i64, i64) -> (i64)\n\
import 1 \"__px_write\" (i64, i64, i64) -> (i64)\n\
import 2 \"__px_close\" (i64) -> (i64)\n\
import 3 \"__px_exit\" (i64) -> ()\n\
data 40000 \"out.txt\"\n\
data 40100 \"leaf\"\n\
func (i64) -> (i64) {\n\
block 0 (vcap: i64) {\n\
  vpath = i64.const 40000\n\
  vplen = i64.const 7\n\
  vflags = i64.const 577\n\
  vfd = call.import 0 (vpath, vplen, vflags)\n\
  vbuf = i64.const 40100\n\
  vn = i64.const 4\n\
  vw = call.import 1 (vfd, vbuf, vn)\n\
  vc = call.import 2 (vfd)\n\
  vfive = i64.const 5\n\
  call.import 3 (vfive)\n\
  unreachable\n\
  }\n\
}\n";

/// `/bin/leaf` that loads through a null pointer, then exits 7.
const NULL_LEAF: &str = "memory 17\n\
import 0 \"__px_exit\" (i64) -> ()\n\
func (i64) -> (i64) {\n\
block 0 (vcap: i64) {\n\
  vnull = i64.const 8\n\
  vx = i64.load vnull\n\
  vseven = i64.const 7\n\
  call.import 0 (vseven)\n\
  unreachable\n\
  }\n\
}\n";

/// Forks; the child execs `/bin/leaf` (exiting 9 if it could not), and the parent exits with the
/// child's exit status. With `pipe`, it first makes a pipe, which the child inherits.
fn guest(pipe: bool) -> String {
    let make_pipe = if pipe {
        "  vfds = i64.const 42000\n  vpp = call.import 4 (vfds)\n"
    } else {
        ""
    };
    format!(
        "memory 17\n\
import 0 \"__px_execve\" (i64, i64, i64) -> (i64)\n\
import 1 \"__px_exit\" (i64) -> ()\n\
import 2 \"__px_fork\" () -> (i64)\n\
import 3 \"__px_wait4\" (i64, i64, i64, i64) -> (i64)\n\
import 4 \"__px_pipe\" (i64) -> (i64)\n\
data 40000 \"/bin/leaf\\x00\"\n\
func () -> () {{\n\
block 0 () {{\n\
{make_pipe}\
  vpid = call.import 2 ()\n\
  vz = i64.const 0\n\
  vchild = i64.eq vpid vz\n\
  br_if vchild 1() 2(vpid)\n\
  }}\n\
block 1 () {{\n\
  vp = i64.const 40000\n\
  vz = i64.const 0\n\
  vr = call.import 0 (vp, vz, vz)\n\
  vnine = i64.const 9\n\
  call.import 1 (vnine)\n\
  unreachable\n\
  }}\n\
block 2 (xpid: i64) {{\n\
  vst = i64.const 41000\n\
  vz = i64.const 0\n\
  vw = call.import 3 (xpid, vst, vz, vz)\n\
  vhi = i64.const 41001\n\
  vsw = i32.load8_u vhi\n\
  vs = i64.extend_i32_u vsw\n\
  call.import 1 (vs)\n\
  unreachable\n\
  }}\n\
}}\n\
export 0 func \"_start\" 0\n"
    )
}

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// How the tree ended: the root's exit, and what `/bin/leaf` wrote.
#[derive(Debug, PartialEq)]
struct Ending {
    root: Result<i32, String>,
    wrote: Option<Vec<u8>>,
}

/// What the engine asked of the host: the `(paged, parks)` of each image it offered to emit, the
/// program index of each process that tiered up at its entry, and how many parked calls it resumed.
#[derive(Debug, Default)]
struct Leaves {
    offered: Vec<(bool, bool)>,
    tierups: Vec<u32>,
    resumes: usize,
}

/// Run the tree with `/bin/leaf` as `command`, emitting leaf images iff `leaf`. Serves each entry
/// tier-up by bouncing the entry, the stand-in for running it emitted. When that bounce parks, the
/// stand-in's suspended frames are the bounce itself, so a resume's results are the entry's.
fn run_tree(guest_text: &str, command: &str, leaf: bool) -> (Ending, Leaves) {
    let guest = module(guest_text);
    let mut host = Host::new();
    let (px, posix) = temen_posix::grant(&mut host, 0, 0, Vec::new());
    let binds = guest
        .imports
        .iter()
        .map(|i| {
            let c = temen_posix::resolve_import(&i.name).expect("a personality op");
            BoundImport::required(c.type_id, c.op, px)
        })
        .collect();
    host.set_import_bindings(binds);
    let command = host.grant_module(&module(command));
    posix.register_executable("/bin/leaf", command, 17);
    posix.set_cwd("/w");
    let offered = Arc::new(Mutex::new(Vec::new()));
    let offers = Arc::clone(&offered);
    let tierup = leaf.then(|| TierUpConfig {
        eligible: Arc::from([]),
        page_checked: false,
        leaf: Some(Arc::new(move |o: &LeafOffer| {
            offers.lock().unwrap().push((o.paged, o.parks));
            true
        })),
    });
    let mut run = CoopRun::new_reserved(
        &guest,
        0,
        &[],
        u64::MAX,
        host,
        tierup,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
    )
    .expect("the bytecode engine runs it")
    .expect("it starts");
    let (mut tierups, mut resumes) = (Vec::new(), 0);
    let end = loop {
        match run.run() {
            CoopEvent::TierUp {
                module, func, argv, ..
            } => {
                tierups.push(module);
                let mut io = argv.to_vec();
                io.resize(io.len().max(1), 0);
                match run.bounce(func, &mut io, None) {
                    Ok(Some(n)) => run.deliver_tierup(&io[..n]),
                    Ok(None) => {}
                    Err(t) => run.deliver_tierup_trap(t),
                }
            }
            CoopEvent::Resume { results } => {
                resumes += 1;
                run.deliver_tierup(&results);
            }
            ev => break ev,
        }
    };
    let root = match end {
        CoopEvent::Trapped(Trap::Exit(s)) => Ok(s),
        CoopEvent::Trapped(t) => Err(format!("trapped {t:?}")),
        CoopEvent::Done(_) => Err("returned".to_string()),
        _ => Err("paused".to_string()),
    };
    let wrote = posix.read_file("/w/out.txt");
    let offered = offered.lock().unwrap().clone();
    let leaves = Leaves {
        offered,
        tierups,
        resumes,
    };
    (Ending { root, wrote }, leaves)
}

#[test]
fn a_leaf_process_runs_whole_where_it_tiers_up_and_ends_as_interpreted() {
    let (interpreted, none) = run_tree(&guest(false), LEAF, false);
    assert_eq!(
        interpreted,
        Ending {
            root: Ok(5),
            wrote: Some(b"leaf".to_vec()),
        },
        "the tree, interpreted: the child wrote its file and exited 5, and the parent exited with it"
    );
    assert!(none.offered.is_empty() && none.tierups.is_empty());
    let (emitted, leaves) = run_tree(&guest(false), LEAF, true);
    assert_eq!(
        emitted, interpreted,
        "the leaf ran at its entry and ended the same way"
    );
    assert_eq!(
        leaves.offered,
        [(false, false)],
        "offered once, unpaged, not parking: it cannot change its pages or park"
    );
    assert!(
        matches!(leaves.tierups[..], [m] if m != 0),
        "one tier-up, at the exec'd image's entry, naming its program, not the root's: {leaves:?}"
    );
}

/// An image bound to an address-space op that changes page state (`vm_map`) is offered paged: a
/// map that leaves a hole takes its window past one bound, where a whole run cannot decline.
#[test]
fn a_leaf_that_can_map_pages_is_offered_paged() {
    let maps = LEAF.replacen(
        "import 3 \"__px_exit\" (i64) -> ()\n",
        "import 3 \"__px_exit\" (i64) -> ()\nimport 4 \"vm_map\" (i64, i64, i64) -> (i64)\n",
        1,
    );
    let (interpreted, _) = run_tree(&guest(false), &maps, false);
    let (emitted, leaves) = run_tree(&guest(false), &maps, true);
    assert_eq!(emitted, interpreted);
    assert_eq!(interpreted.root, Ok(5));
    assert_eq!(leaves.offered, [(true, false)], "offered once, paged");
    assert_eq!(leaves.tierups.len(), 1);
}

/// A leaf that holds one of the personality's own pipe ends may park on it, as the personality
/// answers, and an image that imports `fork` may park in it: both run interpreted, and end as they
/// would anyway. (A core pipe end parks only in a stream call, where a host that suspends can serve
/// it: [`a_leaf_that_parks_on_a_pipe_pauses_and_resumes`].)
#[test]
fn an_image_that_can_park_runs_interpreted() {
    let (interpreted, _) = run_tree(&guest(true), LEAF, false);
    let (declined, leaves) = run_tree(&guest(true), LEAF, true);
    assert_eq!(declined, interpreted);
    assert!(
        leaves.offered.is_empty() && leaves.tierups.is_empty(),
        "a child holding a pipe end is not offered"
    );

    let forks = LEAF.replacen(
        "import 3 \"__px_exit\" (i64) -> ()\n",
        "import 3 \"__px_exit\" (i64) -> ()\nimport 4 \"__px_fork\" (i64) -> (i64)\n",
        1,
    );
    let (interpreted, _) = run_tree(&guest(false), &forks, false);
    let (declined, leaves) = run_tree(&guest(false), &forks, true);
    assert_eq!(declined, interpreted);
    assert_eq!(interpreted.root, Ok(5));
    assert!(
        leaves.offered.is_empty() && leaves.tierups.is_empty(),
        "an image that imports fork is not offered"
    );
}

/// An exec'd image starts behind the NULL guard, as a freshly loaded one does (#1094): a null
/// dereference crashes the process, which its parent reaps as 128, where it once read zeros and
/// exited 7. The emitted tier bakes the guard, so this is also what makes a leaf end the same way
/// emitted as interpreted.
#[test]
fn an_execd_image_starts_behind_the_null_guard() {
    let (interpreted, _) = run_tree(&guest(false), NULL_LEAF, false);
    assert_eq!(
        interpreted,
        Ending {
            root: Ok(128),
            wrote: None,
        },
        "the child crashed on the null load"
    );
    let (emitted, leaves) = run_tree(&guest(false), NULL_LEAF, true);
    assert_eq!(emitted, interpreted);
    assert_eq!(leaves.tierups.len(), 1);
}

/// The C shim's `read`/`write` over a core pipe end, as IR (`c_posix.rs`'s `PIPE_SHIM`): the
/// personality op answers a core pipe fd with a redirect tag naming its handle, and the stream call
/// on that handle moves the bytes, parking on an empty or full pipe. `read`/`write` are the
/// personality imports' indices; the two helpers, read then write, follow a module's own functions.
fn shim(read: u32, write: u32) -> String {
    let op = |import: u32, stream_op: u32| {
        format!(
            "func (i64, i64, i64) -> (i64) {{\n\
block 0 (vfd: i64, vbuf: i64, vlen: i64) {{\n\
  vr = call.import {import} (vfd, vbuf, vlen)\n\
  vlim = i64.const -1048576\n\
  vtag = i64.le_s vr vlim\n\
  br_if vtag 1(vr, vbuf, vlen) 2(vr)\n\
  }}\n\
block 1 (xr: i64, xbuf: i64, xlen: i64) {{\n\
  vbase = i64.const 1048576\n\
  vsum = i64.add xr vbase\n\
  vz = i64.const 0\n\
  vh64 = i64.sub vz vsum\n\
  vh = i32.wrap_i64 vh64\n\
  vn = call.cap 0 {stream_op} (i64, i64) -> (i64) vh (xbuf, xlen)\n\
  return vn\n\
  }}\n\
block 2 (xr: i64) {{\n\
  return xr\n\
  }}\n\
}}\n"
        )
    };
    op(read, 0) + &op(write, 1)
}

/// Forks; the child execs `/bin/leaf`, and the parent answers it over two core pipes, `down` (fds
/// 3, 4) and `up` (fds 5, 6): it reads the child's 4 bytes from `up`, writes `pong` down, and exits
/// with the child's exit status.
fn ping_pong() -> String {
    format!(
        "memory 17\n\
import 0 \"__px_execve\" (i64, i64, i64) -> (i64)\n\
import 1 \"__px_exit\" (i64) -> ()\n\
import 2 \"__px_fork\" () -> (i64)\n\
import 3 \"__px_wait4\" (i64, i64, i64, i64) -> (i64)\n\
import 4 \"__px_pipe_adopt\" (i64, i64, i64) -> (i64)\n\
import 5 \"__px_read\" (i64, i64, i64) -> (i64)\n\
import 6 \"__px_write\" (i64, i64, i64) -> (i64)\n\
data 40000 \"/bin/leaf\\x00\"\n\
data 40100 \"pong\"\n\
func () -> () {{\n\
block 0 () {{\n\
  vh0 = i32.const 0\n\
  vhs = i64.const 42000\n\
  vfds = i64.const 42100\n\
  vpd = call.cap 4294967295 16 (i64) -> (i64) vh0 (vhs)\n\
  vrh32 = i32.load vhs\n\
  vrh = i64.extend_i32_u vrh32\n\
  vwh32 = i32.load vhs offset=4\n\
  vwh = i64.extend_i32_u vwh32\n\
  vad = call.import 4 (vrh, vwh, vfds)\n\
  vpu = call.cap 4294967295 16 (i64) -> (i64) vh0 (vhs)\n\
  vrh32u = i32.load vhs\n\
  vrhu = i64.extend_i32_u vrh32u\n\
  vwh32u = i32.load vhs offset=4\n\
  vwhu = i64.extend_i32_u vwh32u\n\
  vau = call.import 4 (vrhu, vwhu, vfds)\n\
  vpid = call.import 2 ()\n\
  vz = i64.const 0\n\
  vchild = i64.eq vpid vz\n\
  br_if vchild 1() 2(vpid)\n\
  }}\n\
block 1 () {{\n\
  vp = i64.const 40000\n\
  vz = i64.const 0\n\
  vr = call.import 0 (vp, vz, vz)\n\
  vnine = i64.const 9\n\
  call.import 1 (vnine)\n\
  unreachable\n\
  }}\n\
block 2 (xpid: i64) {{\n\
  vupr = i64.const 5\n\
  vbuf = i64.const 43000\n\
  vfour = i64.const 4\n\
  vn = call 1 (vupr, vbuf, vfour)\n\
  vdownw = i64.const 4\n\
  vpong = i64.const 40100\n\
  vw = call 2 (vdownw, vpong, vfour)\n\
  vst = i64.const 41000\n\
  vz = i64.const 0\n\
  vwt = call.import 3 (xpid, vst, vz, vz)\n\
  vhi = i64.const 41001\n\
  vsw = i32.load8_u vhi\n\
  vs = i64.extend_i32_u vsw\n\
  call.import 1 (vs)\n\
  unreachable\n\
  }}\n\
}}\n\
{}\
export 0 func \"_start\" 0\n",
        shim(5, 6)
    )
}

/// `/bin/leaf` for [`ping_pong`]: writes `ping` up (fd 6), after `fills` writes of 16 KiB (four
/// fill the pipe), then reads the reply from down (fd 3), writes what it read to `out.txt`, and ends
/// with the count it read plus 10: it `exit`s with it, or returns it from its entry. The entry only
/// calls, so it emits, and the helpers it calls make the personality and stream calls, which bounce
/// to the interpreter — the shape of a real nim module's `_start`.
fn ping_leaf(fills: usize, exits: bool) -> String {
    let end = match exits {
        true => "  call.import 3 (vst)\n  unreachable\n",
        false => "  return vst\n",
    };
    let fill: String = (0..fills)
        .map(|i| format!("  vf{i} = call 4 (vup, vzeros, vchunk)\n"))
        .collect();
    format!(
        "memory 17\n\
import 0 \"__px_open\" (i64, i64, i64) -> (i64)\n\
import 1 \"__px_write\" (i64, i64, i64) -> (i64)\n\
import 2 \"__px_close\" (i64) -> (i64)\n\
import 3 \"__px_exit\" (i64) -> ()\n\
import 4 \"__px_read\" (i64, i64, i64) -> (i64)\n\
data 40000 \"out.txt\"\n\
data 40100 \"ping\"\n\
func (i64) -> (i64) {{\n\
block 0 (vcap: i64) {{\n\
  vr = call 1 ()\n\
  vst = call 2 (vr)\n\
  return vst\n\
  }}\n\
}}\n\
func () -> (i64) {{\n\
block 0 () {{\n\
  vup = i64.const 6\n\
  vzeros = i64.const 100000\n\
  vchunk = i64.const 16384\n\
{fill}\
  vping = i64.const 40100\n\
  vfour = i64.const 4\n\
  vw = call 4 (vup, vping, vfour)\n\
  vdown = i64.const 3\n\
  vbuf = i64.const 40200\n\
  vr = call 3 (vdown, vbuf, vfour)\n\
  return vr\n\
  }}\n\
}}\n\
func (i64) -> (i64) {{\n\
block 0 (vr: i64) {{\n\
  vpath = i64.const 40000\n\
  vplen = i64.const 7\n\
  vflags = i64.const 577\n\
  vfd = call.import 0 (vpath, vplen, vflags)\n\
  vbuf = i64.const 40200\n\
  vw = call.import 1 (vfd, vbuf, vr)\n\
  vc = call.import 2 (vfd)\n\
  vten = i64.const 10\n\
  vst = i64.add vr vten\n\
{end}\
  }}\n\
}}\n\
{}",
        shim(4, 1)
    )
}

/// #1896 — a leaf that holds pipe ends parks only in its stream calls, so it is offered to a host
/// that can suspend its emitted frames there. A call that parks hands its task the rest of the call,
/// the process parks as an interpreted one would, and once the pipe is ready the engine runs the rest
/// and the host resumes the frames with its results. Here the leaf parks on a read of an empty pipe,
/// and then (its pipe filled first) on a write to a full one; each run ends as it does interpreted,
/// after one tier-up and one resume. (The stand-in's frames are its entry's bounce, so the rest of
/// the call is the rest of the process: an `exit` there ends the process with no resume, which
/// [`a_leaf_that_exits_after_a_parked_call_ends_there`] covers.)
#[test]
fn a_leaf_that_parks_on_a_pipe_pauses_and_resumes() {
    let guest = ping_pong();
    for fill in [0, 4] {
        let leaf = ping_leaf(fill, false);
        let (interpreted, _) = run_tree(&guest, &leaf, false);
        assert_eq!(
            interpreted,
            Ending {
                root: Ok(14),
                wrote: Some(b"pong".to_vec()),
            },
            "interpreted (fill {fill}): the child read the parent's reply and exited 4 + 10"
        );
        let (emitted, leaves) = run_tree(&guest, &leaf, true);
        assert_eq!(emitted, interpreted, "emitted (fill {fill})");
        assert_eq!(
            leaves.offered,
            [(false, true)],
            "offered once, unpaged, as parking (fill {fill})"
        );
        assert_eq!(
            (leaves.tierups.len(), leaves.resumes),
            (1, 1),
            "the leaf tiered up at its entry, parked once, and was resumed (fill {fill}): {leaves:?}"
        );
    }
}

/// #1896 — the rest of a parked call can end the process: here it is the stand-in's whole entry,
/// which `exit`s. The process ends in the engine with no resume, and its host never resumes the
/// frames it holds; the tree ends as it does interpreted.
#[test]
fn a_leaf_that_exits_after_a_parked_call_ends_there() {
    let guest = ping_pong();
    let leaf = ping_leaf(0, true);
    let (interpreted, _) = run_tree(&guest, &leaf, false);
    assert_eq!(interpreted.root, Ok(14));
    let (emitted, leaves) = run_tree(&guest, &leaf, true);
    assert_eq!(emitted, interpreted);
    assert_eq!(
        (leaves.tierups.len(), leaves.resumes),
        (1, 0),
        "tiered up once, parked, and ended in the rest of the call: {leaves:?}"
    );
}
