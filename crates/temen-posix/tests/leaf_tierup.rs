//! #1896 — a process tree's **leaf processes** run whole on the emitted tier. The cooperative driver
//! keeps the tree: it forks, execs and waits on the interpreter, and an exec'd image that cannot park
//! tiers up at its entry. The host runs it and delivers how it ended. This test stands in for the
//! browser host without any wasm, as `coop_tierup.rs` does: it serves the tier-up by bouncing the
//! entry ([`CoopRun::bounce`], the nested interpretation an emitted image's own cap calls bounce
//! into), and checks that the tree ends exactly as it does interpreted.

use std::sync::{Arc, Mutex};

use temen_interp::bytecode::{CoopEvent, CoopRun, TierUpConfig};
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

/// What the engine asked of the host: the `paged` of each image it offered to emit, and the program
/// index of each process that tiered up at its entry.
#[derive(Debug, Default)]
struct Leaves {
    offered: Vec<bool>,
    tierups: Vec<u32>,
}

/// Run the tree with `/bin/leaf` as `command`, emitting leaf images iff `leaf`. Serves each entry
/// tier-up by bouncing the entry, the stand-in for running it emitted.
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
        leaf: Some(Arc::new(move |_, _: &temen_ir::Module, _, paged| {
            offers.lock().unwrap().push(paged);
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
    let mut tierups = Vec::new();
    let end = loop {
        match run.run() {
            CoopEvent::TierUp {
                module, func, argv, ..
            } => {
                tierups.push(module);
                let mut io = argv.to_vec();
                io.resize(io.len().max(1), 0);
                match run.bounce(func, &mut io, None) {
                    Ok(n) => run.deliver_tierup(&io[..n]),
                    Err(t) => run.deliver_tierup_trap(t),
                }
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
    (Ending { root, wrote }, Leaves { offered, tierups })
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
        [false],
        "offered once, unpaged: it cannot change its pages"
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
    assert_eq!(leaves.offered, [true], "offered once, paged");
    assert_eq!(leaves.tierups.len(), 1);
}

/// A leaf that holds a pipe end could park on it, and an image that imports `fork` could park in
/// it: both run interpreted, and end as they would anyway.
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
