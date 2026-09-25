//! #1768 — the JIT's **processes**: a personality `fork`, `execve` and blocking `waitpid` served on
//! the Cranelift JIT (FORK.md §9.4–§9.5).
//!
//! The interpreters serve these caller requests from their schedulers: a forking vCPU is cloned, a
//! waiting one is benched until its child exits. On the JIT a process is a live native stack, so the
//! same three requests are served by a **process tree** — the root process and every fork twin it
//! (transitively) spawned, each on its own OS thread over its own window and powerbox:
//!
//! * **exec** unwinds the run and starts the new image in its place (§9.4) — no continuation to keep.
//! * **fork** is durable freeze → copy the window → thaw both (FORK.md §1). The program is compiled
//!   with its fork calls as durable suspend points ([`ForkPlan`] — the one `temen-durable`
//!   transform, restricted to the calls that can fork), so a fork unwinds the caller's native frames
//!   into its window's shadow stack; the run hands the frozen window to [`Tree::fork`], which
//!   duplicates the process — its pid, its powerbox ([`Host::fork_powerbox_jit`]), a private copy of
//!   its window with `0` injected as the call's result — and starts the twin on its own thread
//!   ([`CompiledModule::run_twin`]) running its parent's code; the parent rewinds in place with the
//!   twin's pid. Both resume past the same call with their own answer: reply injection, never
//!   re-issue (FORK.md §3).
//! * **a blocking `waitpid`** re-runs its op whenever the tree's bell rings — a child exited, a
//!   signal arrived — exactly the interpreters' rewound park (invariant 7), with the OS thread blocked
//!   in between; in a run that delivers signals, a deliverable one completes it `-EINTR`.
//!
//! A program is compiled once per tree (#1825). Its code is an [`Image`] every process running it
//! instantiates ([`temen_jit::SharedCode`]), each over its own powerbox, function table and run state:
//! the process that compiled it, its fork twins, and every later `execve` of the same command — found
//! in the tree's cache by what the code depends on besides the module ([`CodeKey`]).
//!
//! Pids follow the interpreters' (#799: a twin's pid *is* its task id): the root is `1`, twins count
//! from `2` in fork order, a refused powerbox burns its number (#1648), and a fork past the run's
//! vCPU quota answers `-EAGAIN`. When the root's image chain finishes, the run ends with it (DESIGN.md
//! §12 domain lifetime): every running twin is stopped at its next safepoint through the tree's
//! kill-path cell and joined, as the oracle's teardown stops running siblings.
//!
//! Where the JIT cannot fork what the oracle would, it answers `-ENOSYS` — fork unavailable here —
//! never a wrong answer (FORK.md §9.5 enumerates the cases).

use std::collections::{BTreeSet, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use temen_interp::{GuestMem, Host, ParkEvent, Trap, TwinTrap, Value};
use temen_ir::durable_abi::{ShadowArena, STATE_OFF, STATE_UNWINDING};
use temen_ir::errno::{EAGAIN, EINTR, EINVAL, ENOSYS};
use temen_ir::{cap_id, FuncIdx, Inst, Module, ValType};
use temen_jit::{CompiledModule, ForkPoint, JitOutcome, SharedCode, TrapKind, TwinWindow, VmCtx};

use crate::{
    blocked_wait_interrupted, fast_cap_resolver, module_resolver, module_serves,
    production_budget_taker, production_grant_hooks, CapCtx, JitRun, JitSigDelivery,
    CLI_JIT_TABLE_LOG2,
};

/// The root process's pid — the personality's process-table root, as on both interpreters.
const ROOT_PID: u64 = 1;

/// A process thread's stack. A process runs a whole program — a compiler phase, say — on it, so it
/// gets the room a main thread would, not a worker's 2 MiB (only what it touches is committed).
const PROC_STACK: usize = 256 << 20;

/// The personality op that raises [`ParkEvent::ForkSelf`] — `temen-posix`'s `fork`.
fn is_fork_op(type_id: u32, op: u32) -> bool {
    type_id == cap_id::HOST_PROC && op == temen_posix::OP_FORK
}

/// Whether a call handing the host's dispatch entry `(type_id, op)` ([`Inst::host_dispatch`]) can
/// reach an op `pred` names over `host`'s import bindings. A `call.cap` names its op; an import call
/// reaches its slot's binding — or anything, when `import.attach` may retarget the slot; a
/// dynamic-mode call's interface is interned at run time, so it may reach anything.
///
/// Read where a program's reading over a powerbox is made ([`ForkPlan::of`], [`drives_jit`]), over
/// bindings that cannot change afterwards except through a rebindable slot (counted as reaching
/// anything).
fn may_reach(type_id: u32, op: u32, host: &Host, pred: impl Fn(u32, u32) -> bool) -> bool {
    match type_id {
        temen_ir::CAP_IMPORT_TYPE_ID => {
            host.import_rebindable(op & 0xFFFF)
                || host.import_target(op).is_ok_and(|(t, o, _)| pred(t, o))
        }
        temen_ir::CAP_DYN_TYPE_ID => true,
        _ => pred(type_id, op),
    }
}

/// The distinct dispatch pairs ([`Inst::host_dispatch`]) of `m`'s calls out of the window — all a
/// powerbox's reading of the program looks at.
pub(crate) fn host_calls(m: &Module) -> BTreeSet<(u32, u32)> {
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter().flat_map(|b| &b.insts))
        .filter_map(Inst::host_dispatch)
        .collect()
}

/// Whether a program making `calls` can drive the §22 `Jit` interface over `host`: define code of its
/// own at run time, into its own code arena — which then holds more than the code it shares.
pub(crate) fn drives_jit(calls: &BTreeSet<(u32, u32)>, host: &Host) -> bool {
    calls
        .iter()
        .any(|&(t, o)| may_reach(t, o, host, |t, _| t == cap_id::JIT))
}

/// How a program forks on the JIT (FORK.md §9.5): its **fork sites** — the dispatch pairs of the calls
/// that can dispatch the fork op — and the shadow arena its root unwinds into.
pub(crate) struct ForkPlan {
    /// Sorted.
    sites: Vec<(u32, u32)>,
    arena: ShadowArena,
}

impl ForkPlan {
    /// The plan of `m`, which makes `calls`, over `host` — `None` when `m` cannot fork here, in which
    /// case a `fork` it makes answers `-ENOSYS` ("unavailable on this tier").
    ///
    /// `m` can fork on the JIT when it can reach a fork site at all, declares a shadow arena to unwind
    /// into (INVARIANTS.md #16: the placement is the module's; there is no default), and is **bare** —
    /// no fibers, threads, `setjmp`, §14 children or §22 units, the state a fork would have to
    /// duplicate that lives outside the window, or code the transform never saw on the stack. (The
    /// oracle forks such a module when it is momentarily bare; FORK.md §9.5.)
    ///
    /// The plan is what the program is instrumented by ([`Self::instrument`]) and what the cap thunk
    /// reads a fork call against ([`serve_request`]): a call is a site by its dispatch pair — the pair
    /// the thunk is handed — so the compile and the thunk cannot disagree about which calls are sites.
    fn of(m: &Module, calls: &BTreeSet<(u32, u32)>, host: &Host) -> Option<ForkPlan> {
        let arena = m.memory?.shadow?;
        if !arena.region_fits(0) {
            return None;
        }
        // §14 children and §22 units run code the transform never saw: a child on its own vCPU, a unit
        // through the `call.dyn` slot `Jit.install` gives it.
        let foreign = |t: u32, _| t == cap_id::INSTANTIATOR || t == cap_id::JIT;
        let bare = !m.funcs.iter().any(|f| f.uses_fibers_or_threads())
            && !m
                .funcs
                .iter()
                .flat_map(|f| f.blocks.iter().flat_map(|b| &b.insts))
                .any(|i| matches!(i, Inst::SetJmp { .. } | Inst::LongJmp { .. }))
            && !calls.iter().any(|&(t, o)| may_reach(t, o, host, foreign));
        let sites: Vec<(u32, u32)> = calls
            .iter()
            .copied()
            .filter(|&(t, o)| may_reach(t, o, host, is_fork_op))
            .collect();
        (bare && !sites.is_empty()).then_some(ForkPlan { sites, arena })
    }

    /// Whether a call handing the host `pair` ([`Inst::host_dispatch`]) is one of the plan's sites.
    fn is_site(&self, pair: (u32, u32)) -> bool {
        self.sites.binary_search(&pair).is_ok()
    }

    /// `m` instrumented to fork, and where its `entry` enters it now — `None` when the transform
    /// refuses a function that can reach a site (FORK.md §9.5, refusal 2), and `m` cannot fork. The
    /// transform instruments the functions that can reach a fork site — through direct calls, and
    /// through `call.dyn`s that can select a function whose address the program takes
    /// ([`temen_durable::TransformOpts::fork`]) — and leaves everything else byte-identical.
    fn instrument(&self, m: &Module, entry: FuncIdx) -> Option<(Module, FuncIdx)> {
        let site = |i: &Inst| i.host_dispatch().is_some_and(|p| self.is_site(p));
        let t = temen_durable::transform(m, &temen_durable::TransformOpts::fork(&site)).ok()?;
        Some((t.module, t.body[entry as usize]))
    }
}

/// A JIT run's **process tree**: the state its processes share. Lives as long as any of them.
pub(crate) struct Tree {
    state: Mutex<TreeState>,
    /// The root's run has ended: a running twin is being stopped, a blocked one must return.
    torn_down: AtomicBool,
    /// The bell a blocked process waits on (see [`Self::ring`]). Its mutex guards only the wait, and
    /// no path takes another lock while holding it.
    bell: (Mutex<()>, Condvar),
    /// Bumped by every ring — what a waiter snapshots before its op and waits to see change.
    rings: AtomicU64,
    /// The tree's kill-path cell: the run's watchdog cell when a deadline is armed, else the tree's
    /// own. Every image a twin runs polls it (so teardown can stop a running twin), and so does a
    /// root image that can fork.
    interrupt: Arc<AtomicU64>,
    /// Whether the run armed a deadline (then every image polls the cell, the root's included).
    deadline: bool,
    quota: temen_jit::Quota,
    /// The commands the tree compiled (#1825), by what their code depends on.
    code: Mutex<HashMap<CodeKey, Arc<CodeSlot>>>,
}

struct TreeState {
    /// The next twin pid (from 2; a refused powerbox burns one).
    next_pid: u64,
    /// Live processes, the root included — held to the run's vCPU quota, the oracle's task cap.
    live: usize,
    /// Every twin's thread, joined at teardown.
    threads: Vec<std::thread::JoinHandle<()>>,
    /// Twins that trapped (not a clean `exit`), for [`temen_interp::last_twin_traps`].
    traps: Vec<TwinTrap>,
}

impl Tree {
    fn new(interrupt: Option<&Arc<AtomicU64>>, quota: temen_jit::Quota) -> Arc<Tree> {
        Arc::new(Tree {
            state: Mutex::new(TreeState {
                next_pid: 2,
                live: 1,
                threads: Vec::new(),
                traps: Vec::new(),
            }),
            torn_down: AtomicBool::new(false),
            bell: (Mutex::new(()), Condvar::new()),
            rings: AtomicU64::new(0),
            interrupt: interrupt
                .cloned()
                .unwrap_or_else(|| Arc::new(AtomicU64::new(0))),
            deadline: interrupt.is_some(),
            quota,
            code: Mutex::new(HashMap::new()),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TreeState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The kill-path cell an image of process `pid` polls, if any (see [`Self::interrupt`]).
    fn interrupt_for(&self, pid: u64, forks: bool) -> Option<*const AtomicU64> {
        (self.deadline || forks || pid != ROOT_PID).then_some(Arc::as_ptr(&self.interrupt))
    }

    /// Something happened a blocked process may be waiting for: wake every waiter to re-run its op.
    /// Wake-all, never a pick (invariant 4) — each waiter's own op decides whether it proceeds.
    fn ring(&self) {
        self.rings.fetch_add(1, Ordering::SeqCst);
        let _g = self.bell.0.lock().unwrap_or_else(|e| e.into_inner());
        self.bell.1.notify_all();
    }

    /// The ring a process's personality doors pull ([`Host::set_wake_bell`]). Weak: a door that
    /// outlives the run (the embedder's personality does) rings nothing.
    fn bell(self: &Arc<Self>) -> Arc<dyn Fn() + Send + Sync> {
        let tree = Arc::downgrade(self);
        Arc::new(move || {
            if let Some(t) = tree.upgrade() {
                t.ring();
            }
        })
    }

    /// Block until the bell rings past `seen`, the tree is torn down, or the kill-path fires. Wakes
    /// at a bounded cadence too, so a kill-path store (which rings nothing) is observed.
    fn wait(&self, seen: u64) {
        let mut g = self.bell.0.lock().unwrap_or_else(|e| e.into_inner());
        while self.rings.load(Ordering::SeqCst) == seen
            && !self.torn_down.load(Ordering::SeqCst)
            && self.interrupt.load(Ordering::SeqCst) == 0
        {
            g = self
                .bell
                .1
                .wait_timeout(g, std::time::Duration::from_millis(20))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// The code process `pid` starts `program` at `entry` with, over `host` (#1825). An `execve` of a
    /// command the tree already compiled under an equal [`CodeKey`] instantiates that compile. Anything
    /// else is compiled here ([`compile`]): the embedder's program, which no other process starts; a
    /// command that can extend its code, which is then its own; a command's first `execve`, whose
    /// compile the tree keeps. A `process` image (not a serve handler) may fork.
    ///
    /// # Safety
    /// As [`compile_image`].
    unsafe fn load(
        &self,
        pid: u64,
        host: &mut Host,
        program: &Program<'_>,
        entry: FuncIdx,
        process: bool,
    ) -> Result<Loaded, temen_jit::JitError> {
        let m = program.module();
        let calls = host_calls(m);
        let jit = drives_jit(&calls, host);
        let plan = process.then(|| ForkPlan::of(m, &calls, host)).flatten();
        let interrupt = self.interrupt_for(pid, plan.is_some());
        let (cm, image) = match program {
            Program::Command {
                digest, size_log2, ..
            } if !jit => {
                let key = CodeKey {
                    digest: *digest,
                    size_log2: *size_log2,
                    entry,
                    polls: interrupt.is_some(),
                    sites: plan.as_ref().map(|p| p.sites.clone()),
                };
                let slot = Arc::clone(
                    self.code
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .entry(key)
                        .or_insert_with(|| {
                            Arc::new(CodeSlot {
                                image: Mutex::new(None),
                            })
                        }),
                );
                let mut kept = slot.image.lock().unwrap_or_else(|e| e.into_inner());
                match kept.clone() {
                    Some(Some(image)) => {
                        let cm = image.code.instance(CapCtx::Raw(&mut *host).ptr());
                        (cm, Some(image))
                    }
                    Some(None) => {
                        drop(kept);
                        compile(host, program, entry, jit, plan, interrupt, self.quota)?
                    }
                    None => {
                        let (cm, image) =
                            compile(host, program, entry, jit, plan, interrupt, self.quota)?;
                        *kept = Some(image.clone());
                        (cm, image)
                    }
                }
            }
            _ => compile(host, program, entry, jit, plan, interrupt, self.quota)?,
        };
        Ok(Loaded {
            cm,
            image,
            interrupt,
        })
    }

    /// **Fork** the process frozen at `point` over `host` (FORK.md §9.5, the JIT's arm of the
    /// oracle's `fork_vcpu`): mint the twin's pid, copy the window, duplicate the powerbox, and start
    /// the twin running `image` — returning the parent's reply: the twin's pid, or `-EAGAIN` for a
    /// refusal that duplicated nothing (the quota is full; the window aliases shared memory; a
    /// capability the core cannot duplicate). Holds the tree lock throughout, as the oracle holds its
    /// scheduler lock, so concurrent forks mint pids in one order.
    fn fork(self: &Arc<Self>, host: &mut Host, point: &ForkPoint<'_>, image: &Arc<Image>) -> i64 {
        let mut st = self.lock();
        if st.live >= self.quota.max_vcpus || self.torn_down.load(Ordering::SeqCst) {
            return EAGAIN;
        }
        let pid = st.next_pid;
        let pages = host
            .cap_window_pages(point.window_base())
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        // The window first: refusing it has registered nothing, so its number is not burned.
        let Some(window) = point.twin(&pages, 0) else {
            return EAGAIN;
        };
        // #1648 — past here a fork factory may have registered a process under this pid: never
        // reuse it, whatever happens next.
        st.next_pid = pid + 1;
        let Some(twin_host) = host.fork_powerbox_jit(pid, window.base()) else {
            return EAGAIN;
        };
        let hooks = twin_host.exit_hooks();
        let tree = Arc::clone(self);
        let image = Arc::clone(image);
        let args = point.args().to_vec();
        let spawned = std::thread::Builder::new()
            .name(format!("temen-jit-pid{pid}"))
            .stack_size(PROC_STACK)
            .spawn(move || tree.run_twin(pid, twin_host, image, window, args));
        match spawned {
            Ok(t) => {
                st.live += 1;
                st.threads.push(t);
            }
            // The OS would not give the twin a thread, after its personalities registered it: it
            // dies at birth, as a crash — retired through its exit hooks and reaped by its parent
            // like any twin that trapped, never a table entry no one can reap.
            Err(_) => {
                let crash = Err(Trap::ThreadFault);
                for hook in hooks {
                    hook(temen_interp::reap_status(&crash));
                }
                st.traps.push(TwinTrap {
                    task: pid,
                    trap: Trap::ThreadFault,
                    backtrace: Vec::new(),
                    fault: None,
                });
                drop(st);
                self.ring();
            }
        }
        pid as i64
    }

    /// A twin's thread: run its process to the end, then retire it — the exit hooks its
    /// personalities rode in on first (so a parent's re-run `waitpid` finds it a zombie), then the
    /// ring that wakes that parent.
    fn run_twin(
        self: Arc<Self>,
        pid: u64,
        mut host: Host,
        image: Arc<Image>,
        window: TwinWindow,
        args: Vec<i64>,
    ) {
        let start = Start::Twin {
            image,
            window,
            args,
        };
        // SAFETY: this thread owns `host` for the whole process.
        let (r, results, final_host) = unsafe { run_process(&self, pid, &mut host, start, None) };
        let result = interp_result(&r, &results);
        // Retire it from the tree before its exit is observable: once its exit hooks run, its parent
        // can reap it and go on — fork again, or end the run — so its vCPU slot must be free and its
        // crash on record by then, as the oracle's are when its task ends.
        {
            let mut st = self.lock();
            st.live -= 1;
            // A twin the run's end stopped did not crash: the oracle reaps it with the run, recording
            // nothing, and so does the tree.
            let stopped = self.torn_down.load(Ordering::SeqCst);
            if let Err(trap) = result {
                if !matches!(trap, Trap::Exit(_)) && !stopped {
                    st.traps.push(TwinTrap {
                        task: pid,
                        trap,
                        backtrace: Vec::new(),
                        fault: None,
                    });
                }
            }
        }
        for hook in final_host.as_ref().unwrap_or(&host).exit_hooks() {
            hook(temen_interp::reap_status(&result));
        }
        self.ring();
    }

    /// The root's run has ended, and the run ends with it (DESIGN.md §12): stop every running twin
    /// at its next safepoint (the kill-path cell every twin image polls), wake every blocked one,
    /// wait for them all, and publish the ones that trapped.
    fn teardown(&self) {
        self.torn_down.store(true, Ordering::SeqCst);
        let threads = std::mem::take(&mut self.lock().threads);
        if !threads.is_empty() {
            self.interrupt.store(1, Ordering::SeqCst);
            self.ring();
        }
        for t in threads {
            let _ = t.join();
        }
        temen_interp::publish_twin_traps(std::mem::take(&mut self.lock().traps));
    }
}

/// A program compiled for the tree (#1825): the code every process that runs it instantiates — the
/// process that compiled it, its fork twins, and every later `execve` of the same command — each over
/// its own powerbox, function table and run state.
pub(crate) struct Image {
    code: SharedCode,
    /// The entry's result types.
    results: Vec<ValType>,
    /// How the program forks — `None` when it cannot fork here (the code is not instrumented).
    fork: Option<ForkPlan>,
}

/// What a command's code depends on besides the module itself (#1825): an `execve` whose key is equal
/// instantiates the tree's compile of it.
#[derive(PartialEq, Eq, Hash)]
struct CodeKey {
    /// The command module's content digest ([`temen_interp::ExecImage::digest`]): the same code
    /// however it reached the exec — a registered command, or a program the tree built and promoted
    /// on each run of it (#763). Identity is structural (INVARIANTS #10).
    digest: [u8; 32],
    /// The window it runs in: the confinement mask is baked.
    size_log2: u8,
    entry: FuncIdx,
    /// Whether the code polls the tree's kill-path cell.
    polls: bool,
    /// The fork sites its powerbox reads it as having — what its instrumentation depends on.
    sites: Option<Vec<(u32, u32)>>,
}

/// One command's compile, made by the first process to exec it; a later one waits on the lock while
/// it compiles. `None` until then, then whether the code could be shared: when it could not, each
/// process compiles its own.
struct CodeSlot {
    image: Mutex<Option<Option<Arc<Image>>>>,
}

/// What a fresh image runs.
pub(crate) enum Program<'a> {
    /// The embedder's program: the root's first image. No other process starts it.
    Embedder(&'a Module),
    /// An `execve`'s command, from its grant, in a window of `size_log2`: the caller's backed prefix.
    Command {
        grant: Arc<Module>,
        digest: [u8; 32],
        size_log2: u8,
    },
}

impl Program<'_> {
    fn module(&self) -> &Module {
        match self {
            Program::Embedder(m) => m,
            Program::Command { grant, .. } => grant,
        }
    }
}

/// How a process image starts: fresh from its program (the root's first image, or an exec's), or as a
/// fork twin re-entering its parent's image over the window the fork duplicated.
enum Start<'a> {
    Fresh {
        program: Program<'a>,
        entry: FuncIdx,
        args: Vec<i64>,
        init_mem: Option<Vec<u8>>,
    },
    Twin {
        image: Arc<Image>,
        window: TwinWindow,
        args: Vec<i64>,
    },
}

/// What a process image's `call.cap` thunk finds through its vmctx's `embedder` slot: the tree it
/// belongs to, and the image it runs when its code is shared — whose fork plan the thunk reads a fork
/// call against.
pub(crate) struct ProcCtx {
    tree: Arc<Tree>,
    image: Option<Arc<Image>>,
}

impl ProcCtx {
    /// How the process forks — `None` when it cannot.
    fn fork(&self) -> Option<&ForkPlan> {
        self.image.as_ref()?.fork.as_ref()
    }
}

/// The process a thunk call's `trap_out` (the running instance's vmctx) belongs to, if any.
///
/// # Safety
/// `trap_out` is the `trap_out` of a thunk call made by compiled code — which every call over a host
/// armed for caller requests is (only a process run arms one, [`run_process`]).
unsafe fn proc_of<'a>(trap_out: *mut i64) -> Option<&'a ProcCtx> {
    let vm = VmCtx::of_trap_out(trap_out);
    (!vm.embedder.is_null()).then(|| &*(vm.embedder as *const ProcCtx))
}

/// Run one **process** of `tree` — pid `pid` over `host` — from `start`, down its chain of execs.
/// Returns the last image's run, its entry's result types, and — when an exec replaced the first
/// image's powerbox — the powerbox the process ended in (whose exit hooks retire it).
///
/// # Safety
/// `host` is this process's live powerbox, used by no one else while it runs.
unsafe fn run_process(
    tree: &Arc<Tree>,
    pid: u64,
    host: &mut Host,
    first: Start<'_>,
    snapshot_cap: Option<usize>,
) -> (
    Result<JitRun, temen_jit::JitError>,
    Vec<ValType>,
    Option<Host>,
) {
    // The running image's powerbox once an exec has replaced the first one.
    let mut image_host: Option<Host> = None;
    let mut start = first;
    loop {
        let cur: &mut Host = match image_host.as_mut() {
            Some(h) => h,
            None => &mut *host,
        };
        let (r, results) = match start {
            Start::Fresh {
                program,
                entry,
                args,
                init_mem,
            } => {
                let module = program.module();
                let results = module.funcs[entry as usize].results.clone();
                // A serve handler is not a process image: a serving module serves no requests, and
                // does not fork.
                let process = !module_serves(module);
                if process {
                    cur.arm_caller_requests();
                }
                let r = tree
                    .load(pid, cur, &program, entry, process)
                    .and_then(|code| {
                        let ctx = ProcCtx {
                            tree: Arc::clone(tree),
                            image: code.image,
                        };
                        let start = Entry::Fresh {
                            args: &args,
                            init_mem: init_mem.as_deref(),
                            snapshot_cap,
                        };
                        run_image(cur, code.cm, code.interrupt, Some(&ctx), start)
                    });
                (r, results)
            }
            Start::Twin {
                image,
                window,
                args,
            } => {
                cur.arm_caller_requests();
                let cm = image.code.instance(CapCtx::Raw(&mut *cur).ptr());
                let results = image.results.clone();
                let ctx = ProcCtx {
                    tree: Arc::clone(tree),
                    image: Some(image),
                };
                let start = Entry::Twin {
                    window,
                    args: &args,
                };
                let interrupt = tree.interrupt_for(pid, true);
                (run_image(cur, cm, interrupt, Some(&ctx), start), results)
            }
        };
        let unwound = matches!(
            r,
            Ok(JitRun {
                outcome: JitOutcome::HostUnwound,
                ..
            })
        );
        if !unwound {
            cur.disarm_caller_requests();
            return (r, results, image_host);
        }
        // An exec: only an armed process stores the unwind code, and only with an image parked.
        let Some(img) = cur.take_exec_image() else {
            cur.disarm_caller_requests();
            return (Err(temen_jit::JitError::Malformed), results, image_host);
        };
        // The commit: the personality hands over the argv it staged, and the new image finds it in
        // its args region.
        let init = img.host.exec_commit_args().map(|blob| {
            let mut buf = vec![0u8; temen_ir::module_args_base() as usize];
            buf.extend_from_slice(&blob);
            buf
        });
        cur.disarm_caller_requests();
        start = Start::Fresh {
            program: Program::Command {
                grant: img.module,
                digest: img.digest,
                size_log2: img.child_size.trailing_zeros() as u8,
            },
            entry: img.entry as FuncIdx,
            args: img.entry_args,
            init_mem: init,
        };
        image_host = Some(img.host);
    }
}

/// A process image's code ([`Tree::load`]): its own instance, the image it instantiates when the code
/// is shared, and the kill-path cell the code polls.
struct Loaded {
    cm: CompiledModule,
    image: Option<Arc<Image>>,
    interrupt: Option<*const AtomicU64>,
}

/// Compile `program` at `entry` over `host` — instrumented to fork when `plan` says it can — and share
/// its code as an [`Image`], unless the program can extend its code (`jit`) or the code is more than
/// code ([`CompiledModule::share`]).
///
/// # Safety
/// As [`compile_image`].
unsafe fn compile(
    host: &mut Host,
    program: &Program<'_>,
    entry: FuncIdx,
    jit: bool,
    plan: Option<ForkPlan>,
    interrupt: Option<*const AtomicU64>,
    quota: temen_jit::Quota,
) -> Result<(CompiledModule, Option<Arc<Image>>), temen_jit::JitError> {
    let sized;
    let m = match program {
        Program::Embedder(m) => *m,
        // The command runs in a window the size of the caller's backed prefix, as on both
        // interpreters, whose image-replace reuses the caller's window in place (a larger window than
        // the command declares is a safe superset, masked to its actual size).
        Program::Command {
            grant, size_log2, ..
        } => {
            let mut m = (**grant).clone();
            m.memory = m.memory.map(|mc| temen_ir::Memory {
                size_log2: *size_log2,
                ..mc
            });
            sized = m;
            &sized
        }
    };
    let instrumented = plan.as_ref().and_then(|p| p.instrument(m, entry));
    let (code, at) = instrumented
        .as_ref()
        .map_or((m, entry), |(im, at)| (im, *at));
    let cm = compile_image(host, code, at, interrupt, quota, jit)?;
    let image = (!jit).then(|| cm.share()).flatten().map(|code| {
        Arc::new(Image {
            code,
            results: m.funcs[entry as usize].results.clone(),
            fork: plan.filter(|_| instrumented.is_some()),
        })
    });
    Ok((cm, image))
}

/// Where an image's run starts: its entry, fresh (seeded with `init_mem`, snapshotting
/// `snapshot_cap` bytes), or re-entered as a fork twin over the window the fork duplicated.
pub(crate) enum Entry<'a> {
    Fresh {
        args: &'a [i64],
        init_mem: Option<&'a [u8]>,
        snapshot_cap: Option<usize>,
    },
    Twin {
        window: TwinWindow,
        args: &'a [i64],
    },
}

/// The one JIT compile of an image over `host`: the unlocked [`crate::cap_thunk`] + raw `*mut Host` +
/// the D45 fast path, §14 module children resolving their `Module` grant, and the §5 kill-path
/// `interrupt` polled when `Some`. An image that can drive the §22 `Jit` interface (`jit`, see
/// [`drives_jit`]) over a fiber-hosting grant gets the fiber runtime a submitted unit's `cont.*`
/// resolve against; no other image does — none could use it, and it is run state the code would then
/// own (#1825).
///
/// # Safety
/// `host` is the live powerbox the code dispatches into (every instance of shared code passes its own,
/// in the same unlocked shape); `interrupt` (when `Some`) outlives the code.
pub(crate) unsafe fn compile_image(
    host: &mut Host,
    module: &Module,
    entry: FuncIdx,
    interrupt: Option<*const AtomicU64>,
    quota: temen_jit::Quota,
    jit: bool,
) -> Result<CompiledModule, temen_jit::JitError> {
    let cc = CapCtx::Raw(&mut *host);
    let mut cm = CompiledModule::compile(
        module,
        entry,
        cc.thunk(),
        cc.ptr(),
        temen_ir::DEFAULT_RESERVED_LOG2,
        None,
        Some(module_resolver), // §14 module children resolve their `Module` grant
        interrupt,
        None, // no fuel budget armed (the CLI bounds runaways via the interrupt kill-path)
        Some(fast_cap_resolver),
        quota,
        CLI_JIT_TABLE_LOG2,
    )?;
    // Fiber-hosting grant (`set_jit_hosts_fibers`, e.g. the powerbox): stand up the fiber runtime so a
    // submitted unit's `cont.*` resolve even when this top-level module uses no fibers itself.
    if jit && host.jit_hosts_fibers() {
        cm.enable_fiber_hosting(quota)?;
    }
    Ok(cm)
}

/// The single-threaded JIT run of an image's code `cm` over `host` (the unlocked [`crate::cap_thunk`]
/// and a raw `*mut Host`): register the live module for the cap thunk's re-entries, arm the
/// production §14 hooks and the §5 kill-path `interrupt` (the cell the code polls), and run it. As a `process` of a
/// tree, the image also carries its [`ProcCtx`], rings the tree's bell from its personality's doors,
/// and — when it can fork — the fork hook that duplicates it.
///
/// # Safety
/// `host` is the live powerbox `cm` dispatches into, touched by no one else during the run;
/// `interrupt` (when `Some`) is the cell `cm` polls, and outlives the call.
pub(crate) unsafe fn run_image(
    host: &mut Host,
    mut cm: CompiledModule,
    interrupt: Option<*const AtomicU64>,
    process: Option<&ProcCtx>,
    start: Entry<'_>,
) -> Result<JitRun, temen_jit::JitError> {
    let raw_host: *mut Host = host;
    let cc = CapCtx::Raw(raw_host);
    let host = &mut *raw_host;
    if let Some(p) = process {
        cm.set_embedder_ctx(p as *const ProcCtx as *mut c_void);
        host.set_wake_bell(p.tree.bell());
        if let Some(image) = p.image.as_ref().filter(|i| i.fork.is_some()) {
            let tree = Arc::clone(&p.tree);
            let image = Arc::clone(image);
            let host = SendPtr(raw_host);
            cm.set_fork_hook(Some(Box::new(move |point: &ForkPoint<'_>| {
                // SAFETY: the hook runs on the forking process's own thread, between two entries
                // of its run, when nothing else touches its powerbox.
                tree.fork(unsafe { &mut *host.get() }, point, &image)
            })));
        }
    }
    host.set_jit_native_ctx(&mut cm as *mut CompiledModule as usize);
    // CALLS.md 5c.1c — production granted-child hooks + the kill cell for thunk-blocked waits.
    cm.set_grant_child_hooks(Some(production_grant_hooks(cc)));
    cm.set_budget_taker(Some(production_budget_taker(cc)));
    if let Some(ip) = interrupt {
        host.set_epoch_cell(ip as usize);
    }
    // §3.6 / I36 slice 3: register the module for the cap thunk's native serve arm too — a serving
    // module need not hold a `Jit` grant (whose per-domain ctx the line above sets).
    host.set_serve_native_ctx(&mut cm as *mut CompiledModule as usize);
    let r = match start {
        Entry::Fresh {
            args,
            init_mem,
            snapshot_cap,
        } => CompiledModule::run_raw(&mut cm, args, init_mem, snapshot_cap),
        Entry::Twin { window, args } => CompiledModule::run_twin(&mut cm, window, args),
    };
    // The run's native contexts and kill-path cell die with it: a later run over this host must not
    // find them.
    host.set_jit_native_ctx(0);
    host.set_serve_native_ctx(0);
    host.set_epoch_cell(0);
    r.map(|(outcome, snapshot)| JitRun {
        outcome,
        backtrace: cm.last_trap_backtrace().to_vec(),
        trap_fiber: cm.last_trap_fiber(),
        snapshot,
    })
}

/// A raw pointer the fork hook carries to the thread it runs on (its own).
struct SendPtr(*mut Host);
// SAFETY: the pointee is the forking process's powerbox; the hook is called only on that process's
// own thread (see `run_image`).
unsafe impl Send for SendPtr {}
impl SendPtr {
    fn get(&self) -> *mut Host {
        self.0
    }
}

/// #1768 — the JIT's arm of the caller-request decision (the interpreters' `decide`), made by the
/// `call.cap` thunk of a JIT process right after a personality op raised a request. `dispatch` is the
/// `(type_id, op)` the call handed the thunk ([`Inst::host_dispatch`]); `bell` is the tree's ring
/// count read *before* the op ran ([`bell_of`]), so a child that exits between the op's look and the
/// wait below is not slept through. Returns `true` when the op must run again (a waiter was woken).
///
/// * `execve` — admit and build the image through the one rule every engine shares
///   ([`Host::exec_image`]); admitted, park it and unwind the whole run
///   ([`temen_jit::HOST_UNWIND_CODE`]) — an image-replace never returns to its caller, so the
///   caller's native stack is simply discarded (FORK.md §9.4). Refused: `-EINVAL`, nothing changed.
/// * `fork` — on the process's root computation, at one of its image's fork sites
///   ([`ForkPlan::is_site`]), start the unwind ([`begin_fork_unwind`]): the call's trailing poll
///   unwinds the caller into its shadow stack, and the run hands the frozen window to the fork hook.
///   In a fiber, `-EAGAIN`, as the oracle refuses a non-bare fork; anywhere else the JIT cannot unwind
///   to (an image with no fork plan, a host frame or a barrier below the call), `-ENOSYS` — fork
///   unavailable here.
/// * a blocking `waitpid` — `-EINTR` if a deliverable signal is pending (in a run that delivers
///   signals); else wait for the bell and run the op again, which reaps a child that exited or waits
///   on. A kill-path store (a deadline, the tree's teardown) ends the wait with the kill trap.
///
/// `window` is the call's view of the guest window — the one its op read and wrote through.
///
/// # Safety
/// The [`crate::cap_thunk`] contract for `results`/`trap_out`, over a host armed for caller
/// requests.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn serve_request(
    host: &mut Host,
    dispatch: (u32, u32),
    window: Option<&mut dyn GuestMem>,
    window_mapped: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
    bell: u64,
) -> bool {
    let answer = |v: i64| {
        if n_results != 0 {
            *results = v;
        }
    };
    let Some(proc) = proc_of(trap_out) else {
        return false;
    };
    match host.take_park_request() {
        Some(ParkEvent::ExecSelf { cmd }) => {
            // The JIT's own gates — the interpreters' `durable` + clean-root checks: a durable
            // domain's subtree must stay snapshottable, and a fiber is not the process's image to
            // replace.
            let admitted = if host.is_durable() || temen_jit::fiber_active() {
                Err(EINVAL)
            } else {
                host.exec_module(cmd)
                    .and_then(|m| host.exec_image(&m, &[], 0, 0, window_mapped))
            };
            match admitted {
                Ok(img) => {
                    host.stash_exec_image(img);
                    *trap_out = temen_jit::HOST_UNWIND_CODE as i64;
                }
                Err(e) => answer(e),
            }
            false
        }
        Some(ParkEvent::ForkSelf) => {
            if temen_jit::fiber_active() {
                answer(EAGAIN);
            } else if !proc.fork().is_some_and(|plan| {
                plan.is_site(dispatch)
                    && !temen_jit::reentered()
                    && !in_signal_handler(trap_out)
                    && window.is_some_and(|w| begin_fork_unwind(w, plan.arena))
            }) {
                answer(ENOSYS);
            }
            false
        }
        Some(ParkEvent::TaskExit(_) | ParkEvent::TaskExitAny) => {
            // A deliverable signal completes the wait `-EINTR`, as on the interpreters — in a run
            // that delivers signals, where the handler then takes it. A run that delivers none
            // (every process run today: the #932 delivery is armed only by the one-shot entry)
            // would find the same signal still pending on every retry, so it waits on.
            if delivers_signals(trap_out) && host.wait_interrupted() {
                answer(EINTR);
                return false;
            }
            proc.tree.wait(bell);
            if blocked_wait_interrupted(host.epoch_cell(), trap_out) {
                if *trap_out == 0 {
                    *trap_out = TrapKind::OutOfFuel as i64;
                }
                return false;
            }
            true
        }
        None => false,
    }
}

/// Start the root context's unwind for a fork: set the window's freeze word, which the call's
/// trailing poll reads. Only from an **empty** shadow stack — its SP word reads its frame base. The
/// leaf frame must land there, where the fork injects the reply ([`ShadowArena::leaf_reply`]); and a
/// barrier below the call holds it occupied ([`temen_durable::BARRIER_SP`]) when a `call.dyn`
/// reached the fork through an index the program never took, whose uninstrumented frame the unwind
/// could not resume. Read and written as a host op reads the guest's memory: a control word the guest
/// unmapped refuses the fork rather than faulting the host.
fn begin_fork_unwind(window: &mut dyn GuestMem, arena: ShadowArena) -> bool {
    let empty = window
        .read_bytes(arena.region_base(0), 8)
        .and_then(|b| <[u8; 8]>::try_from(b.as_slice()).ok())
        .is_some_and(|b| u64::from_le_bytes(b) == arena.frame_base(0));
    empty
        && window
            .write_bytes(STATE_OFF, &STATE_UNWINDING.to_le_bytes())
            .is_some()
}

/// Whether the run delivers #932 async signals (its compile armed the delivery check).
unsafe fn delivers_signals(trap_out: *mut i64) -> bool {
    !VmCtx::of_trap_out(trap_out).sig_armed.is_null()
}

/// Whether the call is inside a #932 injected signal handler — a frame the JIT's safepoint pushed,
/// which no durable unwind can save. (A process run arms no signal delivery today, so this is the
/// gate's statement of its precondition rather than a live case.)
unsafe fn in_signal_handler(trap_out: *mut i64) -> bool {
    let vm = VmCtx::of_trap_out(trap_out);
    delivers_signals(trap_out) && (*(vm.sig_ctx as *const JitSigDelivery)).depth.get() != 0
}

/// The tree's bell count for a thunk call of a JIT process, read before its op (see
/// [`serve_request`]); `0` outside one.
///
/// # Safety
/// As [`serve_request`].
pub(crate) unsafe fn bell_of(trap_out: *mut i64) -> u64 {
    proc_of(trap_out).map_or(0, |p| p.tree.rings.load(Ordering::SeqCst))
}

/// A process's outcome as the interpreters' result, for its reap status and trap record.
fn interp_result(
    r: &Result<JitRun, temen_jit::JitError>,
    results: &[ValType],
) -> Result<Vec<Value>, Trap> {
    match r {
        Ok(run) => match &run.outcome {
            JitOutcome::Returned(s) => Ok(results
                .iter()
                .zip(s)
                .map(|(t, &v)| crate::typed(*t, v))
                .collect()),
            JitOutcome::Exited(code) => Err(Trap::Exit(*code)),
            JitOutcome::Trapped(kind) => Err(trap_of(*kind)),
            JitOutcome::HostUnwound => Err(Trap::Malformed),
        },
        Err(_) => Err(Trap::Malformed),
    }
}

/// The interpreters' [`Trap`] for a JIT trap kind: the same trap wire code (#1735).
fn trap_of(kind: TrapKind) -> Trap {
    Trap::from_code(kind.code()).expect("every trap kind is a trap")
}

/// Run `m`'s `func` as the root of a process tree — process 1 over the embedder's `host` — then end
/// the tree with it. Returns the root's last run and the result types of the entry that produced it
/// (an exec'd command's, not the caller's). The embedder's `host` stays the powerbox it granted: an
/// image-replace swaps the running image's powerbox, not the caller's.
///
/// # Safety
/// `host` is live and exclusively the run's; `interrupt` (when `Some`) outlives the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_root(
    m: &Module,
    func: FuncIdx,
    slots: &[i64],
    host: &mut Host,
    interrupt: Option<&Arc<AtomicU64>>,
    quota: temen_jit::Quota,
    init_mem: Option<&[u8]>,
    snapshot_cap: Option<usize>,
) -> (Result<JitRun, temen_jit::JitError>, Vec<ValType>) {
    let tree = Tree::new(interrupt, quota);
    let start = Start::Fresh {
        program: Program::Embedder(m),
        entry: func,
        args: slots.to_vec(),
        init_mem: init_mem.map(<[u8]>::to_vec),
    };
    let (r, results, _) = run_process(&tree, ROOT_PID, host, start, snapshot_cap);
    tree.teardown();
    (r, results)
}
