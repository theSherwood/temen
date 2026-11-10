//! #2594 — a module whose functions the Cranelift JIT compiles at their first call, and ahead of it.
//!
//! Each function's record word ([`FnRecord`](super::FnRecord)) starts at the function's **lazy
//! entry**, a copy of its signature's template ([`build_entry`]). The entry asks [`resolve`] for its
//! function, which the service finds by the entry's own address. `resolve` returns once the
//! function's code is published into its word, and the entry then tail-calls the function through
//! its slot, as `return_call.dyn` does. So every word always holds code that runs its function: a
//! call through a word (lazily compiled code's direct calls, `call.dyn`, the trampolines, the host
//! paths that start a function from its record) needs no check of its own.
//!
//! The compiles run on workers kept for the module's life. A function a guest waits on comes first.
//! Then the workers compile what the compiled functions call directly, the most worth it first
//! ([`worth`]), so they walk the call graph ahead of the guest. A worker lowers and compiles without
//! a lock, then defines and finalizes the function into the module's
//! [`JITModule`](cranelift_jit::JITModule) under the arena's lock, and publishes its code with a
//! release store. cranelift-jit puts new code in fresh pages and never writes a finalized segment
//! again, so nothing that runs changes under the guest.
//!
//! Once nothing is queued, the workers compile the module again as an eager compile does, every
//! function with direct calls to the others (the **linked copy**). The worker that finishes it
//! defines and finalizes it in one step, under the lock, and publishes each function's linked code
//! into its word. From then on a call runs today's code at today's speed, while a frame that started
//! earlier keeps running the code it started in. A lazy compile publishes only over its function's
//! lazy entry, so it never replaces a linked copy published first.
//!
//! A guest thread never runs Cranelift. `resolve` blocks it while a worker compiles, as a host call
//! does, which keeps a compile off a fiber's small stack and out of the guest's signal state. A wait
//! ends when the function is published, when the domain ends, or when the host interrupts the run,
//! which `resolve` then raises as the guest's own poll would. A compile that fails ends the run with
//! its error (`COMPILE_FAILED`): what can fail mid-run is Cranelift itself, since every decline was
//! decided before the run (`ensure_supported`, invariant 9).

use super::{
    build_clif, call_host, compile_fn, ctx_args, emit_stack_check, emit_trap_propagate,
    indirect_dispatch, open_function, sig_from, uses_sret, validate, Arena, Callee, Callees,
    FnRecords, Func, FuncSlots, FuncType, HostFn, JitError, LowerPlan, PageEnv, TrapKind, VmCtx,
};
use cranelift_codegen::ir::{
    types::I64, ExtFuncData, ExternalName, Function, InstBuilder, UserExternalName, UserFuncName,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module, ModuleReloc, ModuleRelocTarget};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// How long a wait in [`resolve`] goes before it checks the run's end and the host's interrupt
/// again. Neither has a way to wake the wait, so it polls them, as a running guest does.
const POLL: Duration = Duration::from_millis(5);

/// Whether a compile may compile lazily, from `TEMEN_JIT_LAZY` (read once): `on` compiles every
/// module lazily that can be, `off` none, and anything else leaves it to the module's size
/// ([`MIN_FUNCS`]). The switch is for testing and measuring each way on the same build.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    Auto,
    On,
    Off,
}

pub(crate) fn mode() -> Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("TEMEN_JIT_LAZY").as_deref() {
        Ok("on") => Mode::On,
        Ok("off") => Mode::Off,
        _ => Mode::Auto,
    })
}

/// The fewest functions a module must have to compile lazily on its own (`Mode::Auto`). A smaller
/// module compiles whole about as fast as a lazy one stands up, so it gains nothing.
pub(crate) const MIN_FUNCS: usize = 64;

/// What a lazily compiled module's functions are lowered from and checked against, owned for the
/// module's life: everything a [`LowerPlan`] borrows, and the validator's facts (#2603).
pub(crate) struct Code {
    pub(crate) funcs: Vec<Func>,
    pub(crate) types: Vec<temen_ir::TypeEntry>,
    pub(crate) distinct: Vec<FuncType>,
    pub(crate) callees: Callees,
    /// The functions' ids, which the linked copy defines and links its direct calls to.
    pub(crate) ids: Vec<FuncId>,
    /// The ids each function's lazily compiled code is defined under ([`Entries::code`]).
    pub(crate) lazy_ids: Vec<FuncId>,
    pub(crate) nesting: bool,
    pub(crate) signals: bool,
    pub(crate) mask: u64,
    pub(crate) guard_offset: u64,
    pub(crate) epoch: bool,
    pub(crate) fuel: bool,
    pub(crate) domain_poll: bool,
    pub(crate) fn_table_mask: u64,
    pub(crate) window: Option<validate::WindowLayout>,
    pub(crate) table: u64,
    pub(crate) type_contracts: Vec<Option<validate::Contract>>,
    pub(crate) own: Vec<u32>,
    pub(crate) callee_contracts: Vec<Option<validate::Contract>>,
}

impl Code {
    /// The plan a function is lowered by: lazily compiled, its direct calls go through their callees'
    /// code words, since a callee may not be compiled yet; in the linked copy, they link.
    pub(crate) fn plan(&self, words: bool) -> LowerPlan<'_> {
        LowerPlan {
            callees: &self.callees,
            ids: &self.ids,
            funcs: &self.funcs,
            distinct: &self.distinct,
            type_section: &self.types,
            nesting: self.nesting,
            signals: self.signals,
            page: PageEnv::NONE,
            mask: self.mask,
            guard_offset: self.guard_offset,
            epoch: self.epoch,
            fuel: self.fuel,
            domain_poll: self.domain_poll,
            fn_table_mask: self.fn_table_mask,
            slots: FuncSlots::Own { words },
            srclocs: None,
            var_labels: None,
        }
    }

    /// What the machine-code validator checks the module's code against.
    pub(crate) fn guest(&self) -> validate::Guest<'_> {
        validate::Guest {
            window: self.window,
            table: self.table,
            types: &self.type_contracts,
            own: &self.own,
        }
    }

    /// The contract of the function a relocation names, for the validator.
    fn callee(&self, id: u32) -> Option<validate::Contract> {
        self.callee_contracts.get(id as usize).copied().flatten()
    }

    /// The count of result slots function `f` returns through its sret pointer, for the validator.
    fn sret(&self, f: usize) -> Option<u32> {
        let r = &self.funcs[f].results;
        uses_sret(r).then_some(r.len() as u32)
    }

    /// The size of function `f`, in instructions and terminators: about what its compile costs.
    fn size(&self, f: u32) -> usize {
        self.funcs[f as usize]
            .blocks
            .iter()
            .map(|b| b.insts.len() + 1)
            .sum()
    }

    /// The functions `f` calls directly: what its compile queues ahead of the guest.
    fn callees_of(&self, f: u32) -> impl Iterator<Item = u32> + '_ {
        self.funcs[f as usize].blocks.iter().flat_map(|b| {
            let calls = b.insts.iter().filter_map(|i| match i {
                temen_ir::Inst::Call { func, .. } => Some(*func),
                _ => None,
            });
            let tail = match &b.term {
                temen_ir::Terminator::ReturnCall { func, .. } => Some(*func),
                _ => None,
            };
            calls.chain(tail)
        })
    }
}

/// Where a function's compile stands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    Idle,
    Queued,
    Compiling,
    Done,
    Failed,
}

/// What the workers and the waiting guests share, under [`Lazy::state`]'s lock.
struct State {
    stage: Vec<Stage>,
    /// How many direct calls each function is from one a guest waited on: 0 for one it waited on,
    /// `u32::MAX` for one no compile has reached.
    depth: Vec<u32>,
    /// What guests wait on, in the order they asked.
    waited: VecDeque<u32>,
    /// What to compile ahead of the guest, the most worth it first ([`worth`]), and first queued
    /// first among equals. A function may be in it more than once, and a worker takes only an
    /// entry still `Queued`.
    ahead: BinaryHeap<(u64, Reverse<u64>, u32)>,
    /// How many entries `ahead` has been given: each one's place among equals.
    queued: u64,
    /// The workers compiling the linked copy's functions now.
    linking: usize,
    /// The most that may at once: one fewer than the workers, or the one there is. The guest runs
    /// meanwhile, on the code compiled lazily, and this leaves it a core: a run of a second or so
    /// on four cores is a tenth faster for it.
    linkers: usize,
    /// The module is being dropped: each worker stops before its next function.
    stop: bool,
    /// The first compile that failed. Every run that needs that function ends with it.
    failure: Option<String>,
    /// The linked copy's compile.
    linked: Linked,
}

/// How much compiling a function of `size` instructions ahead of the guest saves it, `depth` direct
/// calls from a function it waited on: the wait it would have, which is about the compile's time and
/// so its size, halved for each call between, since the guest is less likely to get there soon.
/// QuickJS's start reaches its parser's largest functions while the walk is still at their small
/// neighbours; ranking by size alone, or not at all, makes it wait for them.
fn worth(size: usize, depth: u32) -> u64 {
    (size as u64) >> depth.min(63)
}

/// The linked copy's compile: each function compiled again, with direct calls to the copy's other
/// functions, and held until the last one is compiled.
struct Linked {
    /// The order the functions compile in: the smallest first. A worker takes the linked copy's next
    /// function whenever nothing else is queued, which early in a run is only for a moment, and a
    /// guest's next wait then queues behind it: the largest function of QuickJS takes 90 ms.
    order: Vec<u32>,
    /// The next function a worker compiles, as an index into `order`.
    next: usize,
    /// The functions not compiled yet.
    left: usize,
    /// Each function's code and relocations, compiled.
    code: Vec<Option<Emitted>>,
    /// Whether a compile of it failed. The module then keeps its lazily compiled code.
    failed: bool,
}

/// A function's compiled bytes, their alignment, and their relocations.
type Emitted = (Vec<u8>, u64, Vec<ModuleReloc>);

/// What a worker takes next.
enum Job {
    /// Compile function `f` lazily, which a guest waits on or will soon call.
    Lazy(u32),
    /// Compile function `f` for the linked copy.
    Linked(u32),
}

/// A lazily compiled module's compile service: what its workers compile from and publish into, and
/// what a waiting guest waits on.
pub(crate) struct Lazy {
    code: Code,
    isa: cranelift_codegen::isa::OwnedTargetIsa,
    arena: Arc<Arena>,
    records: Arc<FnRecords>,
    /// Each lazy entry's address, with the function it runs, sorted by address.
    entries: Vec<(u64, u32)>,
    /// Each function's lazy entry's address, by function.
    entry_of: Vec<u64>,
    state: Mutex<State>,
    /// Signalled when work is queued, and when the module is dropped: what idle workers wait on.
    work: Condvar,
    /// Signalled when a compile finishes: what a guest in [`resolve`] waits on.
    done: Condvar,
}

impl Lazy {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The function the lazy entry at `entry` runs.
    fn func_of(&self, entry: u64) -> Option<u32> {
        let i = self.entries.partition_point(|&(a, _)| a < entry);
        self.entries
            .get(i)
            .filter(|&&(a, _)| a == entry)
            .map(|&(_, f)| f)
    }

    /// Wait until function `f`'s code is published, asking a worker for it first. Returns early,
    /// having raised what ends the run, if the compile failed or the host interrupted the run, or
    /// once the domain has ended.
    fn wait(&self, f: u32, ctx: &VmCtx) {
        let i = f as usize;
        let mut st = self.lock();
        let mut asked = false;
        loop {
            match st.stage[i] {
                Stage::Done => return,
                Stage::Failed => {
                    ctx.raise(temen_ir::trap_code::COMPILE_FAILED);
                    return;
                }
                Stage::Idle | Stage::Queued if !asked => {
                    st.stage[i] = Stage::Queued;
                    st.depth[i] = 0;
                    st.waited.push_back(f);
                    asked = true;
                    self.work.notify_all();
                }
                // Its callees are what the guest needs next.
                Stage::Compiling if !asked => {
                    st.depth[i] = 0;
                    asked = true;
                }
                _ => {}
            }
            if ctx.ended().is_some() {
                return;
            }
            // SAFETY: a context's kill door outlives every run of its instance.
            if !ctx.epoch.is_null() && unsafe { (*ctx.epoch).fired() } {
                ctx.raise(TrapKind::OutOfFuel as i64);
                return;
            }
            st = self
                .done
                .wait_timeout(st, POLL)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
    }

    /// Lower and compile function `f`: lazily (through code words, under its lazy id) or for the
    /// linked copy (linked, under its own id).
    fn emit(&self, f: usize, words: bool) -> Result<Emitted, String> {
        let code = &self.code;
        let id = if words { code.lazy_ids[f] } else { code.ids[f] };
        let plan = code.plan(words);
        let lower = |ctx: &mut cranelift_codegen::Context| {
            build_clif(&plan, &mut ctx.func, &code.funcs[f], f as u32)
        };
        let callee = |id: u32| code.callee(id);
        let (compiled, relocs) =
            compile_fn(&*self.isa, id, lower, &code.guest(), &callee, code.sret(f))
                .map_err(|_| format!("compiling function {f} panicked"))?
                .map_err(|e| format!("compiling function {f}: {e:?}"))?;
        Ok((
            compiled.code_buffer().to_vec(),
            compiled.buffer.alignment as u64,
            relocs,
        ))
    }

    /// Compile function `f` lazily, define it, and publish it over its lazy entry.
    fn compile(&self, f: u32) -> Result<(), String> {
        let i = f as usize;
        let (bytes, align, relocs) = self.emit(i, true)?;
        let id = self.code.lazy_ids[i];
        let addr = {
            let mut jit = self.arena.lock();
            jit.define_function_bytes(id, align, &bytes, &relocs)
                .map_err(|e| e.to_string())?;
            jit.finalize_definitions().map_err(|e| e.to_string())?;
            jit.get_finalized_function(id) as u64
        };
        // Not over a linked copy that was published first.
        self.records.own[i].replace(self.entry_of[i], addr);
        Ok(())
    }

    /// Define and finalize the linked copy, every function at once since each links to the others,
    /// and publish each function's code into its word.
    fn publish_linked(&self, code: Vec<Emitted>) -> Result<(), String> {
        let ids = &self.code.ids;
        let mut jit = self.arena.lock();
        for (&id, (bytes, align, relocs)) in ids.iter().zip(&code) {
            jit.define_function_bytes(id, *align, bytes, relocs)
                .map_err(|e| e.to_string())?;
        }
        jit.finalize_definitions().map_err(|e| e.to_string())?;
        for (record, &id) in self.records.own.iter().zip(ids) {
            record.publish(jit.get_finalized_function(id) as u64);
        }
        Ok(())
    }

    /// The first compile that failed, if any did.
    pub(crate) fn failure(&self) -> Option<String> {
        self.lock().failure.clone()
    }
}

/// A worker: compile what guests wait on, then what they will soon call, then the linked copy,
/// until the copy is published or the module is dropped.
fn work(lazy: Arc<Lazy>) {
    loop {
        let job = {
            let mut st = lazy.lock();
            loop {
                if st.stop {
                    return;
                }
                if let Some(job) = take(&mut st) {
                    break job;
                }
                if st.linked.left == 0 {
                    // Every function runs its linked code now: no guest waits on a compile again.
                    return;
                }
                st = lazy.work.wait(st).unwrap_or_else(|p| p.into_inner());
            }
        };
        match job {
            Job::Lazy(f) => {
                let out = lazy.compile(f);
                let mut st = lazy.lock();
                let i = f as usize;
                match out {
                    Ok(()) => {
                        st.stage[i] = Stage::Done;
                        // What it calls, the guest may call next.
                        let d = st.depth[i].saturating_add(1);
                        for g in lazy.code.callees_of(f) {
                            let j = g as usize;
                            if matches!(st.stage[j], Stage::Idle | Stage::Queued) && d < st.depth[j]
                            {
                                st.stage[j] = Stage::Queued;
                                st.depth[j] = d;
                                st.queued += 1;
                                let key = (worth(lazy.code.size(g), d), Reverse(st.queued), g);
                                st.ahead.push(key);
                            }
                        }
                    }
                    Err(e) => {
                        st.stage[i] = Stage::Failed;
                        st.failure.get_or_insert(e);
                    }
                }
                drop(st);
                lazy.work.notify_all();
                lazy.done.notify_all();
            }
            Job::Linked(f) => {
                let out = lazy.emit(f as usize, false);
                let mut st = lazy.lock();
                st.linking -= 1;
                lazy.work.notify_one();
                let Ok(code) = out else {
                    // The lazily compiled code stays: it runs the function as well.
                    st.linked.failed = true;
                    drop(st);
                    lazy.done.notify_all();
                    continue;
                };
                st.linked.code[f as usize] = Some(code);
                st.linked.left -= 1;
                if st.linked.left > 0 {
                    continue;
                }
                let code: Option<Vec<Emitted>> =
                    st.linked.code.iter_mut().map(Option::take).collect();
                drop(st);
                let published = code
                    .ok_or(String::new())
                    .and_then(|c| lazy.publish_linked(c));
                let mut st = lazy.lock();
                match published {
                    Ok(()) => {
                        // Every function runs its linked code now, so no guest waits on a compile.
                        st.stage.iter_mut().for_each(|s| {
                            if *s != Stage::Failed {
                                *s = Stage::Done
                            }
                        });
                        st.waited.clear();
                        st.ahead.clear();
                    }
                    Err(_) => {
                        st.linked.failed = true;
                        st.linked.left = usize::MAX;
                    }
                }
                drop(st);
                lazy.work.notify_all();
                lazy.done.notify_all();
            }
        }
    }
}

/// The next job, marked as taken: what a guest waits on, else what is most worth compiling ahead of
/// the guest, else the linked copy's next function, while fewer than `linkers` compile it.
fn take(st: &mut State) -> Option<Job> {
    while let Some(f) = st
        .waited
        .pop_front()
        .or_else(|| st.ahead.pop().map(|(_, _, f)| f))
    {
        // An entry whose function an earlier entry took is spent.
        if st.stage[f as usize] == Stage::Queued {
            st.stage[f as usize] = Stage::Compiling;
            return Some(Job::Lazy(f));
        }
    }
    if st.linking >= st.linkers {
        return None;
    }
    let l = &mut st.linked;
    if l.failed || l.next == l.order.len() {
        return None;
    }
    l.next += 1;
    st.linking += 1;
    Some(Job::Linked(l.order[l.next - 1]))
}

/// A lazily compiled module's compile service, which its arena owns ([`super::OwnedJit`]): when the
/// last instance of the code drops it, its workers stop after the compile each is in. A worker
/// holds the arena, which is freed with the last of them: nothing runs the code by then, and a
/// compile can take a hundred milliseconds, which a dropping module need not wait out.
pub(crate) struct Service {
    lazy: Arc<Lazy>,
}

impl Service {
    /// Start the service of `code`, whose functions' records are `records` and whose lazy entries
    /// are at `entries` (an address and the function it runs, in any order). Nothing is compiled
    /// until a guest enters a function.
    pub(crate) fn start(
        code: Code,
        isa: cranelift_codegen::isa::OwnedTargetIsa,
        arena: Arc<Arena>,
        records: Arc<FnRecords>,
        mut entries: Vec<(u64, u32)>,
    ) -> Service {
        let n = code.funcs.len();
        let mut entry_of = vec![0; n];
        for &(addr, f) in &entries {
            entry_of[f as usize] = addr;
        }
        entries.sort_unstable();
        let workers = std::thread::available_parallelism()
            .map_or(1, |p| p.get())
            .clamp(1, n.max(1));
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&f| code.size(f));
        let lazy = Arc::new(Lazy {
            code,
            isa,
            arena,
            records,
            entries,
            entry_of,
            state: Mutex::new(State {
                stage: vec![Stage::Idle; n],
                depth: vec![u32::MAX; n],
                waited: VecDeque::new(),
                ahead: BinaryHeap::new(),
                queued: 0,
                linking: 0,
                linkers: workers.saturating_sub(1).max(1),
                stop: false,
                failure: None,
                linked: Linked {
                    order,
                    next: 0,
                    left: n,
                    code: (0..n).map(|_| None).collect(),
                    failed: false,
                },
            }),
            work: Condvar::new(),
            done: Condvar::new(),
        });
        for _ in 0..workers {
            let lazy = Arc::clone(&lazy);
            std::thread::Builder::new()
                .name("temen-jit-lazy".into())
                .spawn(move || work(lazy))
                .expect("a lazy compile worker");
        }
        Service { lazy }
    }

    /// The service a lazy entry's [`resolve`] reaches through its context ([`VmCtx::lazy`]).
    pub(crate) fn as_ptr(&self) -> *const core::ffi::c_void {
        Arc::as_ptr(&self.lazy) as *const core::ffi::c_void
    }

    /// The first compile that failed, if any did.
    pub(crate) fn failure(&self) -> Option<String> {
        self.lazy.failure()
    }

    /// Wait until the linked copy is published, or fails to compile: whether it was published.
    pub(crate) fn wait_linked(&self) -> bool {
        let mut st = self.lazy.lock();
        loop {
            if st.linked.left == 0 {
                return true;
            }
            if st.linked.failed {
                return false;
            }
            st = self.lazy.done.wait(st).unwrap_or_else(|p| p.into_inner());
        }
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.lazy.lock().stop = true;
        self.lazy.work.notify_all();
        self.lazy.done.notify_all();
    }
}

/// [`HostFn::Resolve`]: the function the lazy entry at `entry` runs, once its code is published
/// (see the module docs). A compile that failed, or an interrupt while the guest waited, is raised
/// on the domain here; either way the entry's check after this call then unwinds.
///
/// # Safety
/// `vmctx` is the calling instance's context, whose [`VmCtx::lazy`] is its code's service, and
/// `entry` is one of that service's lazy entries.
pub(crate) unsafe extern "C" fn resolve(vmctx: *const VmCtx, entry: u64) -> u32 {
    let ctx = &*vmctx;
    let lazy = &*(ctx.lazy as *const Lazy);
    match lazy.func_of(entry) {
        Some(f) => {
            lazy.wait(f, ctx);
            f
        }
        None => {
            // Only a lazy entry calls this, with its own address: anything else is a host bug,
            // which ends the run rather than run on.
            ctx.raise(temen_ir::trap_code::COMPILE_FAILED);
            0
        }
    }
}

/// A lazily compiled module's lazy entries: a template per signature of its functions, and a copy
/// of its signature's template for each function, the code the function's word starts at.
pub(crate) struct Entries {
    /// The distinct signatures of the module's functions.
    sigs: Vec<FuncType>,
    /// Each signature's template.
    pub(crate) templates: Vec<FuncId>,
    /// Each function's signature, as an index into `sigs`.
    sig_of: Vec<usize>,
    /// Each function's copy.
    pub(crate) copies: Vec<FuncId>,
    /// The id each function's lazily compiled code is defined under. The function's own id is the
    /// linked copy's.
    pub(crate) code: Vec<FuncId>,
}

impl Entries {
    /// Declare the templates and the copies of `funcs`' entries in `module`.
    pub(crate) fn declare(
        module: &mut cranelift_jit::JITModule,
        funcs: &[Func],
    ) -> Result<Entries, JitError> {
        let backend = |e: cranelift_module::ModuleError| JitError::Backend(e.to_string());
        let mut sigs: Vec<FuncType> = Vec::new();
        let mut sig_of = Vec::with_capacity(funcs.len());
        for f in funcs {
            let ty = FuncType {
                params: f.params.clone(),
                results: f.results.clone(),
            };
            let i = sigs.iter().position(|s| *s == ty).unwrap_or_else(|| {
                sigs.push(ty);
                sigs.len() - 1
            });
            sig_of.push(i);
        }
        let templates = sigs
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let sig = sig_from(&ty.params, &ty.results);
                module
                    .declare_function(&format!("lazy_entry{i}"), Linkage::Local, &sig)
                    .map_err(backend)
            })
            .collect::<Result<_, _>>()?;
        let mut declare = |prefix: &str| {
            funcs
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let sig = sig_from(&f.params, &f.results);
                    module
                        .declare_function(&format!("{prefix}{i}"), Linkage::Local, &sig)
                        .map_err(backend)
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let copies = declare("lazy_entry_of")?;
        let code = declare("lazy_code")?;
        Ok(Entries {
            sigs,
            templates,
            sig_of,
            copies,
            code,
        })
    }

    /// Compile the templates on the compile's workers (`plan` lowers through code words, and
    /// `guest` is what the validator checks them against), then define each function's copy: the
    /// template's bytes, with its reference to itself relocated to the copy. Returns the bytes of
    /// code defined.
    pub(crate) fn define(
        &self,
        module: &mut cranelift_jit::JITModule,
        plan: &LowerPlan,
        guest: &validate::Guest,
    ) -> Result<usize, JitError> {
        let mut built: Vec<Option<Emitted>> = (0..self.templates.len()).map(|_| None).collect();
        let mut bytes = super::define_funcs(
            module,
            &self.templates,
            &mut super::Codegen::Compile(None),
            |ctx, t| build_entry(plan, &mut ctx.func, &self.sigs[t], self.templates[t]),
            |_, code, relocs, t| {
                built[t] = Some((
                    code.code_buffer().to_vec(),
                    code.buffer.alignment as u64,
                    relocs.to_vec(),
                ))
            },
            guest,
            &|t: usize| {
                let r = &self.sigs[t].results;
                uses_sret(r).then_some(r.len() as u32)
            },
        )?;
        for (f, &copy) in self.copies.iter().enumerate() {
            let t = self.sig_of[f];
            let (code, align, relocs) = built[t].as_ref().ok_or(JitError::Malformed)?;
            let template = self.templates[t].as_u32();
            let relocs: Vec<ModuleReloc> = relocs
                .iter()
                .map(|r| ModuleReloc {
                    offset: r.offset,
                    kind: r.kind,
                    name: match r.name {
                        ModuleRelocTarget::User {
                            namespace: 0,
                            index,
                        } if index == template => ModuleRelocTarget::user(0, copy.as_u32()),
                        ref other => other.clone(),
                    },
                    addend: r.addend,
                })
                .collect();
            module
                .define_function_bytes(copy, *align, code, &relocs)
                .map_err(|e| JitError::Backend(e.to_string()))?;
            bytes += code.len();
        }
        Ok(bytes)
    }
}

/// Build the lazy entry of signature `ty`, the template each function of `ty` gets a copy of: the
/// code its word holds until its compile publishes its code. `self_id` is the template's own id,
/// whose address it passes to [`resolve`]; each copy relocates that one reference to itself.
///
/// The entry checks the stack, since it calls the host, but neither charges fuel nor polls: the
/// function's own prologue does both once the entry tail-calls it, exactly as a direct call to it
/// would have.
pub(crate) fn build_entry(
    plan: &LowerPlan,
    clif: &mut Function,
    ty: &FuncType,
    self_id: FuncId,
) -> Result<(), JitError> {
    clif.signature = sig_from(&ty.params, &ty.results);
    clif.name = UserFuncName::user(0, 0);
    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);
    let (lower, params) = open_function(&mut b, plan, &ty.params, &ty.results, 0)?;
    emit_stack_check(&mut b, &lower);
    // Its own address, by an absolute reference (a `movabs` of a symbol's address), which each copy
    // relocates to itself.
    let signature = b.func.import_signature(sig_from(&ty.params, &ty.results));
    let name = b.func.declare_imported_user_function(UserExternalName {
        namespace: 0,
        index: self_id.as_u32(),
    });
    let me = b.func.import_function(ExtFuncData {
        name: ExternalName::user(name),
        signature,
        colocated: false,
        patchable: false,
    });
    let addr = b.ins().func_addr(I64, me);
    let vmctx = b.use_var(lower.vmctx_var);
    let call = call_host(plan.callees, &mut b, HostFn::Resolve, &[vmctx, addr])?;
    let func = b.inst_results(call)[0];
    // A failed compile, an interrupt, or another vCPU's end stops the entry here.
    emit_trap_propagate(&mut b, &lower, Callee::Host);
    let code = indirect_dispatch(&mut b, &lower, func, ty);
    let sig = b.import_signature(sig_from(&ty.params, &ty.results));
    let mut args = ctx_args(&mut b, &lower);
    if let Some(sret) = lower.sret_var {
        args.push(b.use_var(sret));
    }
    args.extend(params);
    b.ins().return_call_indirect(sig, code, &args);
    lower.exits.fill(&mut b, &lower);
    b.seal_all_blocks();
    b.finalize();
    Ok(())
}
