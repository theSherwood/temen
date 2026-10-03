//! **A run as data** (#1720, decision C of #1717): the nodes a generated root guest spawns, each as a
//! detached §14 child in its own window, and the capabilities each is granted **by name** — plus the
//! one generator that turns that plan into the root's Temen IR.
//!
//! The root is ordinary guest code. Every spawn and grant it makes goes through an op-17 spawn record
//! and a grant record, so it is verified and confined like any other guest, and the generator that writes it is
//! not trusted: a wrong plan produces a root the verifier or the spawn refuses, never authority the
//! host did not grant. What the root may hand out is exactly what [`Plan::root_args`] granted it.
//!
//! The plan is also the document a guest graph-runner would read (#1717, option B), so moving the
//! generation into a guest later changes where this runs, not what it reads.

use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_ir::Module;

/// The root's own window: grant records, the capability names they point at, and each node's argv
/// payload, all above the NULL guard (#1094). A plan that does not fit is refused, not truncated.
pub const ROOT_WINDOW_LOG2: u8 = 16;

const GUARD: u64 = temen_ir::POWERBOX_NULL_GUARD;
/// Each pipe's `fds` pair as the mint self-op writes it (`[read: i32][write: i32]`, 8 bytes).
const PIPE_BASE: u64 = GUARD + 512;
/// Grant records (16 bytes each: `{name_off: u32, name_len: u32, handle: u32, flags: u32}`).
const REC_BASE: u64 = GUARD + 1024;
/// One 16-byte name slot per root capability, shared by every record that grants it, then two per
/// pipe (the names its ends are granted under).
const NAME_BASE: u64 = GUARD + 2048;
/// One op-17 spawn record per node ([`temen_ir::SPAWN_REC_LEN`] bytes each).
const SPAWN_BASE: u64 = GUARD + 4096;
/// The nodes' argv payloads, back to back.
const ARGV_BASE: u64 = GUARD + 8192;
const SLOT: u64 = 16;

/// A run: the root's own capabilities, and the nodes it spawns and joins.
pub struct Plan {
    /// The capabilities the root holds, by name, in the order it takes them (after its budget).
    /// A node reaches one only if its [`Node::grants`] names it.
    pub caps: Vec<String>,
    /// Spawned in order, then joined in order; the root returns the last node's result.
    pub nodes: Vec<Node>,
    /// Stream edges between nodes.
    pub pipes: Vec<Pipe>,
}

/// Where a grant record's handle comes from in the root: one of its params, or an `i32` in its window
/// (a pipe end the mint wrote).
enum Handle {
    Param(usize),
    At(u64),
}

/// A stream edge (#1807): a pipe the root mints, whose write end node `from` is granted under
/// `from_name` and whose read end node `to` is granted under `to_name` — typically `"stdout"` to
/// `"stdin"`, so each node runs unchanged. The root closes its own copies of both ends once every node
/// is spawned, so the reader sees EOF when the writer exits. A pipe parks its reader (or a writer on a
/// full pipe), so a plan with pipes needs a scheduler that parks: the cooperative bytecode engine.
pub struct Pipe {
    pub from: usize,
    pub from_name: String,
    pub to: usize,
    pub to_name: String,
}

/// One §14 child: a module the host grants the root (func 0 is its entry), run detached (op 17 v1) in
/// a fresh window paid from the root's budget.
pub struct Node {
    /// The node's window, `1 << window_log2` bytes.
    pub window_log2: u8,
    /// Carried as the spawn-time args payload, seeded at the child's `module_args_base()`.
    pub argv: Vec<String>,
    /// The node's §3e environment (`KEY=VALUE` entries), in the same payload after `argv`.
    pub env: Vec<Vec<u8>>,
    /// The root capabilities this node is granted, by name: its whole powerbox.
    pub grants: Vec<String>,
}

impl Plan {
    /// One node granted every root capability, in order: the nim phase driver's shape.
    pub fn single(window_log2: u8, argv: &[&str], caps: &[&str]) -> Plan {
        let owned = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        Plan {
            caps: owned(caps),
            nodes: vec![Node {
                window_log2,
                argv: owned(argv),
                env: Vec::new(),
                grants: owned(caps),
            }],
            pipes: Vec::new(),
        }
    }

    /// The root guest's Temen IR text. Its params are `(inst, module_0 … module_{n-1}, budget, cap_0 …)`
    /// — the order [`root_args`](Self::root_args) builds. `Err` names what does not fit or resolve.
    pub fn root_src(&self) -> Result<String, String> {
        let n = self.nodes.len();
        if n == 0 {
            return Err("a plan spawns at least one node".into());
        }
        // Name slots: each root capability's, then each pipe's two end names.
        let names: Vec<&String> = self
            .caps
            .iter()
            .chain(self.pipes.iter().flat_map(|p| [&p.from_name, &p.to_name]))
            .collect();
        if NAME_BASE + names.len() as u64 * SLOT > SPAWN_BASE {
            return Err(format!("{} capability names do not fit", names.len()));
        }
        if SPAWN_BASE + n as u64 * temen_ir::SPAWN_REC_LEN as u64 > ARGV_BASE {
            return Err(format!("{n} nodes' spawn records do not fit"));
        }
        if PIPE_BASE + self.pipes.len() as u64 * 8 > REC_BASE {
            return Err(format!("{} pipes do not fit", self.pipes.len()));
        }
        if self.pipes.iter().any(|p| p.from >= n || p.to >= n) {
            return Err("a pipe names a node the plan does not have".into());
        }
        // Plain ASCII so the name is its own data literal.
        let mut data = String::new();
        for (i, name) in names.iter().enumerate() {
            let ok = !name.is_empty()
                && name.len() as u64 <= SLOT
                && name
                    .bytes()
                    .all(|b| b.is_ascii_graphic() && b != b'"' && b != b'\\');
            if !ok {
                return Err(format!(
                    "capability name {name:?} is not a short ASCII word"
                ));
            }
            data.push_str(&format!(
                "data {} \"{name}\"\n",
                NAME_BASE + i as u64 * SLOT
            ));
        }
        // Each node's §3e args payload: `[argc: u32][envc: u32]`, then its NUL-terminated args and env.
        let mut argv_at = Vec::with_capacity(n);
        let mut off = ARGV_BASE;
        for node in &self.nodes {
            let args: Vec<&[u8]> = node.argv.iter().map(|a| a.as_bytes()).collect();
            let env: Vec<&[u8]> = node.env.iter().map(Vec::as_slice).collect();
            let blob = temen_ir::write_args_blob(&args, &env);
            let esc: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
            data.push_str(&format!("data {off} \"{esc}\"\n"));
            argv_at.push((off, blob.len()));
            off += blob.len() as u64;
        }
        if off > 1u64 << ROOT_WINDOW_LOG2 {
            return Err("the nodes' argv does not fit the root's window".into());
        }
        // Mint every pipe first (the self-op writes its `[read][write]` handles at its fds slot). A
        // failed mint leaves the slot zeroed — the root's own Instantiator, which no grant can carry —
        // so the spawn that names it refuses: fail closed.
        let mut records = String::new();
        for p in 0..self.pipes.len() {
            records.push_str(&format!(
                "  pf{p} = i64.const {fds}\n  pz{p} = i32.const 0\n  \
                 pm{p} = call.cap 4294967295 16 (i64) -> (i32) pz{p} (pf{p})\n",
                fds = PIPE_BASE + p as u64 * 8,
            ));
        }
        // Grant records, each node's contiguous: word0 = {name_off | name_len << 32}, then the handle
        // (the root's param for a capability, or a pipe end read back from its fds slot).
        let mut first_rec = Vec::with_capacity(n);
        let mut rec_n = Vec::with_capacity(n);
        let mut r = 0u64;
        for (k, node) in self.nodes.iter().enumerate() {
            first_rec.push(REC_BASE + r * SLOT);
            // (name slot, handle): the node's root capabilities, then its pipe ends.
            let mut grants: Vec<(usize, Handle)> = Vec::new();
            for name in &node.grants {
                let Some(i) = self.caps.iter().position(|c| c == name) else {
                    return Err(format!(
                        "a node is granted {name:?}, which the root does not hold"
                    ));
                };
                grants.push((i, Handle::Param(2 + n + i)));
            }
            for (p, pipe) in self.pipes.iter().enumerate() {
                let slot = self.caps.len() + 2 * p;
                let fds = PIPE_BASE + p as u64 * 8;
                if pipe.from == k {
                    grants.push((slot, Handle::At(fds + 4)));
                }
                if pipe.to == k {
                    grants.push((slot + 1, Handle::At(fds)));
                }
            }
            let mut seen = std::collections::BTreeSet::new();
            for (slot, handle) in &grants {
                let name = names[*slot];
                if !seen.insert(name) {
                    return Err(format!("a node is granted {name:?} twice"));
                }
                let roff = REC_BASE + r * SLOT;
                let w0 = (NAME_BASE + *slot as u64 * SLOT) | ((name.len() as u64) << 32);
                let (load, hv) = match handle {
                    Handle::Param(i) => (String::new(), format!("v{i}")),
                    Handle::At(at) => (
                        format!("  la{roff} = i64.const {at}\n  hv{roff} = i32.load la{roff}\n"),
                        format!("hv{roff}"),
                    ),
                };
                records.push_str(&format!(
                    "  xr{roff} = i64.const {w0}\n  or{roff} = i64.const {roff}\n  i64.store or{roff} xr{roff}\n\
                     {load}  h{roff} = i64.extend_i32_u {hv}\n  oh{roff} = i64.const {hoff}\n  \
                     i64.store oh{roff} h{roff}\n",
                    hoff = roff + 8,
                ));
                r += 1;
            }
            rec_n.push(grants.len());
        }
        if REC_BASE + r * SLOT > NAME_BASE {
            return Err(format!("{r} grant records do not fit"));
        }
        // Spawn every node (op 17 v1: its own window, paid from the budget), then join every node.
        let sfx = |k: usize| {
            if k == 0 {
                String::new()
            } else {
                format!("_{k}")
            }
        };
        let mut body = String::new();
        for (k, node) in self.nodes.iter().enumerate() {
            let s = sfx(k);
            let at = SPAWN_BASE + k as u64 * temen_ir::SPAWN_REC_LEN as u64;
            let rec = temen_ir::SpawnRec {
                size_log2: node.window_log2 as i64,
                grants_ptr: first_rec[k],
                grants_n: rec_n[k] as u64,
                args: (argv_at[k].0, argv_at[k].1 as u64),
                ..temen_ir::SpawnRec::v1(0)
            };
            let (seg, stores) = spawn_rec_ir(
                at,
                &rec,
                Some(&format!("v{}", 1 + k)),
                &format!("v{}", 1 + n),
            );
            data.push_str(&seg);
            body.push_str(&format!(
                "{stores}  vrp{s} = i64.const {at}\n  vh{s} = call.cap 6 17 (i64) -> (i32) v0 (vrp{s})\n"
            ));
        }
        // Every node holds its own ends now; drop the root's, so a reader sees EOF when its writer exits.
        for p in 0..self.pipes.len() {
            let fds = PIPE_BASE + p as u64 * 8;
            for (end, at) in [("r", fds), ("w", fds + 4)] {
                body.push_str(&format!(
                    "  ca{end}{p} = i64.const {at}\n  ch{end}{p} = i32.load ca{end}{p}\n  \
                     cc{end}{p} = call.cap 0 2 () -> (i64) ch{end}{p} ()\n"
                ));
            }
        }
        for k in 0..n {
            let s = sfx(k);
            body.push_str(&format!(
                "  vr{s} = call.cap 6 1 (i32) -> (i64) v0 (vh{s})\n"
            ));
        }
        let params = 2 + n + self.caps.len();
        let sig = vec!["i32"; params].join(", ");
        let bparams = (0..params)
            .map(|i| format!("v{i}: i32"))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!(
            "memory {ROOT_WINDOW_LOG2}\n{data}func ({sig}) -> (i64) {{\nblock 0 ({bparams}) {{\n{records}{body}  \
             return vr{last}\n  }}\n}}\n",
            last = sfx(n - 1),
        ))
    }

    /// Grant the root what it takes besides `caps` — an Instantiator over its own window, one `Module`
    /// per node (`modules`, in node order) and the budget every node's window is minted from — and
    /// return its args in the order [`root_src`](Self::root_src)'s params declare: `inst`, the modules,
    /// the budget, then `caps` (already granted on `host`, one handle per [`Plan::caps`] name).
    ///
    /// The budget is unbounded on its own level: it pays for each node's window and for what the node
    /// grows past it (#1909), so a node grows as freely as it would at the root, bounded by its own
    /// memory, not by its declared size.
    pub fn root_args(&self, host: &mut Host, modules: &[&Module], caps: &[i32]) -> Vec<Value> {
        debug_assert_eq!(modules.len(), self.nodes.len());
        debug_assert_eq!(caps.len(), self.caps.len());
        let inst = host.grant_instantiator(0, 1u64 << ROOT_WINDOW_LOG2);
        let mods: Vec<i32> = modules.iter().map(|m| host.grant_module(m)).collect();
        let budget = host.grant_budget(-1, -1, -1);
        std::iter::once(inst)
            .chain(mods)
            .chain(std::iter::once(budget))
            .chain(caps.iter().copied())
            .map(Value::I32)
            .collect()
    }
}

/// An op-17 spawn record `rec` at window offset `at` of a guest being generated: the data segment that
/// holds its bytes, and the stores that fill its `module` field (unless `modh` is `None`: the record's
/// own, e.g. `-1` for the spawner's module) and its `budget` field from `i32` values (handles are only
/// known at run time).
pub fn spawn_rec_ir(
    at: u64,
    rec: &temen_ir::SpawnRec,
    modh: Option<&str>,
    budget: &str,
) -> (String, String) {
    let esc: String = rec.encode().iter().map(|b| format!("\\x{b:02x}")).collect();
    let mut stores = String::new();
    if let Some(modh) = modh {
        stores.push_str(&format!(
            "  rm{at} = i64.const {m}\n  i32.store rm{at} {modh}\n",
            m = at + 24
        ));
    }
    stores.push_str(&format!(
        "  rb{at} = i64.const {b}\n  i32.store rb{at} {budget}\n",
        b = at + 28
    ));
    (format!("data {at} \"{esc}\"\n"), stores)
}

/// Run `plan` to completion on the resumable interpreter (the native / single-threaded driver,
/// [`drive_op13`]): generate the root, grant it [`Plan::root_args`] over `host` (which already holds
/// `caps`), and return what the root returns — the last node's result.
pub fn run(
    plan: &Plan,
    modules: &[&Module],
    mut host: Host,
    caps: &[i32],
) -> Result<Vec<Value>, Trap> {
    // This driver runs each child to completion as it is spawned, so a pipe's reader could never
    // wait for its writer: refuse a plan with pipes up front rather than trap mid-run.
    if !plan.pipes.is_empty() {
        return Err(Trap::Malformed);
    }
    let src = plan.root_src().map_err(|_| Trap::Malformed)?;
    let root = temen_text::parse_module(&src).map_err(|_| Trap::Malformed)?;
    let prog = bytecode::VcpuProgram::compile(&root).ok_or(Trap::Malformed)?;
    let args = plan.root_args(&mut host, modules, caps);
    let win = 1usize << ROOT_WINDOW_LOG2;
    let layout = std::alloc::Layout::from_size_align(win, 8).map_err(|_| Trap::Malformed)?;
    // SAFETY: non-zero 8-aligned layout; owned here until the dealloc below, after the root vCPU and
    // every region view over it are dropped.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    if base.is_null() {
        return Err(Trap::Malformed);
    }
    let back = std::sync::Arc::new(unsafe { Region::shared(base, win as u64) });
    let out = bytecode::Vcpu::new_root_with_powerbox(
        &prog,
        0,
        &args,
        std::sync::Arc::clone(&back),
        &[],
        host,
    )
    .and_then(|root| drive_op13(&prog, root));
    drop(back);
    // SAFETY: same layout; the root vCPU and its region views are dropped above.
    unsafe { std::alloc::dealloc(base, layout) };
    out
}

/// The resumable-engine drive loop (mirrors `temen-run/tests/child_entry_fs.rs`), for a root (see
/// [`run`]) or a child a driver runs on the interpreter. The engine admits every spawn, its powerbox
/// built; this loop only starts the child. On a **detached** spawn (#1288) it mints the fresh window's
/// backing, a root-sized lazily-reserved `Region::new` (an `mmap` natively; the sparse `Paged`
/// fallback on wasm32), which the engine seeds (committed window = the declared size, starter caps
/// over the reservation, so `vm_map` grows it), and drives the child to completion — at any depth.
/// `Join` delivers the child's result. A carve child (retired, #1289) is the driver's decline.
pub(crate) fn drive_op13<'p>(
    prog: &'p bytecode::VcpuProgram,
    mut vcpu: bytecode::Vcpu<'p>,
) -> Result<Vec<Value>, Trap> {
    let mut children: Vec<Option<Result<Vec<Value>, Trap>>> = Vec::new();
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(v) => return Ok(v),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::InstantiateDetached { .. } => {
                // The fresh window's backing, a lazy reservation; the engine seeds it.
                let back = std::sync::Arc::new(Region::new(
                    1u64 << temen_ir::DEFAULT_RESERVED_LOG2,
                    temen_interp::host_page_size(),
                ));
                let r = vcpu
                    .take_child()
                    .ok_or(Trap::Malformed)
                    .and_then(|c| c.start(prog, back, None))
                    .and_then(|c| drive_op13(prog, c));
                let token = children.len() as u64;
                children.push(Some(r));
                vcpu.deliver_child(token);
            }
            // The engine resolved the guest's handle (a bad one traps in the vCPU) and hands each
            // child's token back once.
            bytecode::VcpuEvent::Join { child } => {
                let banked = children[child as usize]
                    .take()
                    .expect("the engine hands a child's token back once");
                vcpu.deliver_join(banked);
            }
            // #1296 — a child holding a re-granted `Jit`: `install` fills a slot of the child's OWN
            // dispatch table (its `own_dom`); `invoke` runs the unit interpreted over the child's own
            // window — this inline (interpreter) path has no emitted-unit servicer (that is the staged
            // `JIT_RUN` path's bounce), correct and slower. The unit resolves on the child's host.
            bytecode::VcpuEvent::JitInstall { handle, code } => {
                let (funcs, types) = match crate::par_resolve_unit_rt(vcpu.host_mut(), handle, code)
                {
                    Ok((f, t, _wasm, _id)) => (Ok(f), t),
                    Err(t) => (Err(t), std::sync::Arc::from(Vec::new())),
                };
                let _ = vcpu.deliver_jit_install(funcs, types);
            }
            bytecode::VcpuEvent::JitUninstall { handle, .. } => {
                let authorized = vcpu.host_mut().resolve_jit_domain(handle).map(|_| ());
                let _ = vcpu.deliver_jit_uninstall(authorized);
            }
            bytecode::VcpuEvent::JitInvoke { handle, code, .. } => {
                match crate::par_resolve_unit_rt(vcpu.host_mut(), handle, code) {
                    Ok((funcs, types, _wasm, _id)) => vcpu.deliver_jit_invoke(Ok(funcs), types),
                    Err(t) => vcpu.deliver_jit_invoke(Err(t), std::sync::Arc::from(Vec::new())),
                }
            }
            // The children this loop drives are single-threaded and non-interactive: no threads, no
            // tier-up (this is the interpreter path), no cap or stdin park. The driver's decline — a
            // value at the parent's join, never a parent-killing trap (see `crate::declined_child`).
            // Named rather than `_` (see `VcpuEvent`).
            bytecode::VcpuEvent::TierUp { .. }
            | bytecode::VcpuEvent::Instantiate { .. }
            | bytecode::VcpuEvent::Spawn { .. }
            | bytecode::VcpuEvent::Wait { .. }
            | bytecode::VcpuEvent::Notify { .. }
            | bytecode::VcpuEvent::CapPending { .. }
            | bytecode::VcpuEvent::StdinPark => return crate::declined_child(),
        }
    }
}
