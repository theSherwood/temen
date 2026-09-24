//! **A run as data** (#1720, decision C of #1717): the nodes a generated root guest spawns, each as a
//! detached §14 child in its own window, and the capabilities each is granted **by name** — plus the
//! one generator that turns that plan into the root's Temen IR.
//!
//! The root is ordinary guest code. Every spawn and grant it makes goes through op 15 and a grant
//! record, so it is verified and confined like any other guest, and the generator that writes it is
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
/// Grant records (16 bytes each: `{name_off: u32, name_len: u32, handle: u32, flags: u32}`).
const REC_BASE: u64 = GUARD + 1024;
/// One 16-byte name slot per root capability, shared by every record that grants it.
const NAME_BASE: u64 = GUARD + 2048;
/// The nodes' argv payloads, back to back.
const ARGV_BASE: u64 = GUARD + 4096;
const SLOT: u64 = 16;

/// A run: the root's own capabilities, and the nodes it spawns and joins.
pub struct Plan {
    /// The capabilities the root holds, by name, in the order it takes them (after its budget).
    /// A node reaches one only if its [`Node::grants`] names it.
    pub caps: Vec<String>,
    /// Spawned in order, then joined in order; the root returns the last node's result.
    pub nodes: Vec<Node>,
}

/// One §14 child: a module the host grants the root (func 0 is its entry), run detached (op 15) in a
/// fresh window minted from the root's budget.
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
        }
    }

    /// What the root's budget must hold to mint every node's window.
    pub fn window_bytes(&self) -> u64 {
        self.nodes.iter().map(|n| 1u64 << n.window_log2).sum()
    }

    /// The root guest's Temen IR text. Its params are `(inst, module_0 … module_{n-1}, budget, cap_0 …)`
    /// — the order [`root_args`](Self::root_args) builds. `Err` names what does not fit or resolve.
    pub fn root_src(&self) -> Result<String, String> {
        let n = self.nodes.len();
        if n == 0 {
            return Err("a plan spawns at least one node".into());
        }
        if NAME_BASE + self.caps.len() as u64 * SLOT > ARGV_BASE {
            return Err(format!("{} root capabilities do not fit", self.caps.len()));
        }
        // Capability names, one slot each. Plain ASCII so the name is its own data literal.
        let mut data = String::new();
        for (i, name) in self.caps.iter().enumerate() {
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
        // Grant records, each node's contiguous: word0 = {name_off | name_len << 32}, then the handle
        // (the root's param for that capability).
        let cap_param = |i: usize| 2 + n + i;
        let mut records = String::new();
        let mut first_rec = Vec::with_capacity(n);
        let mut r = 0u64;
        for node in &self.nodes {
            first_rec.push(REC_BASE + r * SLOT);
            for name in &node.grants {
                let Some(i) = self.caps.iter().position(|c| c == name) else {
                    return Err(format!(
                        "a node is granted {name:?}, which the root does not hold"
                    ));
                };
                let roff = REC_BASE + r * SLOT;
                let w0 = (NAME_BASE + i as u64 * SLOT) | ((name.len() as u64) << 32);
                records.push_str(&format!(
                    "  xr{roff} = i64.const {w0}\n  or{roff} = i64.const {roff}\n  i64.store or{roff} xr{roff}\n  \
                     h{roff} = i64.extend_i32_u v{vi}\n  oh{roff} = i64.const {hoff}\n  i64.store oh{roff} h{roff}\n",
                    vi = cap_param(i),
                    hoff = roff + 8,
                ));
                r += 1;
            }
        }
        if REC_BASE + r * SLOT > NAME_BASE {
            return Err(format!("{r} grant records do not fit"));
        }
        // Spawn every node (op 15: its own window, minted from the budget), then join every node.
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
            body.push_str(&format!("  vmh{s} = i64.extend_i32_u v{}\n", 1 + k));
            if k == 0 {
                body.push_str(&format!("  vmin = i64.extend_i32_u v{}\n", 1 + n));
            }
            let (ap, al) = argv_at[k];
            body.push_str(&format!(
                "  vgptr{s} = i64.const {gptr}\n  vgn{s} = i64.const {gn}\n  ventry{s} = i64.const 0\n  \
                 vlog{s} = i64.const {log}\n  vq{s} = i64.const 0\n  vap{s} = i64.const {ap}\n  \
                 val{s} = i64.const {al}\n  vh{s} = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) \
                 -> (i32) v0 (vmin, vmh{s}, vgptr{s}, vgn{s}, ventry{s}, vlog{s}, vq{s}, vap{s}, val{s})\n",
                gptr = first_rec[k],
                gn = node.grants.len(),
                log = node.window_log2,
            ));
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
    pub fn root_args(&self, host: &mut Host, modules: &[&Module], caps: &[i32]) -> Vec<Value> {
        debug_assert_eq!(modules.len(), self.nodes.len());
        debug_assert_eq!(caps.len(), self.caps.len());
        let inst = host.grant_instantiator(0, 1u64 << ROOT_WINDOW_LOG2);
        let mods: Vec<i32> = modules.iter().map(|m| host.grant_module(m)).collect();
        let budget = host.grant_budget(0, self.window_bytes() as i64, 0);
        std::iter::once(inst)
            .chain(mods)
            .chain(std::iter::once(budget))
            .chain(caps.iter().copied())
            .map(Value::I32)
            .collect()
    }
}

/// Run `plan` to completion on the resumable interpreter (the native / single-threaded driver,
/// [`crate::nimc::drive_op13`]): generate the root, grant it [`Plan::root_args`] over `host` (which
/// already holds `caps`), and return what the root returns — the last node's result.
pub fn run(
    plan: &Plan,
    modules: &[&Module],
    mut host: Host,
    caps: &[i32],
) -> Result<Vec<Value>, Trap> {
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
    .and_then(|root| crate::nimc::drive_op13(&prog, base, root, None));
    drop(back);
    // SAFETY: same layout; the root vCPU and its region views are dropped above.
    unsafe { std::alloc::dealloc(base, layout) };
    out
}
