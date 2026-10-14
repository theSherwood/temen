//! #2353 — the shared harness for **fuzzing the capability ABI**: no argument a guest chooses may
//! crash its host (INVARIANTS #3, #5). Coverage-guided bytes become a straight-line guest that makes
//! a random sequence of `call.cap`s — any type, op, handle, signature and arguments, the handles and
//! arguments drawn from the powerbox, earlier results, pointers into fuzzer-chosen window bytes and
//! raw values — against a powerbox holding each capability a default one grants a guest, under
//! small budgets. It runs on the tree-walker, the bytecode engine and the Cranelift JIT (#2400), so
//! it reaches both the host's dispatch and each engine's own arms (spawns, joins, page ops, the JIT's
//! window and thunks), each set up as production sets it up.
//!
//! A find is a host panic, an abort, or one host allocation past [`ALLOC_LIMIT`] ([`Bounded`]
//! refuses it, and the allocation-failure handler aborts). A refusal, a trap or an errno is the
//! guest's answer, never a find. `fuzz_one` drives it from coverage-guided bytes; the stable
//! `cap_abi` test drives the same function from seeds.

#![allow(dead_code)] // included via `#[path]` by both the fuzz target and the stable test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use temen_interp::{bytecode, cap_id, Host, MemLayout, Value};
use temen_ir::ValType;
use temen_jit::JitOutcome;

/// The most one host allocation may take: well above what the budgets below let a guest hold, so
/// a larger one is a size the guest chose that nothing charged.
pub const ALLOC_LIMIT: usize = 256 << 20;

/// A global allocator that refuses any one allocation past [`ALLOC_LIMIT`]: a fallible allocation
/// (`try_reserve`) sees the refusal, and an infallible one aborts in the allocation-failure handler,
/// which libFuzzer reports with its size.
pub struct Bounded;

// SAFETY: every call forwards to `System` unchanged, or refuses with a null pointer, which the
// `GlobalAlloc` contract allows any allocation to return.
unsafe impl GlobalAlloc for Bounded {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > ALLOC_LIMIT {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > ALLOC_LIMIT {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > ALLOC_LIMIT {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// The guest's window: 1 MiB.
const WINDOW_LOG2: u8 = 20;
const WINDOW: i64 = 1 << WINDOW_LOG2;
/// Where the fuzzer's bytes sit in the window: what a pointer argument points at.
const SCRATCH: i64 = 1 << 16;
/// Fuel for a run; a spawned child draws its own from the granted budget.
const FUEL: u64 = 1_000_000;
/// The root's ceilings: `mem`, `channel`, `spawn`, `maps`.
const GRANT: (i64, i64, i64, i64) = (64 << 20, 16 << 20, 16, 4096);
/// The `call.dyn` table a `Jit.install` may fill: 16 slots.
const TABLE_LOG2: u8 = 4;

/// The powerbox, in the order the entry function takes its handles, with each one's type.
const CAPS: [u32; 15] = [
    cap_id::STREAM,        // 0 stdout
    cap_id::STREAM,        // 1 stderr
    cap_id::STREAM,        // 2 stdin
    cap_id::INSTANTIATOR,  // 3
    cap_id::ADDRESS_SPACE, // 4
    cap_id::BUDGET,        // 5
    cap_id::MODULE,        // 6 the program itself
    cap_id::MODULE_LOADER, // 7
    cap_id::CLOCK,         // 8
    cap_id::SHARED_REGION, // 9
    cap_id::HOST_PROC,     // 10 memfs
    cap_id::HOST_PROC,     // 11 vm_fs
    cap_id::HOST_PROC,     // 12 the POSIX personality
    cap_id::HOST_PROC,     // 13 a scripted exec
    cap_id::JIT,           // 14
];

/// Types a call may name other than its handle's own: every built-in one, the self namespace, and
/// the dynamic forms.
const TYPES: [u32; 20] = [
    cap_id::STREAM,
    cap_id::EXIT,
    cap_id::CLOCK,
    3,
    cap_id::SHARED_REGION,
    cap_id::ADDRESS_SPACE,
    cap_id::INSTANTIATOR,
    cap_id::MODULE_LOADER,
    cap_id::MODULE,
    9,
    cap_id::BLOCKING,
    cap_id::JIT,
    cap_id::JIT_CODE,
    cap_id::HOST_PROC,
    cap_id::BUDGET,
    cap_id::FREEZE_AUTHORITY,
    cap_id::GUEST_IMPL_BASE,
    temen_ir::CAP_SELF_TYPE_ID,
    temen_ir::CAP_DYN_TYPE_ID,
    temen_ir::CAP_IMPORT_TYPE_ID,
];

/// Argument values worth hitting: edges of sizes, offsets, handles and integer widths.
const DICT: [i64; 32] = [
    0,
    1,
    2,
    3,
    4,
    7,
    8,
    16,
    64,
    255,
    256,
    4096,
    16384,
    65535,
    SCRATCH,
    SCRATCH + 8,
    WINDOW - 1,
    WINDOW,
    WINDOW + 1,
    1 << 30,
    (1 << 31) - 1,
    1 << 31,
    (1 << 32) - 1,
    1 << 32,
    1 << 33,
    1 << 37,
    1 << 40,
    i64::MAX,
    i64::MIN,
    -1,
    -2,
    -4096,
];

/// The fuzzer's bytes, read front to back; zeros once they run out.
struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn byte(&mut self) -> u8 {
        let (&b, rest) = self.0.split_first().unwrap_or((&0, &[]));
        self.0 = rest;
        b
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(std::array::from_fn(|_| self.byte()))
    }
}

/// A value a call returned: its name, whether it is an `i64` (else an `i32`), and the type the call
/// named.
#[derive(Clone)]
struct Val {
    name: String,
    wide: bool,
    type_id: u32,
}

/// The program under construction: its instructions, and the values its calls returned.
struct Prog {
    body: String,
    next: usize,
    results: Vec<Val>,
}

impl Prog {
    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("v{}", self.next)
    }

    /// `v` as the width `wide`, converting it if it is the other.
    fn as_width(&mut self, v: &Val, wide: bool) -> String {
        if v.wide == wide {
            return v.name.clone();
        }
        let name = self.fresh();
        let op = if wide {
            "i64.extend_i32_s"
        } else {
            "i32.wrap_i64"
        };
        let _ = writeln!(self.body, "  {name} = {op} {}", v.name);
        name
    }

    fn constant(&mut self, wide: bool, value: i64) -> String {
        let name = self.fresh();
        let (ty, value) = if wide {
            ("i64", value)
        } else {
            ("i32", value as i32 as i64)
        };
        let _ = writeln!(self.body, "  {name} = {ty}.const {value}");
        name
    }

    /// An argument of width `wide`: a dictionary value, an earlier result, a pointer into the
    /// fuzzer's bytes, or a raw value.
    fn arg(&mut self, b: &mut Bytes, wide: bool) -> String {
        let k = b.byte();
        match k {
            0..=159 => self.constant(wide, DICT[k as usize % DICT.len()]),
            160..=199 if !self.results.is_empty() => {
                let v = self.results[b.byte() as usize % self.results.len()].clone();
                self.as_width(&v, wide)
            }
            200..=231 => self.constant(wide, SCRATCH + b.byte() as i64),
            _ => self.constant(wide, b.u64() as i64),
        }
    }

    /// Store a handle the guest holds, or an earlier result, into the fuzzer's bytes: how a spawn
    /// record or a grant list comes to name a live capability.
    fn store(&mut self, b: &mut Bytes) {
        let pick = b.byte();
        let value = match pick % 4 {
            0..=2 => Val {
                name: format!("p{}", (pick >> 2) as usize % CAPS.len()),
                wide: false,
                type_id: 0,
            },
            _ if !self.results.is_empty() => {
                self.results[b.byte() as usize % self.results.len()].clone()
            }
            _ => return,
        };
        let at = self.constant(true, SCRATCH + (b.byte() as i64 & !3));
        let ty = if value.wide { "i64" } else { "i32" };
        let _ = writeln!(self.body, "  {ty}.store {at} {}", value.name);
    }

    /// One `call.cap`: its handle and type, op, signature and arguments, all from the fuzzer.
    fn call(&mut self, b: &mut Bytes) {
        let pick = b.byte();
        let (handle, own) = match pick % 8 {
            0..=5 => {
                let cap = (pick >> 3) as usize % CAPS.len();
                (format!("p{cap}"), Some(CAPS[cap]))
            }
            6 if !self.results.is_empty() => {
                let v = self.results[b.byte() as usize % self.results.len()].clone();
                (self.as_width(&v, false), None)
            }
            _ => (self.constant(false, b.u64() as i64), None),
        };
        let t = b.byte();
        let type_id = match own {
            Some(own) if t < 216 => own,
            _ => TYPES[t as usize % TYPES.len()],
        };
        let o = b.byte();
        let op = if type_id == temen_ir::CAP_SELF_TYPE_ID && o >= 0xF0 {
            // The packed self ops: `type_id`, `covers`, `export.handle` of entry `idx`.
            (6 + o as u32 % 3) | ((b.byte() as u32 % 4) << 8)
        } else if o < 192 {
            // Most capabilities' ops are in the low twenties.
            o as u32 % 24
        } else {
            o as u32 % 64
        };
        // The signature: an op's pinned one, where its type has one ([`builtin_iface_shape`]);
        // otherwise most often six `i64`s to one, which a handler reading its arguments by index
        // accepts whatever it takes; otherwise any.
        let shape = b.byte();
        let widths = b.byte();
        let pinned = temen_interp::builtin_iface_shape(type_id)
            .and_then(|ops| ops.get(op as usize).map(|(_, ft)| ft.clone()));
        let (params, rwide): (Vec<bool>, Vec<bool>) = match pinned {
            Some(ft) if shape < 192 => (
                ft.params.iter().map(|t| *t == ValType::I64).collect(),
                ft.results.iter().map(|t| *t == ValType::I64).collect(),
            ),
            _ if shape < 96 => (vec![true; 6], vec![true]),
            _ => (
                (0..shape as usize % 7)
                    .map(|i| widths & (1 << i) == 0)
                    .collect(),
                (0..(shape as usize / 7) % 3)
                    .map(|i| widths & (0x40 << i) == 0)
                    .collect(),
            ),
        };
        // `svc.poll` and `svc.wait` sleep as long as the guest asks, which the interpreters do not
        // bound by the clock (only the JIT's watchdog does): their arguments stay short, so a run
        // ends.
        let svc = type_id == temen_ir::CAP_SELF_TYPE_ID
            && matches!(
                op,
                temen_interp::CAP_SELF_SVC_POLL | temen_interp::CAP_SELF_SVC_WAIT
            );
        let args: Vec<String> = params
            .iter()
            .map(|&wide| match svc {
                true => self.constant(wide, b.byte() as i64),
                false => self.arg(b, wide),
            })
            .collect();
        let params: Vec<&str> = params
            .iter()
            .map(|&w| if w { "i64" } else { "i32" })
            .collect();
        let outs: Vec<String> = rwide.iter().map(|_| self.fresh()).collect();
        let bind = if outs.is_empty() {
            String::new()
        } else {
            format!("{} = ", outs.join(", "))
        };
        let results: Vec<&str> = rwide
            .iter()
            .map(|&w| if w { "i64" } else { "i32" })
            .collect();
        let _ = writeln!(
            self.body,
            "  {bind}call.cap {type_id} {op} ({}) -> ({}) {handle} ({})",
            params.join(", "),
            results.join(", "),
            args.join(", ")
        );
        for (name, wide) in outs.into_iter().zip(rwide) {
            self.results.push(Val {
                name,
                wide,
                type_id,
            });
        }
    }
}

/// The guest `data` becomes: its text, and the type each of its returned values' calls named.
fn program(data: &[u8]) -> (String, Vec<u32>) {
    let mut b = Bytes(data);
    let scratch: Vec<u8> = (0..b.byte() as usize * 2).map(|_| b.byte()).collect();
    let mut p = Prog {
        body: String::new(),
        next: 0,
        results: Vec::new(),
    };
    for _ in 0..1 + b.byte() % 24 {
        if b.byte() < 48 {
            p.store(&mut b);
        } else {
            p.call(&mut b);
        }
    }
    // Every result, widened, is the entry's return: what the stable test reads to see the calls
    // land.
    let rets: Vec<String> = p
        .results
        .clone()
        .iter()
        .map(|v| p.as_width(v, true))
        .collect();
    let mut src = format!("memory {WINDOW_LOG2}\n");
    if !scratch.is_empty() {
        let bytes: String = scratch.iter().map(|b| format!("\\x{b:02x}")).collect();
        let _ = writeln!(src, "data {SCRATCH} \"{bytes}\"");
    }
    let params: Vec<&str> = CAPS.iter().map(|_| "i32").collect();
    let names: Vec<String> = (0..CAPS.len()).map(|i| format!("p{i}: i32")).collect();
    let results: Vec<&str> = rets.iter().map(|_| "i64").collect();
    let _ = write!(
        src,
        "func ({}) -> ({}) {{\nblock 0 ({}) {{\n{}  return {}\n  }}\n}}\n",
        params.join(", "),
        results.join(", "),
        names.join(", "),
        p.body,
        rets.join(" ")
    );
    // A child entry, for a spawn to start.
    src.push_str("func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  return v0\n  }\n}\n");
    (src, p.results.iter().map(|v| v.type_id).collect())
}

/// The guest `data` becomes, as text: what to read when an input is a find.
pub fn guest(data: &[u8]) -> String {
    program(data).0
}

/// A fresh host holding the powerbox [`CAPS`] lists, and the entry's arguments: its handles.
fn powerbox(m: &temen_ir::Module) -> (Host, Vec<i32>) {
    let mut h = Host::new();
    let (mem, channel, spawn, maps) = GRANT;
    h.set_grant(mem, channel, spawn, maps);
    // OS-backed regions, which the JIT maps for real, as production's are.
    h.set_region_factory(temen_run::new_shared_region);
    let handles = [
        h.grant_stream(temen_interp::StreamRole::Out),
        h.grant_stream(temen_interp::StreamRole::Err),
        h.grant_stream(temen_interp::StreamRole::In),
        h.grant_instantiator(0, WINDOW as u64),
        h.grant_address_space(0, WINDOW as u64),
        h.grant_budget(FUEL as i64, 4 << 20, 4),
        h.grant_module(m),
        temen_run::grant_module_loader(&mut h),
        h.grant_clock(),
        h.grant_shared_region(1 << 16),
        temen_run::fs::mem_fs().install(&mut h, 0),
        temen_run::fs::grant_vm_fs(&mut h, None),
        temen_run::posix::posix_cap(WINDOW as u64 / 2, WINDOW as u64, Vec::new())
            .0
            .install(&mut h, 0),
        temen_run::exec::scripted_exec(vec![temen_run::exec::ScriptedEntry {
            argv_prefix: vec!["echo".into()],
            stdout: b"hi\n".to_vec(),
            stderr: Vec::new(),
            exit: 0,
        }])
        .install(&mut h, 0),
        temen_run::grant_jit(&mut h, m, TABLE_LOG2),
    ];
    (h, handles.to_vec())
}

/// What one input did, for the stable test to check the harness reaches the capabilities: how the
/// runs ended, and what the calls of the completed ones returned.
#[derive(Debug, Default)]
pub struct Runs {
    pub completed: usize,
    pub trapped: usize,
    pub declined: usize,
    /// Each result a completed run's calls returned, with the type the call named.
    pub results: Vec<(u32, i64)>,
}

/// How a run ended: what it returned, each result as an `i64`, or the wire code it ended with
/// (`temen_ir::trap_code`: a trap, an exit, a host unwind); `None` when the engine declined the
/// module.
pub type Outcome = Option<Result<Vec<i64>, i64>>;

/// `m`'s entry on the tree-walker, the bytecode engine and the Cranelift JIT, each over a fresh
/// powerbox, which it takes as its arguments.
pub fn on_every_engine(m: &temen_ir::Module) -> [Outcome; 3] {
    let interp = |r: Result<Vec<Value>, temen_interp::Trap>| {
        let wide = |v: &Value| match v {
            Value::I64(x) => *x,
            _ => i64::MIN,
        };
        r.map(|vals| vals.iter().map(wide).collect())
            .map_err(temen_interp::Trap::code)
    };
    let (mut h, handles) = powerbox(m);
    let args: Vec<Value> = handles.into_iter().map(Value::I32).collect();
    let mut fuel = FUEL;
    let tree_walker = Some(interp(temen_interp::run_with_host(
        m, 0, &args, &mut fuel, &mut h,
    )));
    let (mut h, handles) = powerbox(m);
    let args: Vec<Value> = handles.into_iter().map(Value::I32).collect();
    let mut fuel = FUEL;
    let bytecode = bytecode::compile_and_run_with_host(m, 0, &args, &mut fuel, &mut h).map(interp);
    let (mut h, handles) = powerbox(m);
    let args: Vec<i64> = handles.into_iter().map(i64::from).collect();
    let window = MemLayout::image(vec![0; WINDOW as usize]);
    let jit = match temen_run::jit_cap_run(m, 0, &args, &window, 0, TABLE_LOG2, &mut h, None) {
        Ok((JitOutcome::Returned(vals), _)) => Some(Ok(vals)),
        Ok((JitOutcome::Trapped(kind), _)) => Some(Err(kind as i64)),
        Ok((JitOutcome::Exited(code), _)) => Some(Err(temen_ir::trap_code::exit(code))),
        Ok((JitOutcome::HostUnwound, _)) => Some(Err(temen_jit::HOST_UNWIND_CODE.into())),
        Err(_) => None,
    };
    [tree_walker, bytecode, jit]
}

/// Run the guest `data` becomes on every engine ([`on_every_engine`]). A host panic or abort is the
/// find; anything the guest sees is not.
pub fn fuzz_one(data: &[u8]) -> Runs {
    let (src, types) = program(data);
    let m = temen_text::parse_module(&src).expect("the generated guest parses");
    temen_verify::verify_module(&m).expect("the generated guest verifies");
    let mut runs = Runs::default();
    for outcome in on_every_engine(&m) {
        match outcome {
            Some(Ok(vals)) => {
                runs.completed += 1;
                runs.results.extend(types.iter().copied().zip(vals));
            }
            Some(Err(_)) => runs.trapped += 1,
            None => runs.declined += 1,
        }
    }
    runs
}
