//! §22 "Unit-own funcrefs" (DESIGN.md), by route — on the tree-walk oracle, every bytecode driver and
//! Cranelift alike (#2314).
//!
//! A unit that names one of its own functions — `ref.func N`, or the entry of a `thread.spawn N` — has
//! its functions installed into the domain's `call.dyn` table once, in the first free padding slots,
//! and its `ref.func N` yields the slot its function N fills. Each `install` of the unit then fills
//! one more slot with its entry; an `invoke` reuses the slots too. Cranelift does it when it compiles
//! the unit (`define_extra`), the interpreters when the unit is first installed or invoked, so the
//! slots a guest sees agree whenever nothing is installed between the compile and the first use.
//!
//! Before #2314 only Cranelift did this for an **installed** unit: the tree-walker remapped only an
//! invoked unit's `ref.func`, and the bytecode engine never did. So an installed unit's funcref named
//! module 0's function of that index, and a closure the unit made over its own function called the
//! program's — the JACL shell's REPL lines trapped `IndirectCallType` on the tree-walker and bytecode.
//!
//! Every guest carries a decoy as its func 1: the type of the units' func 1, returning `x + 1000`
//! where theirs returns `x + 42`. A funcref that names the wrong module therefore answers `1005`
//! instead of trapping. The guests compile their units in-run from blobs in data segments, as the
//! shell compiles each line, so Cranelift reserves the unit's slots at the compile op.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::DEFAULT_RESERVED_LOG2;
use temen_jit::{JitOutcome, TrapKind};
use temen_run::{grant_jit, grant_jit_threads, jit_cap_run};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Where the guest's data segments hold its unit blobs: one per `BLOB_STRIDE` bytes from here.
const BLOB_AT: u64 = 20480;
const BLOB_STRIDE: u64 = 2048;

/// The decoy every guest carries as func 1 — what a funcref naming module 0 calls.
const DECOY: &str = r#"
func (i64) -> (i64) {
block 0 (vx: i64) {
  vk = i64.const 1000
  vr = i64.add vx vk
  return vr
  }
}
"#;

/// The units' func 1: the decoy's type, returning `x + 42`.
const OWN_F1: &str = r#"
func (i64) -> (i64) {
block 0 (vx: i64) {
  vk = i64.const 42
  vr = i64.add vx vk
  return vr
  }
}
"#;

/// A unit whose entry `call.dyn`s its own func 1 through `ref.func 1`: `x -> x + 42`.
fn calls_own() -> String {
    format!(
        r#"memory 16
func (i64) -> (i64) {{
block 0 (vx: i64) {{
  vf = ref.func 1
  vr = call.dyn (i64) -> (i64) vf (vx)
  return vr
  }}
}}
{OWN_F1}"#
    )
}

/// A unit whose entry makes a closure over its own func 1 — `ref.func 1`, stored at 17408 the way a
/// REPL line defining a proc stores it — and returns it.
fn makes_closure() -> String {
    format!(
        r#"memory 16
func () -> (i32) {{
block 0 () {{
  vf = ref.func 1
  vp = i64.const 17408
  i32.store vp vf
  return vf
  }}
}}
{OWN_F1}"#
    )
}

/// A unit that names no function of its own: it calls the closure stored at 17408 with `x`, the way a
/// later REPL line calls a proc an earlier line defined.
const CALLS_CLOSURE: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vx: i64) {
  vp = i64.const 17408
  vf = i32.load vp
  vr = call.dyn (i64) -> (i64) vf (vx)
  return vr
  }
}
"#;

/// A unit that names its own func 1 only as a `thread.spawn` entry: it spawns it, joins it and
/// returns its value, 7.
const SPAWNS_OWN: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  varg = i64.const 0
  vt = thread.spawn 1 vsp varg
  vr_st, vr = thread.join vt
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v7 = i64.const 7
  return v7
  }
}
"#;

/// The text IR that compiles unit `i` of the guest (its blob at `BLOB_AT + i * BLOB_STRIDE`) into
/// register `{dst}`.
fn compile(i: usize, dst: &str, units: &[Vec<u8>]) -> String {
    let at = BLOB_AT + i as u64 * BLOB_STRIDE;
    let len = units[i].len();
    format!(
        "  {dst}p = i64.const {at}\n  {dst}l = i64.const {len}\n  \
         {dst} = call.cap 11 0 (i64, i64) -> (i64) vjit ({dst}p, {dst}l)\n"
    )
}

/// The guest: func 0, whose body `entry` writes (it takes the `Jit` handle as `vjit`, and gets the
/// text that compiles unit `i` into a register from the [`compile`] it is handed); the decoy as
/// func 1; and each unit's blob in a data segment.
fn guest(
    entry: impl Fn(&dyn Fn(usize, &str) -> String) -> String,
    units: &[String],
) -> temen_ir::Module {
    let blobs: Vec<Vec<u8>> = units
        .iter()
        .map(|src| {
            let u = parse_module(src).expect("parse unit");
            verify_module(&u).expect("verify unit");
            temen_encode::encode_module(&u)
        })
        .collect();
    let body = entry(&|i, dst| compile(i, dst, &blobs));
    let data: String = blobs
        .iter()
        .enumerate()
        .map(|(i, b)| {
            assert!((b.len() as u64) < BLOB_STRIDE, "blob {i} fits its stride");
            let bytes: String = b.iter().map(|x| format!("\\x{x:02x}")).collect();
            format!("data {} \"{bytes}\"\n", BLOB_AT + i as u64 * BLOB_STRIDE)
        })
        .collect();
    let src = format!(
        "memory 16\nfunc (i32) -> (i64) {{\nblock 0 (vjit: i32) {{\n{body}  }}\n}}\n{DECOY}{data}"
    );
    let m = parse_module(&src).expect("parse guest");
    verify_module(&m).expect("verify guest");
    m
}

/// What a run came to, in one vocabulary across the engines.
#[derive(Debug, PartialEq)]
enum Outcome {
    Returned(i64),
    Trapped(String),
    Other(String),
}

fn interp(r: Result<Vec<Value>, Trap>) -> Outcome {
    match r.as_deref() {
        Ok([Value::I64(x)]) => Outcome::Returned(*x),
        Err(t) => Outcome::Trapped(format!("{t:?}")),
        other => Outcome::Other(format!("{other:?}")),
    }
}

/// How the guest's powerbox is set up: the `Jit` grant (threads or not) and its table's `log2`.
#[derive(Clone, Copy)]
struct Grant {
    threads: bool,
    table_log2: u8,
}

impl Grant {
    const PLAIN: Grant = Grant {
        threads: false,
        table_log2: 4,
    };

    fn host(self, m: &temen_ir::Module) -> (Host, i32) {
        let mut host = Host::new();
        let jit = if self.threads {
            grant_jit_threads(&mut host, m, self.table_log2)
        } else {
            grant_jit(&mut host, m, self.table_log2)
        };
        (host, jit)
    }
}

fn cranelift(m: &temen_ir::Module, grant: Grant) -> Outcome {
    let (mut host, jit) = grant.host(m);
    let (out, _) = jit_cap_run(
        m,
        0,
        &[jit as i64],
        &MemLayout::image(Vec::new()),
        DEFAULT_RESERVED_LOG2,
        grant.table_log2,
        &mut host,
        None,
    )
    .expect("jit run");
    match out {
        JitOutcome::Returned(v) if v.len() == 1 => Outcome::Returned(v[0]),
        JitOutcome::Trapped(TrapKind::CapFault) => Outcome::Trapped("CapFault".into()),
        other => Outcome::Other(format!("{other:?}")),
    }
}

/// Assert that `m` comes to `want` on every engine: the tree-walk oracle, the four bytecode drivers
/// ([`drivers::ALL`]) and Cranelift. A failure names every engine that disagreed.
fn on_every_engine(m: &temen_ir::Module, grant: Grant, want: Outcome) {
    let powerbox = || {
        let (host, jit) = grant.host(m);
        (host, vec![Value::I32(jit)])
    };
    let mut got: Vec<(String, Outcome)> = drivers::ALL
        .iter()
        .map(|&d| {
            let o = drivers::run_on(d, m, &powerbox)
                .map_or(Outcome::Other("declined".into()), |r| interp(r.result));
            (format!("{d:?}"), o)
        })
        .collect();
    got.push(("Cranelift".into(), cranelift(m, grant)));
    let wrong: Vec<_> = got.iter().filter(|(_, o)| *o != want).collect();
    assert!(
        wrong.is_empty(),
        "want {want:?} on every engine, but {wrong:?}"
    );
}

/// Pack the slots and funcrefs a guest saw, 4 bits each above its 8-bit result: `r | a << 8 | b << 12
/// | …`. Every slot here is below the 16-slot table, and every result below 256.
fn packed(r: i64, slots: &[i64]) -> i64 {
    slots
        .iter()
        .enumerate()
        .fold(r, |acc, (i, s)| acc | (s << (8 + 4 * i)))
}

/// The text IR folding the `i32` / `i64` registers `regs` into `vacc` as [`packed`] does, starting
/// from the `i64` result `r`.
fn pack(r: &str, regs: &[(&str, bool)]) -> String {
    let mut s = format!("  vacc0 = i64.or {r} {r}\n");
    for (i, (reg, is_i32)) in regs.iter().enumerate() {
        let wide = if *is_i32 {
            s += &format!("  vw{i} = i64.extend_i32_u {reg}\n");
            format!("vw{i}")
        } else {
            reg.to_string()
        };
        s += &format!(
            "  vsh{i} = i64.const {}\n  vt{i} = i64.shl {wide} vsh{i}\n  vacc{} = i64.or vacc{i} vt{i}\n",
            8 + 4 * i,
            i + 1
        );
    }
    s + &format!("  return vacc{}\n", regs.len())
}

/// The bug as the shell met it: an **installed** unit `call.dyn`s its own func 1 through `ref.func 1`
/// and returns its value. Module 0's func 1 has the same type, so a funcref naming module 0 runs the
/// decoy and answers 1005 instead of trapping. Before #2314 the tree-walker and every bytecode driver
/// answered 1005.
#[test]
fn an_installed_unit_calls_its_own_function() {
    let m = guest(
        |compile| {
            format!(
                "{}  vs = call.cap 11 3 (i64) -> (i64) vjit (vc)\n  vs32 = i32.wrap_i64 vs\n  \
                 vx = i64.const 5\n  vr = call.dyn (i64) -> (i64) vs32 (vx)\n  return vr\n",
                compile(0, "vc")
            )
        },
        &[calls_own()],
    );
    on_every_engine(&m, Grant::PLAIN, Outcome::Returned(47));
}

/// The funcref escapes: an installed unit makes a closure over its own func 1 and returns it, module 0
/// calls it — and so does a second installed unit that names no function of its own, which finds the
/// closure where the first stored it. That is the shell's REPL: line 1 defines a proc, line 2 calls it.
/// The slots each install returned and the closure agree on every engine too: the closure is the
/// unit's func 1, auto-installed after its func 0 into the first padding (2 and 3, module 0 holding
/// two functions), the unit's entry fills 4, and the second unit, compiled after it, fills 5.
#[test]
fn an_installed_units_closure_calls_its_own_function() {
    let m = guest(
        |compile| {
            format!(
                "{}  vs1 = call.cap 11 3 (i64) -> (i64) vjit (vc1)\n  vs1w = i32.wrap_i64 vs1\n  \
                 vf = call.dyn () -> (i32) vs1w ()\n  vx = i64.const 5\n  \
                 vr0 = call.dyn (i64) -> (i64) vf (vx)\n\
                 {}  vs2 = call.cap 11 3 (i64) -> (i64) vjit (vc2)\n  vs2w = i32.wrap_i64 vs2\n  \
                 vr1 = call.dyn (i64) -> (i64) vs2w (vr0)\n{}",
                compile(0, "vc1"),
                compile(1, "vc2"),
                pack("vr1", &[("vs1", false), ("vs2", false), ("vf", true)]),
            )
        },
        &[makes_closure(), CALLS_CLOSURE.to_string()],
    );
    // 5 + 42 by module 0's call, + 42 again by the second unit's.
    on_every_engine(&m, Grant::PLAIN, Outcome::Returned(packed(89, &[4, 5, 3])));
}

/// One unit, invoked once and installed twice: its functions are installed once, at its first use,
/// and every route reuses them. The invoke's closure, each install's slot and each installed entry's
/// closure agree on every engine — the invoke takes 2 and 3 for the unit's functions, the installs
/// fill 4 and 5 with its entry, and every closure is 3 — and the last closure still calls the unit's
/// own func 1.
#[test]
fn a_units_functions_are_installed_once_across_invoke_and_installs() {
    let m = guest(
        |compile| {
            format!(
                "{}  vfi = call.cap 11 1 (i64) -> (i32) vjit (vc)\n  \
                 vs1 = call.cap 11 3 (i64) -> (i64) vjit (vc)\n  \
                 vs2 = call.cap 11 3 (i64) -> (i64) vjit (vc)\n  \
                 vs1w = i32.wrap_i64 vs1\n  vf1 = call.dyn () -> (i32) vs1w ()\n  \
                 vs2w = i32.wrap_i64 vs2\n  vf2 = call.dyn () -> (i32) vs2w ()\n  \
                 vx = i64.const 5\n  vr = call.dyn (i64) -> (i64) vf2 (vx)\n{}",
                compile(0, "vc"),
                pack(
                    "vr",
                    &[
                        ("vfi", true),
                        ("vs1", false),
                        ("vs2", false),
                        ("vf1", true),
                        ("vf2", true)
                    ]
                ),
            )
        },
        &[makes_closure()],
    );
    on_every_engine(
        &m,
        Grant::PLAIN,
        Outcome::Returned(packed(47, &[3, 4, 5, 3, 3])),
    );
}

/// An **invoked** unit `call.dyn`s its own func 1 through `ref.func 1`. The tree-walker already
/// remapped it; before #2314 every bytecode driver ran the decoy and answered 1005.
#[test]
fn an_invoked_unit_calls_its_own_function() {
    let m = guest(
        |compile| {
            format!(
                "{}  vx = i64.const 5\n  vr = call.cap 11 1 (i64, i64) -> (i64) vjit (vc, vx)\n  return vr\n",
                compile(0, "vc")
            )
        },
        &[calls_own()],
    );
    on_every_engine(&m, Grant::PLAIN, Outcome::Returned(47));
}

/// A unit that names its own function only as a `thread.spawn` entry takes its slots too, as
/// Cranelift's spawn dispatches through them: its install returns 4 on every engine, after the unit's
/// two functions in 2 and 3. Before #2314 the interpreters installed no slots for it, and returned 2.
#[test]
fn a_unit_spawning_its_own_function_takes_its_slots() {
    let m = guest(
        |compile| {
            format!(
                "{}  vs = call.cap 11 3 (i64) -> (i64) vjit (vc)\n  vsw = i32.wrap_i64 vs\n  \
                 vr = call.dyn () -> (i64) vsw ()\n{}",
                compile(0, "vc"),
                pack("vr", &[("vs", false)]),
            )
        },
        &[SPAWNS_OWN.to_string()],
    );
    let grant = Grant {
        threads: true,
        ..Grant::PLAIN
    };
    on_every_engine(&m, grant, Outcome::Returned(packed(7, &[4])));
}

/// The table fills: with two padding slots, a two-function unit's functions take both, so its install
/// finds none for its entry and is `-ENOSPC` on every engine (Cranelift took the two at the compile).
#[test]
fn an_install_with_no_slot_left_for_the_entry_is_enospc() {
    let m = guest(
        |compile| {
            format!(
                "{}  vs = call.cap 11 3 (i64) -> (i64) vjit (vc)\n  return vs\n",
                compile(0, "vc")
            )
        },
        &[calls_own()],
    );
    let grant = Grant {
        table_log2: 2,
        ..Grant::PLAIN
    };
    on_every_engine(&m, grant, Outcome::Returned(-28));
}

/// A unit with more functions than the table has padding cannot name its own: Cranelift refuses its
/// compile (`-EINVAL`, so the invoke's handle is forged), and the interpreters refuse its first use.
/// The invoke is a `CapFault` on every engine, never a run of the decoy.
#[test]
fn an_invoke_with_too_few_slots_for_the_units_functions_capfaults() {
    let unit = format!("{}{OWN_F1}", calls_own());
    let m = guest(
        |compile| {
            format!(
                "{}  vx = i64.const 5\n  vr = call.cap 11 1 (i64, i64) -> (i64) vjit (vc, vx)\n  return vr\n",
                compile(0, "vc")
            )
        },
        &[unit],
    );
    let grant = Grant {
        table_log2: 2,
        ..Grant::PLAIN
    };
    on_every_engine(&m, grant, Outcome::Trapped("CapFault".into()));
}
