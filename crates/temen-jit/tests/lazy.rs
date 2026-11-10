//! #2594 — a module the JIT compiles function by function, at each one's first call and ahead of
//! it: one of [`FUNCS`] functions, which is enough for a compile to go lazily on its own (unless
//! `TEMEN_JIT_LAZY=off`). Its runs must answer as the interpreter's do while the guest waits on
//! compiles: through tail calls, `call.dyn`, results returned through memory, from threads that race
//! to the same first calls and from a fiber's stack, and again once the linked copy has replaced the
//! code compiled lazily.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use temen_interp::Value;
use temen_ir::DEFAULT_RESERVED_LOG2;
use temen_jit::{CompiledModule, InstanceAddrs, JitOutcome, INERT_CAP_THUNK};

/// The functions before the chain: the root (0), a thread's body (1), a fiber's body (2), one with
/// six results (3), one of floats (4), one that tail-calls through `call.dyn` (5) into another (6),
/// and the chain's entry from an `i32` (7).
const FIRST: usize = 8;
/// The chain: each of its functions calls the next, by one of four shapes in turn.
const CHAIN: usize = 88;
const FUNCS: usize = FIRST + CHAIN;

/// The module, with `root` as function 0.
fn program(root: &str) -> temen_ir::Module {
    let mut s = String::from(root);
    s += &format!(
        "func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  v2 = call {FIRST} (v1)
  return v2
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  v2 = call {FIRST} (v1)
  v3 = suspend v2
  v4 = call {FIRST} (v3)
  return v4
  }}
}}
func (i64) -> (i64, i64, i64, i64, i64, i64) {{
block 0 (v0: i64) {{
  v1 = i64.const 1
  v2 = i64.add v0 v1
  v3 = i64.add v2 v1
  v4 = i64.add v3 v1
  v5 = i64.add v4 v1
  v6 = i64.add v5 v1
  return v0 v2 v3 v4 v5 v6
  }}
}}
func (f64, i32) -> (f64) {{
block 0 (v0: f64, v1: i32) {{
  v2 = f64.convert_i32_s v1
  v3 = f64.mul v2 v0
  v4 = f64.const 3.25
  v5 = f64.add v3 v4
  return v5
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v1 = ref.func 6
  return_call.dyn (i64) -> (i64) v1 (v0)
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v1 = i64.const 7
  v2 = i64.add v0 v1
  return v2
  }}
}}
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  v1 = i64.extend_i32_s v0
  v2 = call {FIRST} (v1)
  return v2
  }}
}}
"
    );
    for k in 0..CHAIN {
        let next = FIRST + k + 1;
        let body = if k == CHAIN - 1 {
            "  return v0\n".to_string()
        } else {
            match k % 4 {
                0 => format!(
                    "  v1 = i64.const 31\n  v2 = i64.mul v0 v1\n  v3 = i64.const {k}\n  \
                     v4 = i64.add v2 v3\n  v5 = i64.const 1048575\n  v6 = i64.and v4 v5\n  \
                     return_call {next} (v6)\n"
                ),
                1 => format!(
                    "  v1 = i64.const {k}\n  v2 = i64.xor v0 v1\n  v3 = call {next} (v2)\n  \
                     v4 = i64.const 1\n  v5 = i64.add v3 v4\n  return v5\n"
                ),
                2 => format!(
                    "  v1, v2, v3, v4, v5, v6 = call 3 (v0)\n  v7 = i64.add v1 v2\n  \
                     v8 = i64.add v7 v3\n  v9 = i64.add v8 v4\n  v10 = i64.add v9 v5\n  \
                     v11 = i64.add v10 v6\n  v12 = i64.const 1048575\n  v13 = i64.and v11 v12\n  \
                     v14 = call {next} (v13)\n  return v14\n"
                ),
                _ => format!(
                    "  v1 = i32.wrap_i64 v0\n  v2 = f64.const 0.5\n  v3 = call 4 (v2, v1)\n  \
                     v4 = i64.trunc_sat_f64_s v3\n  v5 = i64.const 1048575\n  v6 = i64.and v4 v5\n  \
                     v7 = ref.func 5\n  v8 = call.dyn (i64) -> (i64) v7 (v6)\n  \
                     v9 = call {next} (v8)\n  return v9\n"
                ),
            }
        };
        s += &format!("func (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n{body}  }}\n}}\n");
    }
    let m = temen_text::parse_module(&s).unwrap_or_else(|e| panic!("parse: {e:?}\n{s}"));
    assert_eq!(m.funcs.len(), FUNCS);
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    m
}

/// The root of the plain program: the chain from two of its entries.
const PLAIN: &str = "func () -> (i64) {
block 0 () {
  v0 = i32.const 5
  v1 = call 7 (v0)
  v2 = i64.const 9
  v3 = call 8 (v2)
  v4 = i64.add v1 v3
  return v4
  }
}
";

/// A root that starts [`THREADS`] threads of function 1 at once, each on its own argument, and
/// sums what they return: they race to the same first calls.
fn threads() -> String {
    let mut s = String::from("func () -> (i64) {\nblock 0 () {\n  vsp = i64.const 4096\n");
    for t in 0..THREADS {
        s += &format!("  va{t} = i64.const {t}\n  vt{t} = thread.spawn 1 vsp va{t}\n");
    }
    s += "  vsum0 = i64.const 0\n";
    for t in 0..THREADS {
        s += &format!(
            "  vst{t}, vr{t} = thread.join vt{t}\n  vsum{} = i64.add vsum{t} vr{t}\n",
            t + 1
        );
    }
    s += &format!("  return vsum{THREADS}\n  }}\n}}\n");
    s
}
const THREADS: usize = 16;

/// A root that runs function 2 on a fiber, which suspends between two runs of the chain.
const FIBER: &str = "func () -> (i64) {
block 0 () {
  v0 = ref.func 2
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 10
  v4, v5 = cont.resume v2 v3
  v6, v7 = cont.resume v2 v5
  v8 = i64.add v5 v7
  return v8
  }
}
";

/// What the interpreter answers for `m`.
fn interpreted(m: &temen_ir::Module) -> i64 {
    let mut fuel = 100_000_000u64;
    match temen_interp::run(m, 0, &[], &mut fuel).expect("the interpreter runs it")[..] {
        [Value::I64(v)] => v,
        ref other => panic!("{other:?}"),
    }
}

fn compile(m: &temen_ir::Module) -> CompiledModule {
    let cm = CompiledModule::compile(
        m,
        0,
        INERT_CAP_THUNK,
        core::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        None,
        0,
    )
    .expect("jit compiles");
    assert!(
        cm.compiles_lazily() || std::env::var("TEMEN_JIT_LAZY").as_deref() == Ok("off"),
        "a module of {FUNCS} functions compiles lazily"
    );
    cm
}

fn run(cm: &mut CompiledModule) -> i64 {
    match cm.run(&[], None, None).expect("run").0 {
        JitOutcome::Returned(r) => r[0],
        other => panic!("{other:?}"),
    }
}

/// Compile `m`, run it, and run it again once its linked copy has replaced what was compiled
/// lazily: both answer as the interpreter does.
fn runs_as_interpreted(m: &temen_ir::Module) {
    let want = interpreted(m);
    let mut cm = compile(m);
    assert_eq!(run(&mut cm), want, "while compiling at first calls");
    assert!(
        cm.wait_linked() || !cm.compiles_lazily(),
        "the linked copy is published"
    );
    assert_eq!(run(&mut cm), want, "on the linked copy");
}

#[test]
fn a_module_compiled_at_first_calls_runs_as_interpreted() {
    runs_as_interpreted(&program(PLAIN));
}

#[cfg(unix)]
#[test]
fn threads_that_race_to_first_calls_run_as_interpreted() {
    runs_as_interpreted(&program(&threads()));
}

#[cfg(any(unix, all(windows, target_arch = "x86_64")))]
#[test]
fn a_fiber_waits_on_a_compile_on_its_own_stack() {
    runs_as_interpreted(&program(FIBER));
}

/// Instances of one compile, run at once: their first calls wait on the same compiles.
#[test]
fn instances_of_one_compile_wait_on_the_same_compiles() {
    let m = program(PLAIN);
    let want = interpreted(&m);
    let first = compile(&m);
    let code = first.share().expect("only code");
    std::thread::scope(|s| {
        for _ in 0..4 {
            let code = &code;
            s.spawn(move || {
                let mut cm = code.instance(InstanceAddrs::NONE, None);
                assert_eq!(run(&mut cm), want);
            });
        }
    });
}
