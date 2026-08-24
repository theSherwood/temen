//! **Lua's own test suite** on the on-ramp — three unmodified files from the official Lua 5.4.7
//! distribution (`testes/vararg.lua`, `testes/bwcoercion.lua`, `testes/pm.lua`) run through the whole
//! VM with the base/`string`/`table`/`math`/`utf8` libraries open, each as its own chunk under
//! `pcall`. A Lua test signals failure by raising (an `assert`), which `pcall` catches, so a clean
//! **exit 0** means every `assert` in all three files held — identical to running them on native Lua
//! (the suite's own pass/fail contract). Byte-for-byte the same outcome on the tree-walker, bytecode,
//! and JIT.
//!
//! The fixture (`tests/fixtures/lua/lua_testsuite.ll`) links the Lua core + those five libraries with
//! the guest libc shim, guest `libm`, guest `strtod` (incl. hex floats), the guest runtime `snprintf`,
//! and fdlibm inverse-trig/`modf` (`lua_testsuite_trig.c`) — see the fixtures README. The three files
//! were chosen because they are self-contained (no `require`/`os`/`io`/`debug`/`coroutine` and no
//! internal `T` test library): `vararg` exercises `...`/`select`/`table.unpack`; `bwcoercion` the
//! string↔number bitwise coercions with `_ENV = nil`; `pm` the full pattern-matching engine
//! (`find`/`match`/`gmatch`/`gsub`, captures, anchors, `%b`, `%f`).

use temen_run::{Backend, Limits, Outcome, RunConfig, Value};

fn run(backend: Backend) -> temen_run::Run {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_testsuite.ll"
    );
    let t = temen_llvm::translate_ll_path(bc).expect("translate Lua test-suite bitcode");
    let inst = temen_run::instantiate(t.module).expect("instantiate");
    let config = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![],
        env: vec![],
        ..RunConfig::default()
    };
    inst.run(backend, &config)
        .expect("run Lua test suite through the powerbox")
}

/// `main` returns 0 = every file's asserts held. On any failure the harness returns the 1-based index
/// of the first failing file and prints `<name>: FAILED: <error>` to stdout, surfaced here.
fn check(backend: Backend) {
    let out = run(backend);
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: Lua test suite failed\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout),
    );
}

#[test]
fn lua_testsuite_tree_walker() {
    check(Backend::TreeWalk);
}

#[test]
fn lua_testsuite_bytecode() {
    check(Backend::Bytecode);
}

#[test]
fn lua_testsuite_jit() {
    check(Backend::Jit);
}
