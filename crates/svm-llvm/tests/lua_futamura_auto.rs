//! **Capture automation, end to end: many programs, zero per-program configuration.**
//!
//! Every one of these scripts runs through the SAME automated pipeline (`futamura::auto`): one
//! profiling run discovers the call sites (Lua / C / tail, loops deduped), captures each callee
//! occupancy at its own dispatch, derives the pins, overlays, and return tags, specializes, embeds,
//! and executes. The matrix covers every call shape the `lua_futamura_call*` tests established by
//! hand — plus a combined program exercising several at once — each asserted byte-identical to the
//! interpreter and fully dispatch-folded (`br_table == 0`).
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_auto -- --nocapture`

mod futamura;

fn check(name: &str, script: &str, expect: &[u8]) {
    let r = futamura::auto(script);
    println!(
        "{name}: {} blocks, br_table={}, sites lua={} c={} tail={} -> {:?}",
        r.residual_blocks,
        r.br_tables,
        r.lua_sites,
        r.c_sites,
        r.tail_sites,
        String::from_utf8_lossy(&r.residual_stdout)
    );
    assert_eq!(r.baseline_stdout, expect, "{name}: baseline output");
    assert_eq!(
        r.residual_stdout, r.baseline_stdout,
        "{name}: residual byte-identical"
    );
    assert_eq!(r.br_tables, 0, "{name}: dispatch fully folded");
}

#[test]
fn arithmetic_result() {
    check(
        "arith",
        "local function add(a, b) return a + b end\nprint(add(2, 3) * 10)\n",
        b"50\n",
    );
}

#[test]
fn float_return() {
    check(
        "float",
        "local function half(x) return x / 2 end\nprint(half(5) + 0.25)\n",
        b"2.75\n",
    );
}

#[test]
fn sequential_shared_callinfo() {
    check(
        "seq",
        "local function add(a, b) return a + b end\n\
         local function mul(a, b) return a * b end\n\
         print(add(2, 3) + mul(4, 5))\n",
        b"25\n",
    );
}

#[test]
fn nested_two_frames() {
    check(
        "nested",
        "local function inner(x) return x + 1 end\n\
         local function outer(y) return inner(y) * 2 end\n\
         print(outer(10))\n",
        b"22\n",
    );
}

#[test]
fn tail_call() {
    check(
        "tail",
        "local function inner(x) return x + 1 end\n\
         local function outer(y) return inner(y) end\n\
         print(outer(10))\n",
        b"11\n",
    );
}

#[test]
fn call_bearing_loop() {
    check(
        "loop",
        "local function add(a, b) return a + b end\n\
         local x = 0\n\
         for i = 1, 5 do x = add(x, 3) end\n\
         print(x)\n",
        b"15\n",
    );
}

#[test]
fn combined_shapes() {
    // Nested calls, a tail call, sequential shared-node reuse, an uncalled definition, and a loop —
    // one program, no shape-specific setup. (All top-level returns integer: the selective model
    // keys ONE return tag per frame node, so mixing int/float returns through the same node is a
    // documented limitation — see the driver doc.)
    check(
        "combined",
        "local function double(x) return x * 2 end\n\
         local function addone(x) return x + 1 end\n\
         local function both(x) return double(addone(x)) end\n\
         local unused = function(q) return q end\n\
         local acc = 0\n\
         for i = 1, 3 do acc = acc + both(i) end\n\
         print(acc + double(7))\n",
        b"32\n",
    );
}
