//! **#2220 — a parent grants a capability into its running child** (`Instantiator.grant`, op 19):
//! `grant(child, handle)` re-grants `handle`, resolved in the parent's powerbox, into the powerbox of
//! a detached child that is already running or parked, with the spawn grants' policy, and answers
//! the child's index of it. The parent tells the child the index itself, here through a word of a
//! region it pre-mapped into the child at the spawn; the child gains no operation. INVARIANTS #3
//! already sanctions the transfer: the granter holds both ends.
//!
//! Every case runs on the tree-walk oracle, the cooperative, parallel and debug bytecode drivers
//! ([`drivers::SCHEDULING`]) and the Cranelift JIT, and each must give the oracle's answer. The
//! resumable `Vcpu`, whose host runs the children, traps `ThreadFault` when op 19 runs, as it does
//! for `poll`/`detach`/`kill` (#2083).
//!
//! A granted index depends on what the child minted meanwhile (a grant takes the lowest free slot of
//! the child's table), so no case pins one: each child learns its index from its parent.

use std::sync::Arc;

use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::JitOutcome;

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use drivers::{agree_on, run_on, Driver, Ran, SCHEDULING};

/// Where the first spawn record sits, and the second's: above the durable control words (rec.rs).
const REC_A: u64 = 17408;
const REC_B: u64 = 17504;
/// The mailbox region: mapped here in the parent, and pre-mapped at the same offset into each child.
/// Each child waits on its word for the index of what its parent grants it.
const MAILBOX: u64 = 65536;
/// Where the parent maps the region it grants, and where a child maps it once granted.
const GRANTED: u64 = 131072;
/// Where a record's pre-mapped region handle sits, from the record's start.
const REGION_AT: u64 = 72;
/// The mailbox word a child posts once it runs, which its parent waits for before it grants.
const READY: u64 = 24;
/// Where a parent keeps its `Instantiator`, child and region handles across its own wait: a block
/// sees only its own values and parameters.
const SAVED: u64 = 20480;
/// Where the name `"budget"` sits, by which a detached child resolves the budget that paid for it.
const BUDGET_NAME: u64 = 17664;

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The parent's powerbox: its `Instantiator` and `AddressSpace` over its 256 KiB window, and a 2 MiB
/// `Budget` that pays for its children. Regions are OS shared memory, so the JIT can alias them.
fn setup(m: &Module) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        h.set_region_factory(temen_run::new_shared_region);
        h.set_self_module(&Arc::new(m.clone()));
        let i = h.grant_instantiator(0, 1 << 18);
        let a = h.grant_address_space(0, 1 << 18);
        let b = h.grant_budget(-1, 1 << 21, -1);
        (h, vec![Value::I32(i), Value::I32(a), Value::I32(b)])
    }
}

fn ok(v: i64) -> Ran {
    Ran {
        result: Ok(vec![Value::I64(v)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    }
}

fn trapped(t: Trap) -> Ran {
    Ran {
        result: Err(t),
        stdout: Vec::new(),
        stderr: Vec::new(),
    }
}

/// `m` on the Cranelift JIT, over the powerbox [`setup`] builds, on a thread of its own: a run that
/// hangs fails here instead of stalling the suite.
fn on_the_jit(m: &Module) -> Result<Vec<Value>, Trap> {
    let m = Arc::new(m.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut host, args) = setup(&m)();
        let args: Vec<i64> = args
            .iter()
            .map(|v| match v {
                Value::I32(x) => i64::from(*x),
                other => panic!("unexpected arg {other:?}"),
            })
            .collect();
        let window = MemLayout::image(vec![0u8; 1 << 18]);
        let (jo, _) = temen_run::jit_cap_run(&m, 0, &args, &window, 0, 0, &mut host, None)
            .expect("the JIT compiles it");
        let _ = tx.send(jo);
    });
    match rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the JIT run hung")
    {
        JitOutcome::Returned(v) => Ok(v.into_iter().map(Value::I64).collect()),
        JitOutcome::Trapped(k) => Err(Trap::from_code(k as i64).expect("a trap code")),
        other => panic!("the JIT run ended as {other:?}"),
    }
}

/// `m` gives `want` on the oracle, every scheduling bytecode driver and the Cranelift JIT.
fn agree_everywhere(what: &str, m: &Module, want: &Ran) {
    agree_on(&SCHEDULING, what, m, &setup(m), want);
    assert_eq!(on_the_jit(m), want.result, "{what}: the Cranelift JIT");
}

// ---- Text-IR pieces. Each takes a register prefix `p`, so two uses in one function do not clash.

/// Mint a 64 KiB region through `vas` into register `r`, and map it at `at` of the window.
fn mint_and_map(p: &str, r: &str, at: u64) -> String {
    format!(
        "  {p}len = i64.const 65536\n  {p}h = call.cap 5 5 (i64) -> (i64) vas ({p}len)\n\
         \x20 {r} = i32.wrap_i64 {p}h\n  {p}at = i64.const {at}\n  {p}z = i64.const 0\n\
         \x20 {p}prot = i32.const 3\n\
         \x20 {p}m = call.cap 4 0 (i64, i64, i64, i32) -> (i64) {r} ({p}at, {p}z, {p}len, {p}prot)\n"
    )
}

/// Spawn by the record at `rec_at`, paid from `vbud`, into register `dst`; with `mailbox`, that
/// region is pre-mapped into the child.
fn spawn(p: &str, rec_at: u64, mailbox: Option<&str>, dst: &str) -> String {
    let region = mailbox.map_or(String::new(), |r| {
        format!(
            "  {p}ra = i64.const {}\n  i32.store {p}ra {r}\n",
            rec_at + REGION_AT
        )
    });
    format!(
        "  {p}ba = i64.const {}\n  i32.store {p}ba vbud\n{region}  {p}rp = i64.const {rec_at}\n\
         \x20 {dst} = call.cap 6 17 (i64) -> (i32) vinst ({p}rp)\n",
        rec_at + rec::BUDGET_AT
    )
}

/// `grant(child, handle)` (op 19) into register `dst`.
fn grant(p: &str, child: &str, handle: &str, dst: &str) -> String {
    format!(
        "  {p}h = i64.extend_i32_s {handle}\n\
         \x20 {dst} = call.cap 6 19 (i32, i64) -> (i32) vinst ({child}, {p}h)\n"
    )
}

/// Store the `i32` register `v` at `at` and wake whoever waits on it.
fn post(p: &str, at: u64, v: &str) -> String {
    format!(
        "  {p}a = i64.const {at}\n  i32.atomic.store {p}a {v}\n  {p}c = i32.const 1\n\
         \x20 {p}n = atomic.notify {p}a {p}c\n"
    )
}

/// Store the `i32` registers `regs` at [`SAVED`] on, for [`restore`] in a later block.
fn save(regs: &[&str]) -> String {
    regs.iter()
        .enumerate()
        .map(|(i, r)| {
            let at = SAVED + 4 * i as u64;
            format!("  sv{i} = i64.const {at}\n  i32.store sv{i} {r}\n")
        })
        .collect()
}

/// Load the `i32` registers [`save`] stored, under the same names.
fn restore(regs: &[&str]) -> String {
    regs.iter()
        .enumerate()
        .map(|(i, r)| {
            let at = SAVED + 4 * i as u64;
            format!("  rs{i} = i64.const {at}\n  {r} = i32.load rs{i}\n")
        })
        .collect()
}

/// Blocks `first` to `first + 3` of a function: wait until the `i32` at `at` is nonzero, then branch
/// to block `done` with it. The function enters by `br {first}(k)`, `k` the number of 100 ms waits it
/// allows; when they run out it returns -1000, so a child its parent never wakes ends rather than
/// hangs the run.
fn await_word(p: &str, at: u64, first: u32, done: u32) -> String {
    let (wait, gave_up, sleep) = (first + 1, first + 2, first + 3);
    format!(
        "block {first} ({p}k: i64) {{
  {p}a = i64.const {at}
  {p}v = i32.atomic.load {p}a
  {p}z = i32.const 0
  {p}set = i32.ne {p}v {p}z
  br_if {p}set {done}({p}v) {wait}({p}k)
}}
block {wait} ({p}k1: i64) {{
  {p}z1 = i64.const 0
  {p}out = i64.eq {p}k1 {p}z1
  br_if {p}out {gave_up}() {sleep}({p}k1)
}}
block {gave_up} () {{
  {p}bad = i64.const -1000
  return {p}bad
}}
block {sleep} ({p}k3: i64) {{
  {p}a3 = i64.const {at}
  {p}e = i32.const 0
  {p}to = i64.const 100000000
  {p}w = i32.atomic.wait {p}a3 {p}e {p}to
  {p}one = i64.const 1
  {p}k4 = i64.sub {p}k3 {p}one
  br {first}({p}k4)
}}
"
    )
}

/// A child (entry `func 1`) that posts `1` at mailbox word [`READY`], then waits on mailbox word
/// `slot` for the index of a region, maps it at [`GRANTED`] and returns the `i64` there, plus a
/// million times the map's status.
fn child_reads_granted(slot: u64) -> String {
    format!(
        "func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vone = i32.const 1
{ready}  wn = i64.const 100
  br 1(wn)
}}
{wait}block 5 (vh: i32) {{
  vlen = i64.const 65536
  vat = i64.const {GRANTED}
  vz = i64.const 0
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vh (vat, vz, vlen, vprot)
  vword = i64.load vat
  vmil = i64.const 1000000
  vst = i64.mul vm vmil
  vr = i64.add vword vst
  return vr
  }}
}}
",
        ready = post("r", MAILBOX + READY, "vone"),
        wait = await_word("w", MAILBOX + slot, 1, 5),
    )
}

/// A child (entry `func 1`) that returns 7.
const CHILD_RETURNS_7: &str = "func (i64) -> (i64) {
block 0 (v0: i64) {
  v7 = i64.const 7
  return v7
  }
}
";

/// A child (entry `func 1`) that spins until it is killed.
const CHILD_SPINS: &str = "func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
}
block 1 () {
  br 1()
  }
}
";

/// A parent `(i32 inst, i32 aspace, i32 budget) -> i64` whose block 0 runs `body` and returns `vr`,
/// over the records `recs` and the name a child resolves its `"budget"` by, with the child functions
/// `children` after it.
fn parent(recs: &[(u64, SpawnRec)], body: &str, children: &str) -> String {
    let recs: String = recs.iter().map(|(at, r)| rec::segment(*at, r)).collect();
    format!(
        "memory 18
data {BUDGET_NAME} \"budget\"
{recs}func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vas: i32, vbud: i32) {{
{body}  return vr
  }}
}}
{children}"
    )
}

/// The record of a child (entry `entry`) whose mailbox is pre-mapped at [`MAILBOX`].
fn mailbox_rec(entry: u32) -> SpawnRec {
    SpawnRec {
        child_off: MAILBOX,
        ..SpawnRec::v1(entry)
    }
}

/// #2220 — a running detached child receives a region its parent granted it: the parent writes a word
/// into the region and waits until the child has posted that it runs; then it grants the region and
/// posts the index in the mailbox the child waits on. The child maps the region and returns the word.
#[test]
fn a_running_child_maps_a_region_its_parent_granted_it() {
    let held = ["vinst", "vch", "vreg"];
    let body = format!(
        "{}{}  vword = i64.const 4242\n  vwat = i64.const {GRANTED}\n  i64.store vwat vword\n\
         {}{}  wn = i64.const 100\n  br 1(wn)\n}}\n{}block 5 (vready: i32) {{\n{}{}{}\
         \x20 vr = call.cap 6 1 (i32) -> (i64) vinst (vch)\n",
        mint_and_map("m", "vm", MAILBOX),
        mint_and_map("g", "vreg", GRANTED),
        spawn("s", REC_A, Some("vm"), "vch"),
        save(&held),
        await_word("w", MAILBOX + READY, 1, 5),
        restore(&held),
        grant("q", "vch", "vreg", "vgot"),
        post("p", MAILBOX, "vgot"),
    );
    let m = module(&parent(
        &[(REC_A, mailbox_rec(1))],
        &body,
        &child_reads_granted(0),
    ));
    agree_everywhere("a grant into a running child", &m, &ok(4242));
}

/// #2220 — the grant holds one level down (INVARIANTS #14, nesting): the root spawns child C and joins
/// it; C, a detached child, does what [`a_running_child_maps_a_region_its_parent_granted_it`]'s parent
/// does, through its own starter `AddressSpace` and the `"budget"` that paid for it, granting a region
/// into its own running child G (func 2), which returns the word C wrote there.
#[test]
fn a_detached_child_grants_into_its_own_running_child() {
    let held = ["vinst", "vch", "vreg"];
    let root = format!(
        "{}  vr = call.cap 6 1 (i32) -> (i64) vinst (vch)\n",
        spawn("s", REC_A, None, "vch"),
    );
    let c = format!(
        "func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  vinst = i32.wrap_i64 v0
  vas = i32.wrap_i64 v1
  vnp = i64.const {name}
  vnl = i64.const 6
  vbud = self.resolve vnp vnl
{}{}  vword = i64.const 4242
  vwat = i64.const {GRANTED}
  i64.store vwat vword
{}{}  wn = i64.const 100
  br 1(wn)
}}
{}block 5 (vready: i32) {{
{}{}{}  vr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  return vr
  }}
}}
",
        mint_and_map("m", "vm", MAILBOX),
        mint_and_map("g", "vreg", GRANTED),
        spawn("s", REC_B, Some("vm"), "vch"),
        save(&held),
        await_word("w", MAILBOX + READY, 1, 5),
        restore(&held),
        grant("q", "vch", "vreg", "vgot"),
        post("p", MAILBOX, "vgot"),
        name = BUDGET_NAME,
    );
    let m = module(&parent(
        &[(REC_A, SpawnRec::v1(1)), (REC_B, mailbox_rec(2))],
        &root,
        &format!("{c}{}", child_reads_granted(0)),
    ));
    agree_everywhere("a grant from a detached child into its own", &m, &ok(4242));
}

/// #2220 — a parent connects two running children by granting both one region, and they rendezvous on
/// it: child A writes 77 into the region and wakes the word it posts there; child B waits on that word
/// and returns what A wrote. The parent returns `A * 1000 + B`.
#[test]
fn two_running_children_rendezvous_on_a_region_their_parent_granted_both() {
    let body = format!(
        "{}{}{}{}{}{}{}{}  vja = call.cap 6 1 (i32) -> (i64) vinst (vca)\n\
         \x20 vjb = call.cap 6 1 (i32) -> (i64) vinst (vcb)\n  vk = i64.const 1000\n\
         \x20 vx = i64.mul vja vk\n  vr = i64.add vx vjb\n",
        mint_and_map("m", "vm", MAILBOX),
        mint_and_map("g", "vreg", GRANTED),
        spawn("sa", REC_A, Some("vm"), "vca"),
        spawn("sb", REC_B, Some("vm"), "vcb"),
        grant("qa", "vca", "vreg", "vga"),
        grant("qb", "vcb", "vreg", "vgb"),
        post("pa", MAILBOX, "vga"),
        post("pb", MAILBOX + 8, "vgb"),
    );
    // A: wait for its index, map the region, write 77 behind the flag word, raise the flag.
    let a = format!(
        "func (i64) -> (i64) {{
block 0 (v0: i64) {{
  wn = i64.const 100
  br 1(wn)
}}
{wait}block 5 (vh: i32) {{
  vlen = i64.const 65536
  vat = i64.const {GRANTED}
  vz = i64.const 0
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vh (vat, vz, vlen, vprot)
  vdat = i64.const {data}
  v77 = i64.const 77
  i64.store vdat v77
  vone = i32.const 1
{flag}  vr = i64.const 1
  return vr
  }}
}}
",
        wait = await_word("w", MAILBOX, 1, 5),
        data = GRANTED + 8,
        flag = post("f", GRANTED, "vone"),
    );
    // B: wait for its index, map the region, wait for A's flag there, return what A wrote.
    let b = format!(
        "func (i64) -> (i64) {{
block 0 (v0: i64) {{
  wn = i64.const 100
  br 1(wn)
}}
{wait}block 5 (vh: i32) {{
  vlen = i64.const 65536
  vat = i64.const {GRANTED}
  vz = i64.const 0
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vh (vat, vz, vlen, vprot)
  wn2 = i64.const 100
  br 6(wn2)
}}
{flag}block 10 (vf: i32) {{
  vdat = i64.const {data}
  vr = i64.load vdat
  return vr
  }}
}}
",
        wait = await_word("w", MAILBOX + 8, 1, 5),
        flag = await_word("x", GRANTED, 6, 10),
        data = GRANTED + 8,
    );
    let m = module(&parent(
        &[(REC_A, mailbox_rec(1)), (REC_B, mailbox_rec(2))],
        &body,
        &format!("{a}{b}"),
    ));
    agree_everywhere("two children meeting on a granted region", &m, &ok(1077));
}

/// The body of a parent that spawns its child (no mailbox) by [`REC_A`], runs `before`, grants it a
/// region, then runs `after`: `grant * 1000 + vj`, which `before` or `after` defines.
fn grant_between(before: &str, after: &str) -> String {
    format!(
        "{}{}{before}{}{after}  vg64 = i64.extend_i32_s vgot\n  vk = i64.const 1000\n\
         \x20 vx = i64.mul vg64 vk\n  vr = i64.add vx vj\n",
        mint_and_map("g", "vreg", GRANTED),
        spawn("s", REC_A, None, "vch"),
        grant("q", "vch", "vreg", "vgot"),
    )
}

/// #2220 — a grant to a child that has ended fails `-EINVAL`: one that returned, which a `wait` saw end
/// and left for its `join` (`grant * 1000 + join`), and one that was killed, which the `wait` reaped
/// (`grant * 1000 + wait`, the `wait` answering `THREAD_FAULT`).
#[test]
fn a_grant_to_an_ended_child_fails() {
    let returned = module(&parent(
        &[(REC_A, SpawnRec::v1(1))],
        &grant_between(
            "  vw = call.cap 6 18 (i32) -> (i64) vinst (vch)\n",
            "  vj = call.cap 6 1 (i32) -> (i64) vinst (vch)\n",
        ),
        CHILD_RETURNS_7,
    ));
    agree_everywhere(
        "a grant after the child returned",
        &returned,
        &ok(-22 * 1000 + 7),
    );
    let killed = module(&parent(
        &[(REC_A, SpawnRec::v1(1))],
        &grant_between(
            "  vk1 = call.cap 6 12 (i32) -> (i32) vinst (vch)\n\
             \x20 vj = call.cap 6 18 (i32) -> (i64) vinst (vch)\n",
            "",
        ),
        CHILD_SPINS,
    ));
    agree_everywhere(
        "a grant after the child was killed",
        &killed,
        &ok(-22 * 1000 + temen_ir::trap_code::THREAD_FAULT),
    );
    // The resumable `Vcpu`'s host runs its children: op 19 fails closed there when it runs (#2083).
    assert_eq!(
        run_on(Driver::Vcpu, &returned, &setup(&returned)),
        Some(trapped(Trap::ThreadFault)),
        "the Vcpu fails closed"
    );
}

/// #2220 — a grant into a carve child (op 0) is refused `-EINVAL`, live as the child is: the carve
/// path is retired (INVARIANTS #13, #1867), and no new feature lands on it. The parent grants into a
/// carve child that spins until it is killed, then kills and waits for it: `grant * 1000 + wait`.
#[test]
fn a_grant_into_a_carve_child_is_refused() {
    let body = format!(
        "{}  ve = i64.const 1\n  voff = i64.const {MAILBOX}\n  vsl = i64.const 16\n\
         \x20 vq = i64.const 0\n\
         \x20 vch = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vinst (ve, voff, vsl, vq)\n{}\
         \x20 vk1 = call.cap 6 12 (i32) -> (i32) vinst (vch)\n\
         \x20 vj = call.cap 6 18 (i32) -> (i64) vinst (vch)\n  vg64 = i64.extend_i32_s vgot\n\
         \x20 vk = i64.const 1000\n  vx = i64.mul vg64 vk\n  vr = i64.add vx vj\n",
        mint_and_map("g", "vreg", GRANTED),
        grant("q", "vch", "vreg", "vgot"),
    );
    let m = module(&parent(&[], &body, CHILD_SPINS));
    agree_everywhere(
        "a grant into a carve child",
        &m,
        &ok(-22 * 1000 + temen_ir::trap_code::THREAD_FAULT),
    );
}

/// #2220 — a forged handle faults as a spawn grant's does (`CapFault`), and so does a handle no spawn
/// may grant (the parent's `AddressSpace`, which names its own window's coordinates): the handle is
/// checked before the child, so the answer does not depend on how far the child has got.
#[test]
fn a_handle_no_spawn_could_grant_faults() {
    for (what, handle) in [
        ("a forged handle", "  vforged = i32.const 1193046\n"),
        (
            "the parent's AddressSpace",
            "  vforged = i32.add vas vzero\n",
        ),
    ] {
        let body = format!(
            "  vzero = i32.const 0\n{}{handle}{}  vr = call.cap 6 1 (i32) -> (i64) vinst (vch)\n",
            spawn("s", REC_A, None, "vch"),
            grant("q", "vch", "vforged", "vgot"),
        );
        let m = module(&parent(&[(REC_A, SpawnRec::v1(1))], &body, CHILD_RETURNS_7));
        agree_everywhere(what, &m, &trapped(Trap::CapFault));
    }
}

/// #2220 — a grant into a child whose table is full fails `-EMFILE`: the parent grants one region
/// into a running child until it is refused, then releases the child (mailbox word 16) and joins
/// it. It returns `errno * 1_000_000 + grants * 1000 + join`. A fresh detached child holds four of
/// its table's 256 slots — its `Instantiator`, `AddressSpace`, `"budget"` and the pre-mapped
/// mailbox — so 252 grants land.
#[test]
fn a_grant_into_a_full_table_fails_emfile() {
    let body = format!(
        "{}{}{}  vn0 = i64.const 0\n  br 1(vinst, vch, vreg, vn0)\n}}
block 1 (vi: i32, vc: i32, vh: i32, vn: i64) {{
  vh64 = i64.extend_i32_s vh
  vg = call.cap 6 19 (i32, i64) -> (i32) vi (vc, vh64)
  vz = i32.const 0
  vneg = i32.lt_s vg vz
  br_if vneg 3(vi, vc, vg, vn) 2(vi, vc, vh, vn)
}}
block 2 (vi2: i32, vc2: i32, vh2: i32, vn2: i64) {{
  vone = i64.const 1
  vn3 = i64.add vn2 vone
  br 1(vi2, vc2, vh2, vn3)
}}
block 3 (vi3: i32, vc3: i32, vg3: i32, vn4: i64) {{
  vgo = i32.const 1
{}  vj = call.cap 6 1 (i32) -> (i64) vi3 (vc3)
  vg64 = i64.extend_i32_s vg3
  vmil = i64.const 1000000
  vx = i64.mul vg64 vmil
  vk = i64.const 1000
  vy = i64.mul vn4 vk
  vxy = i64.add vx vy
  vr = i64.add vxy vj
",
        mint_and_map("m", "vm", MAILBOX),
        mint_and_map("g", "vreg", GRANTED),
        spawn("s", REC_A, Some("vm"), "vch"),
        post("p", MAILBOX + 16, "vgo"),
    );
    let child = format!(
        "func (i64) -> (i64) {{
block 0 (v0: i64) {{
  wn = i64.const 100
  br 1(wn)
}}
{wait}block 5 (vgo: i32) {{
  v7 = i64.const 7
  return v7
  }}
}}
",
        wait = await_word("w", MAILBOX + 16, 1, 5),
    );
    let m = module(&parent(&[(REC_A, mailbox_rec(1))], &body, &child));
    agree_everywhere(
        "grants until the child's table is full",
        &m,
        &ok(-24 * 1_000_000 + 252 * 1000 + 7),
    );
}
