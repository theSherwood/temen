//! **#1936 — notifications in a region, posts to them, and a child's end post.**
//!
//! A notification is a `u64` of bits at an 8-aligned offset of a §13 region and a `u32` sequence
//! word 8 bytes on. Its waiter maps the region: it reads the sequence, takes the bits with a swap,
//! and if they were 0 waits on the sequence word with `atomic.wait`. A **post** is a signal-only
//! handle to it, minted from the region (SharedRegion op 6 `post(off, badge)`): op 7 `signal` ORs
//! the badge into the bits, bumps the sequence and wakes the word's waiters; op 5 closes it; every
//! other op answers `-EINVAL`, so a poster can't map, read or write the region.
//!
//! A spawn names a child's **end post** as a grant named `end`. The child never holds it; temen
//! signals it when the child ends, however it ends.
//!
//! Every case runs on the tree-walk oracle, the cooperative, parallel and debug bytecode drivers
//! ([`drivers::ALL`]) and the Cranelift JIT, and each must give the oracle's answer. The waits are
//! untimed: a wake that never comes is each engine's deadlock verdict, `ThreadFault`, not a hang.

use std::sync::Arc;

use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::JitOutcome;

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use drivers::{agree_on_every_driver, Ran};

/// Where the parent maps its region: the notification's bits are at `MAP`, its sequence at `MAP + 8`.
const MAP: u64 = 65536;
/// Spawn records, one per child: above the durable control words (rec.rs), 96 bytes apart.
const RECS: u64 = 17408;
/// Each child's one grant record, 16 bytes apart.
const GRANTS: u64 = 18048;
/// The grant names.
const POST_NAME: u64 = 18176;
const END_NAME: u64 = 18192;
/// Where the name `"budget"` sits, by which a detached child resolves the budget that paid for it.
const BUDGET_NAME: u64 = 18208;
/// Where a parent names its child functions: the name of the `i`th at `NAMES + 16 * i`.
const NAMES: u64 = 18240;

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The parent's powerbox: its `Instantiator` and `AddressSpace` over its 256 KiB window, a 2 MiB
/// `Budget` that pays for its children, and the child image of each function it exports, under the
/// export's name. Regions are OS shared memory, so the JIT can map them.
fn setup(m: &Module) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        h.set_region_factory(temen_run::new_shared_region);
        let i = h.grant_instantiator();
        let a = h.grant_address_space(0, 1 << 18);
        let b = h.grant_budget(-1, 1 << 21, -1);
        for e in &m.exports {
            let image = temen_ir::child_image_at(m, e.func).expect("child image");
            let mh = h.grant_module(&image);
            h.register_cap_name(&e.name, mh);
        }
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

/// `m` gives `want` on the oracle, every bytecode driver and the Cranelift JIT.
fn agree_everywhere(what: &str, m: &Module, want: &Ran) {
    agree_on_every_driver(what, m, &setup(m), want);
    assert_eq!(on_the_jit(m), want.result, "{what}: the Cranelift JIT");
}

// ---- Text-IR pieces. Each takes a register prefix `p`, so two uses in one function do not clash.

/// Mint a 64 KiB region through `vas` into register `vr`, and map it at [`MAP`].
fn mint_and_map() -> String {
    format!(
        "  mlen = i64.const 65536\n  mh = call.cap 5 5 (i64) -> (i64) vas (mlen)\n\
         \x20 vr = i32.wrap_i64 mh\n  mat = i64.const {MAP}\n  mz = i64.const 0\n\
         \x20 mprot = i32.const 3\n\
         \x20 mm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vr (mat, mz, mlen, mprot)\n"
    )
}

/// Mint a post to the notification at offset `off` of region `vr`, with `badge`, into register `dst`.
fn post(p: &str, off: u64, badge: u64, dst: &str) -> String {
    format!(
        "  {p}o = i64.const {off}\n  {p}b = i64.const {badge}\n\
         \x20 {p}h = call.cap 4 6 (i64, i64) -> (i64) vr ({p}o, {p}b)\n  {dst} = i32.wrap_i64 {p}h\n"
    )
}

/// Spawn child image `i` (named `name`) by record `RECS + 96 * i`, paid from `vbud`, granting the
/// handle in register `handle` under the name at `name_at`, into register `dst`.
fn spawn(p: &str, i: u64, name: &str, grant_name: (u64, usize), handle: &str, dst: &str) -> String {
    let rec_at = RECS + 96 * i;
    let grant_at = GRANTS + 16 * i;
    format!(
        "  {p}np = i64.const {}\n  {p}nl = i64.const {}\n  {p}mod = self.resolve {p}np {p}nl\n\
         {}  {p}ma = i64.const {}\n  i32.store {p}ma {p}mod\n\
         \x20 {p}ba = i64.const {}\n  i32.store {p}ba vbud\n  {p}rp = i64.const {rec_at}\n\
         \x20 {dst} = call.cap 6 17 (i64) -> (i32) vinst ({p}rp)\n",
        NAMES + 16 * i,
        name.len(),
        rec::grant(
            &format!("{p}g"),
            grant_at,
            grant_name.0,
            grant_name.1,
            handle
        ),
        rec_at + SpawnRec::MODULE_AT,
        rec_at + SpawnRec::BUDGET_AT,
    )
}

/// Blocks `first` to `first + 2`: the waiter's loop over the notification at [`MAP`]. It enters by
/// `br {first}(acc)` and leaves by `br {done}(acc)` once the bits it took, OR-ed into `acc`, cover
/// `want`. Each round reads the sequence, swaps the bits out, and waits on the sequence word,
/// untimed, when they were 0.
fn wait_for(p: &str, want: u64, first: u32, done: u32) -> String {
    let (take, sleep) = (first + 1, first + 2);
    format!(
        "block {first} ({p}acc: i64) {{
  {p}want = i64.const {want}
  {p}got = i64.and {p}acc {p}want
  {p}all = i64.eq {p}got {p}want
  br_if {p}all {done}({p}acc) {take}({p}acc)
}}
block {take} ({p}acc1: i64) {{
  {p}sa = i64.const {seq}
  {p}seq = i32.atomic.load {p}sa
  {p}ba = i64.const {MAP}
  {p}z = i64.const 0
  {p}bits = i64.atomic.rmw.xchg {p}ba {p}z
  {p}acc2 = i64.or {p}acc1 {p}bits
  {p}none = i64.eqz {p}bits
  br_if {p}none {sleep}({p}acc2, {p}seq) {first}({p}acc2)
}}
block {sleep} ({p}acc3: i64, {p}seq3: i32) {{
  {p}sa3 = i64.const {seq}
  {p}inf = i64.const -1
  {p}w = i32.atomic.wait {p}sa3 {p}seq3 {p}inf
  br {first}({p}acc3)
}}
",
        seq = MAP + 8,
    )
}

/// A parent `(i32 inst, i32 aspace, i32 budget) -> i64` whose block 0 runs `body`, then the blocks
/// `blocks`, with the child functions `children` after it, each exported under its name.
fn parent(body: &str, blocks: &str, children: &[(&str, &str)]) -> String {
    let mut names = String::new();
    let mut exports = String::new();
    let mut funcs = String::new();
    let mut recs = String::new();
    for (i, (name, func)) in children.iter().enumerate() {
        names.push_str(&format!("data {} \"{name}\"\n", NAMES + 16 * i as u64));
        exports.push_str(&format!("export {i} func \"{name}\" {}\n", i + 1));
        funcs.push_str(func);
        let r = SpawnRec {
            grants_ptr: GRANTS + 16 * i as u64,
            grants_n: 1,
            ..SpawnRec::default()
        };
        recs.push_str(&rec::segment(RECS + 96 * i as u64, &r));
    }
    format!(
        "memory 18
data {POST_NAME} \"post\"
data {END_NAME} \"end\"
data {BUDGET_NAME} \"budget\"
{names}{recs}{exports}func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vas: i32, vbud: i32) {{
{body}}}
{blocks}}}
{funcs}"
    )
}

/// A child that signals the post it was granted as `"post"`, and returns what the signal answered.
const CHILD_SIGNALS: &str = "func (i64) -> (i64) {
block 0 (v0: i64) {
  vnp = i64.const 18176
  vnl = i64.const 4
  vp = self.resolve vnp vnl
  vs = call.cap 4 7 () -> (i64) vp ()
  return vs
  }
}
";

/// Many posters and one waiter: three children, each holding a post to one notification with its
/// own badge, signal it; the parent waits until it has taken every badge, and returns them.
#[test]
fn a_waiter_takes_every_posters_badge() {
    let mut body = mint_and_map();
    let children: Vec<(&str, &str)> = ["c0", "c1", "c2"]
        .iter()
        .map(|n| (*n, CHILD_SIGNALS))
        .collect();
    for (i, badge) in [1u64, 2, 4].iter().enumerate() {
        body.push_str(&post(&format!("p{i}"), 0, *badge, &format!("vp{i}")));
        body.push_str(&spawn(
            &format!("s{i}"),
            i as u64,
            children[i].0,
            (POST_NAME, 4),
            &format!("vp{i}"),
            &format!("vc{i}"),
        ));
    }
    body.push_str("  vacc = i64.const 0\n  br 1(vacc)\n");
    let blocks = format!(
        "{}block 4 (vgot: i64) {{\n  return vgot\n}}\n",
        wait_for("w", 7, 1, 4)
    );
    let m = module(&parent(&body, &blocks, &children));
    agree_everywhere("every badge reaches the waiter", &m, &ok(7));
}

/// A post signals and closes, and nothing else: it can't map, size or re-post its region. A region
/// handle can't signal. A post needs a nonzero badge and 12 bytes at an 8-aligned offset. A signal
/// ORs its badge into the bits and bumps the sequence. Each check sets one bit of the answer.
#[test]
fn a_post_signals_and_nothing_else() {
    let einval = |p: &str, call: &str| -> String {
        format!(
            "  {p}r = {call}\n  {p}e = i64.const -22\n  {p}is = i64.eq {p}r {p}e\n\
             \x20 {p}x = i64.extend_i32_u {p}is\n"
        )
    };
    let mut body = mint_and_map();
    body.push_str(&post("p", 0, 5, "vp"));
    body.push_str("  vz = i64.const 0\n  vlen = i64.const 4096\n  vprot = i32.const 3\n  vat = i64.const 131072\n");
    body.push_str(&einval(
        "qa",
        "call.cap 4 0 (i64, i64, i64, i32) -> (i64) vp (vat, vz, vlen, vprot)",
    ));
    body.push_str(&einval("qb", "call.cap 4 2 () -> (i64) vp ()"));
    body.push_str(&einval("qc", "call.cap 4 3 () -> (i64) vp ()"));
    body.push_str("  vone = i64.const 1\n");
    body.push_str(&einval(
        "qd",
        "call.cap 4 6 (i64, i64) -> (i64) vp (vz, vone)",
    ));
    body.push_str(&einval("qe", "call.cap 4 7 () -> (i64) vr ()"));
    body.push_str(&einval(
        "qf",
        "call.cap 4 6 (i64, i64) -> (i64) vr (vz, vz)",
    ));
    body.push_str("  vfour = i64.const 4\n  vlast = i64.const 65528\n  vfits = i64.const 65520\n");
    body.push_str(&einval(
        "qg",
        "call.cap 4 6 (i64, i64) -> (i64) vr (vfour, vone)",
    ));
    body.push_str(&einval(
        "qh",
        "call.cap 4 6 (i64, i64) -> (i64) vr (vlast, vone)",
    ));
    // The last notification that fits, and a signal through the first post: its badge in the bits,
    // the sequence bumped.
    body.push_str(&format!(
        "  vok = call.cap 4 6 (i64, i64) -> (i64) vr (vfits, vone)\n  vokz = i64.const 0\n\
         \x20 viok = i64.lt_s vok vokz\n  vjok = i32.eqz viok\n  ix = i64.extend_i32_u vjok\n\
         \x20 vs = call.cap 4 7 () -> (i64) vp ()\n  vsz = i64.eqz vs\n  jx = i64.extend_i32_u vsz\n\
         \x20 vba = i64.const {MAP}\n  vbits = i64.load vba\n  vfive = i64.const 5\n\
         \x20 vbe = i64.eq vbits vfive\n  kx = i64.extend_i32_u vbe\n\
         \x20 vsa = i64.const {}\n  vseq = i32.load vsa\n  vsone = i32.const 1\n\
         \x20 vse = i32.eq vseq vsone\n  lx = i64.extend_i32_u vse\n",
        MAP + 8
    ));
    let mut acc = String::from("  r0 = i64.const 0\n");
    for (i, x) in [
        "qax", "qbx", "qcx", "qdx", "qex", "qfx", "qgx", "qhx", "ix", "jx", "kx", "lx",
    ]
    .iter()
    .enumerate()
    {
        acc.push_str(&format!(
            "  sh{i} = i64.const {i}\n  b{i} = i64.shl {x} sh{i}\n  r{} = i64.or r{i} b{i}\n",
            i + 1
        ));
    }
    body.push_str(&acc);
    body.push_str("  return r12\n");
    let m = module(&parent(&body, "", &[]));
    agree_everywhere("a post's ops", &m, &ok(0xFFF));
}

/// A child that returns 7.
const CHILD_RETURNS: &str = "func (i64) -> (i64) {
block 0 (v0: i64) {
  v7 = i64.const 7
  return v7
  }
}
";

/// A child that traps.
const CHILD_TRAPS: &str = "func (i64) -> (i64) {
block 0 (v0: i64) {
  unreachable
  }
}
";

/// A child that spins until something ends it: its parent's kill, or its fuel.
const CHILD_SPINS: &str = "func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
}
block 1 () {
  br 1()
  }
}
";

/// A parent that spawns `child` with an end post (badge 1), runs `then` with the child in `vc`, and
/// waits, untimed, for the end post, returning the bits it took.
fn ends(child: &str, then: &str) -> Module {
    let mut body = mint_and_map();
    body.push_str(&post("p", 0, 1, "vp"));
    body.push_str(&spawn("s", 0, "c", (END_NAME, 3), "vp", "vc"));
    body.push_str(then);
    body.push_str("  vacc = i64.const 0\n  br 1(vacc)\n");
    let blocks = format!(
        "{}block 4 (vgot: i64) {{\n  return vgot\n}}\n",
        wait_for("w", 1, 1, 4)
    );
    module(&parent(&body, &blocks, &[("c", child)]))
}

/// A child's end signals its end post, however it ends.
#[test]
fn a_childs_end_signals_its_end_post() {
    agree_everywhere("a child that returns", &ends(CHILD_RETURNS, ""), &ok(1));
    agree_everywhere("a child that traps", &ends(CHILD_TRAPS, ""), &ok(1));
    let kill = "  vk = call.cap 6 12 (i32) -> (i32) vinst (vc)\n";
    agree_everywhere("a child its parent kills", &ends(CHILD_SPINS, kill), &ok(1));
}

/// A child that runs out of fuel ends, and its end post is signalled: its budget allows it a little
/// fuel, and it spins.
#[test]
fn a_child_out_of_fuel_signals_its_end_post() {
    let mut body = mint_and_map();
    body.push_str(&post("p", 0, 1, "vp"));
    // A sub-budget with a little fuel, for the child's window to spend.
    body.push_str(
        "  ffuel = i64.const 100000\n  fmem = i64.const -1\n  fspawn = i64.const -1\n\
         \x20 fbh = call.cap 14 0 (i64, i64, i64) -> (i64) vbud (ffuel, fmem, fspawn)\n\
         \x20 vbud2 = i32.wrap_i64 fbh\n",
    );
    body.push_str(
        &spawn("s", 0, "c", (END_NAME, 3), "vp", "vc")
            .replace("i32.store sba vbud", "i32.store sba vbud2"),
    );
    body.push_str("  vacc = i64.const 0\n  br 1(vacc)\n");
    let blocks = format!(
        "{}block 4 (vgot: i64) {{\n  return vgot\n}}\n",
        wait_for("w", 1, 1, 4)
    );
    let m = module(&parent(&body, &blocks, &[("c", CHILD_SPINS)]));
    agree_everywhere("a child out of fuel", &m, &ok(1));
}

/// The child never holds its end post: its name binds nothing in the child's directory.
#[test]
fn a_child_cannot_see_its_end_post() {
    let child = format!(
        "func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vnp = i64.const {END_NAME}
  vnl = i64.const 3
  vp = self.resolve vnp vnl
  vz = i32.const 0
  vneg = i32.lt_s vp vz
  vr = i64.extend_i32_u vneg
  return vr
  }}
}}
"
    );
    let mut body = mint_and_map();
    body.push_str(&post("p", 0, 1, "vp"));
    body.push_str(&spawn("s", 0, "c", (END_NAME, 3), "vp", "vc"));
    body.push_str("  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vinst (vc)\n  return vj\n");
    let m = module(&parent(&body, "", &[("c", &child)]));
    agree_everywhere("the end post is not the child's", &m, &ok(1));
}

/// A spawn whose `end` grant is not a post refuses, as a grant its spawner may not make does.
#[test]
fn an_end_grant_must_be_a_post() {
    let mut body = mint_and_map();
    body.push_str(&spawn("s", 0, "c", (END_NAME, 3), "vr", "vc"));
    body.push_str("  vr64 = i64.extend_i32_s vc\n  return vr64\n");
    let m = module(&parent(&body, "", &[("c", CHILD_RETURNS)]));
    let want = Ran {
        result: Err(Trap::CapFault),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_everywhere("a region is not an end post", &m, &want);
}

/// A child whose root returns while a thread it spawned still runs: its end post is signalled
/// once, when the child ends, and its waiting parent is woken, never read as deadlocked.
#[test]
fn a_child_whose_thread_outlives_its_root_signals_its_end_post() {
    let child = "func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  vt = thread.spawn 2 vz vz
  v7 = i64.const 7
  return v7
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz = i64.const 0
  br 1(vz)
}
block 1 (cnt: i64) {
  vlim = i64.const 200000
  vlt = i64.lt_u cnt vlim
  br_if vlt 2(cnt) 3()
}
block 2 (cnt2: i64) {
  vone = i64.const 1
  vnext = i64.add cnt2 vone
  br 1(vnext)
}
block 3 () {
  vd = i64.const 0
  return vd
  }
}
";
    for _ in 0..20 {
        agree_everywhere("a thread outlives its root", &ends(child, ""), &ok(1));
    }
}
