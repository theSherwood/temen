//! #2958 — the kernel-route probe's park-and-wake programs (`--kernel-spike`).

/// The #2868 pingpong ([`crate::kernel_spike::pingpong_ir`]): `func 0 (n, k)`.
pub const PINGPONG_IR: &str = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (vn: i64, vk: i64) {
  vz = i64.const 0
  br 1(vn, vk, vz)
}
block 1 (vn1: i64, vk1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vk1
  br_if vlt1 2(vn1, vk1, vi1) 3(vn1, vk1)
}
block 2 (vn2: i64, vk2: i64, vi2: i64) {
  vh2 = thread.spawn 1 vi2 vi2
  vone2 = i64.const 1
  vin2 = i64.add vi2 vone2
  br 1(vn2, vk2, vin2)
}
block 3 (vn3: i64, vk3: i64) {
  va3 = i64.const 16384
  vc3 = i64.atomic.load va3
  vlt3 = i64.lt_u vc3 vk3
  br_if vlt3 3(vn3, vk3) 4(vn3)
}
block 4 (vn4: i64) {
  vb4 = thread.spawn 2 vn4 vn4
  vz4 = i64.const 0
  br 5(vn4, vb4, vz4)
}
block 5 (vn5: i64, vb5: i32, vi5: i64) {
  vlt5 = i64.lt_u vi5 vn5
  br_if vlt5 6(vn5, vb5, vi5) 8(vb5)
}
block 6 (vn6: i64, vb6: i32, vi6: i64) {
  vt6 = i64.const 16400
  vcur6 = i32.atomic.load vt6
  vone6 = i32.const 1
  veq6 = i32.eq vcur6 vone6
  br_if veq6 7(vn6, vb6, vi6) 9(vn6, vb6, vi6)
}
block 7 (vn7: i64, vb7: i32, vi7: i64) {
  vt7 = i64.const 16400
  vone7 = i32.const 1
  vinf7 = i64.const -1
  vs7 = i32.atomic.wait vt7 vone7 vinf7
  br 6(vn7, vb7, vi7)
}
block 8 (vb8: i32) {
  vst8, vr8 = thread.join vb8
  return vr8
}
block 9 (vn9: i64, vb9: i32, vi9: i64) {
  vt9 = i64.const 16400
  vone9 = i32.const 1
  i32.atomic.store vt9 vone9
  vw9 = atomic.notify vt9 vone9
  vinc9 = i64.const 1
  vin9 = i64.add vi9 vinc9
  br 5(vn9, vb9, vin9)
  }
}
func (i64, i64) -> (i64) {
block 0 (vi: i64, vj: i64) {
  va = i64.const 16384
  vone = i64.const 1
  vold = i64.atomic.rmw.add va vone
  vw = i64.const 16392
  vz = i32.const 0
  vinf = i64.const -1
  vs = i32.atomic.wait vw vz vinf
  vr = i64.const 0
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vn: i64, vm: i64) {
  vz = i64.const 0
  br 1(vn, vz)
}
block 1 (vn1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vn1
  br_if vlt1 2(vn1, vi1) 4(vi1)
}
block 2 (vn2: i64, vi2: i64) {
  vt2 = i64.const 16400
  vcur2 = i32.atomic.load vt2
  vz2 = i32.const 0
  veq2 = i32.eq vcur2 vz2
  br_if veq2 3(vn2, vi2) 5(vn2, vi2)
}
block 3 (vn3: i64, vi3: i64) {
  vt3 = i64.const 16400
  vz3 = i32.const 0
  vinf3 = i64.const -1
  vs3 = i32.atomic.wait vt3 vz3 vinf3
  br 2(vn3, vi3)
}
block 4 (vr4: i64) {
  return vr4
}
block 5 (vn5: i64, vi5: i64) {
  vt5 = i64.const 16400
  vz5 = i32.const 0
  i32.atomic.store vt5 vz5
  vone5 = i32.const 1
  vw5 = atomic.notify vt5 vone5
  vinc5 = i64.const 1
  vin5 = i64.add vi5 vinc5
  br 1(vn5, vin5)
  }
}
"#;

/// [`crate::kernel_spike::run_pairs`]'s program: `func 0 (n, p)` spawns `p` pairs (`func 1` and `func 2`, each taking
/// `n` turns on pair `i`'s word) and joins them.
pub const PAIRS_IR: &str = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (vn: i64, vp: i64) {
  vz = i64.const 0
  br 1(vn, vp, vz)
}
block 1 (vn1: i64, vp1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vp1
  br_if vlt1 2(vn1, vp1, vi1) 3(vp1)
}
block 2 (vn2: i64, vp2: i64, vi2: i64) {
  va2 = thread.spawn 1 vn2 vi2
  vb2 = thread.spawn 2 vn2 vi2
  vbase2 = i64.const 24576
  vsz2 = i64.const 16
  voff2 = i64.mul vi2 vsz2
  vslot2 = i64.add vbase2 voff2
  i32.store vslot2 va2
  veight2 = i64.const 8
  vslotb2 = i64.add vslot2 veight2
  i32.store vslotb2 vb2
  vone2 = i64.const 1
  vin2 = i64.add vi2 vone2
  br 1(vn2, vp2, vin2)
}
block 3 (vp3: i64) {
  vz3 = i64.const 0
  br 4(vp3, vz3)
}
block 4 (vp4: i64, vj4: i64) {
  vlt4 = i64.lt_u vj4 vp4
  br_if vlt4 5(vp4, vj4) 6()
}
block 5 (vp5: i64, vj5: i64) {
  vbase5 = i64.const 24576
  vsz5 = i64.const 16
  voff5 = i64.mul vj5 vsz5
  vslot5 = i64.add vbase5 voff5
  vha5 = i32.load vslot5
  vsa5, vra5 = thread.join vha5
  veight5 = i64.const 8
  vslotb5 = i64.add vslot5 veight5
  vhb5 = i32.load vslotb5
  vsb5, vrb5 = thread.join vhb5
  vone5 = i64.const 1
  vjn5 = i64.add vj5 vone5
  br 4(vp5, vjn5)
}
block 6 () {
  vr6 = i64.const 0
  return vr6
  }
}
func (i64, i64) -> (i64) {
block 0 (vn: i64, vx: i64) {
  vbase = i64.const 16448
  vsz = i64.const 64
  voff = i64.mul vx vsz
  vt = i64.add vbase voff
  vz = i64.const 0
  br 1(vn, vt, vz)
}
block 1 (vn1: i64, vt1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vn1
  br_if vlt1 2(vn1, vt1, vi1) 4(vi1)
}
block 2 (vn2: i64, vt2: i64, vi2: i64) {
  vcur2 = i32.atomic.load vt2
  vone2 = i32.const 1
  veq2 = i32.eq vcur2 vone2
  br_if veq2 3(vn2, vt2, vi2) 5(vn2, vt2, vi2)
}
block 3 (vn3: i64, vt3: i64, vi3: i64) {
  vone3 = i32.const 1
  vinf3 = i64.const -1
  vs3 = i32.atomic.wait vt3 vone3 vinf3
  br 2(vn3, vt3, vi3)
}
block 4 (vr4: i64) {
  return vr4
}
block 5 (vn5: i64, vt5: i64, vi5: i64) {
  vone5 = i32.const 1
  i32.atomic.store vt5 vone5
  vw5 = atomic.notify vt5 vone5
  vinc5 = i64.const 1
  vin5 = i64.add vi5 vinc5
  br 1(vn5, vt5, vin5)
  }
}
func (i64, i64) -> (i64) {
block 0 (vn: i64, vx: i64) {
  vbase = i64.const 16448
  vsz = i64.const 64
  voff = i64.mul vx vsz
  vt = i64.add vbase voff
  vz = i64.const 0
  br 1(vn, vt, vz)
}
block 1 (vn1: i64, vt1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vn1
  br_if vlt1 2(vn1, vt1, vi1) 4(vi1)
}
block 2 (vn2: i64, vt2: i64, vi2: i64) {
  vcur2 = i32.atomic.load vt2
  vz2 = i32.const 0
  veq2 = i32.eq vcur2 vz2
  br_if veq2 3(vn2, vt2, vi2) 5(vn2, vt2, vi2)
}
block 3 (vn3: i64, vt3: i64, vi3: i64) {
  vz3 = i32.const 0
  vinf3 = i64.const -1
  vs3 = i32.atomic.wait vt3 vz3 vinf3
  br 2(vn3, vt3, vi3)
}
block 4 (vr4: i64) {
  return vr4
}
block 5 (vn5: i64, vt5: i64, vi5: i64) {
  vz5 = i32.const 0
  i32.atomic.store vt5 vz5
  vone5 = i32.const 1
  vw5 = atomic.notify vt5 vone5
  vinc5 = i64.const 1
  vin5 = i64.add vi5 vinc5
  br 1(vn5, vt5, vin5)
  }
}
"#;

/// The no-wake probe ([`crate::kernel_spike::run_nowake`]): `func 0 (n, k, mode)` spawns `k` threads
/// parked for good, waits until each has counted itself in, then makes `n` ops that wake nobody:
/// mode 0 a `memory.notify` on a word nobody waits on, mode 1 a `memory.wait` whose word already
/// differs (`WAIT_NOT_EQUAL` at once).
pub const NOWAKE_IR: &str = r#"memory 16
func (i64, i64, i64) -> (i64) {
block 0 (vn: i64, vk: i64, vm: i64) {
  vz = i64.const 0
  br 1(vn, vk, vm, vz)
}
block 1 (vn1: i64, vk1: i64, vm1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vk1
  br_if vlt1 2(vn1, vk1, vm1, vi1) 3(vn1, vk1, vm1)
}
block 2 (vn2: i64, vk2: i64, vm2: i64, vi2: i64) {
  vh2 = thread.spawn 1 vi2 vi2
  vone2 = i64.const 1
  vin2 = i64.add vi2 vone2
  br 1(vn2, vk2, vm2, vin2)
}
block 3 (vn3: i64, vk3: i64, vm3: i64) {
  va3 = i64.const 16384
  vc3 = i64.atomic.load va3
  vlt3 = i64.lt_u vc3 vk3
  br_if vlt3 3(vn3, vk3, vm3) 4(vn3, vm3, vc3)
}
block 4 (vn4: i64, vm4: i64, vz4: i64) {
  vzero4 = i64.const 0
  br 5(vn4, vm4, vzero4, vzero4)
}
block 5 (vn5: i64, vm5: i64, vi5: i64, vacc5: i64) {
  vlt5 = i64.lt_u vi5 vn5
  br_if vlt5 6(vn5, vm5, vi5, vacc5) 9(vacc5)
}
block 6 (vn6: i64, vm6: i64, vi6: i64, vacc6: i64) {
  vzero6 = i64.const 0
  veq6 = i64.eq vm6 vzero6
  br_if veq6 7(vn6, vm6, vi6, vacc6) 8(vn6, vm6, vi6, vacc6)
}
block 7 (vn7: i64, vm7: i64, vi7: i64, vacc7: i64) {
  vx7 = i64.const 16448
  vone7 = i32.const 1
  vw7 = atomic.notify vx7 vone7
  vw647 = i64.extend_i32_u vw7
  vacc27 = i64.add vacc7 vw647
  vinc7 = i64.const 1
  vin7 = i64.add vi7 vinc7
  br 5(vn7, vm7, vin7, vacc27)
}
block 8 (vn8: i64, vm8: i64, vi8: i64, vacc8: i64) {
  vx8 = i64.const 16448
  vone8 = i32.const 1
  vto8 = i64.const -1
  vs8 = i32.atomic.wait vx8 vone8 vto8
  vs648 = i64.extend_i32_u vs8
  vacc28 = i64.add vacc8 vs648
  vinc8 = i64.const 1
  vin8 = i64.add vi8 vinc8
  br 5(vn8, vm8, vin8, vacc28)
}
block 9 (vr9: i64) {
  return vr9
  }
}
func (i64, i64) -> (i64) {
block 0 (vi: i64, vj: i64) {
  va = i64.const 16384
  vone = i64.const 1
  vold = i64.atomic.rmw.add va vone
  vw = i64.const 16392
  vz = i32.const 0
  vinf = i64.const -1
  vs = i32.atomic.wait vw vz vinf
  vr = i64.const 0
  return vr
  }
}
"#;

/// One side of [`always_ir_of`]'s pingpong, as function text: `func (n, w)` takes `n` turns on the turn
/// word, each when the word reads `me`: `w` steps of a dependent multiply-add (the work, kept live
/// through a store), then the word set to `next` and a notify. `head` opens its block 0, which must
/// branch to block 1 with `(vn, vw, vh, vi, vx)`; `end` is its block 7 `(vh9, vx9)`. `fiber`: the
/// side's wait is made in a fiber of its own, function [`ALWAYS_FIBER`], which it `cont.resume.block`s
/// (#3078 slice 3, exit row 2).
fn always_side(me: u32, next: u32, head: &str, end: &str, notify: bool, fiber: bool) -> String {
    let notify = if notify {
        "  vc6 = i32.const 1\n  vwk6 = atomic.notify vt6 vc6\n"
    } else {
        ""
    };
    let wait = if fiber {
        "  vcur643 = i64.extend_i32_u vcur3\n  vf3 = ref.func 2\n  vk3 = cont.new vf3 vcur643\n  vz3 = i64.const 0\n  vs3, vv3 = cont.resume.block vk3 vz3\n"
    } else {
        "  vt3 = i64.const 16400\n  vinf3 = i64.const -1\n  vs3 = i32.atomic.wait vt3 vcur3 vinf3\n"
    };
    format!(
        r#"func (i64, i64) -> (i64) {{
{head}
block 1 (vn1: i64, vw1: i64, vh1: i32, vi1: i64, vx1: i64) {{
  vlt1 = i64.lt_u vi1 vn1
  br_if vlt1 2(vn1, vw1, vh1, vi1, vx1) 7(vh1, vx1)
}}
block 2 (vn2: i64, vw2: i64, vh2: i32, vi2: i64, vx2: i64) {{
  vt2 = i64.const 16400
  vcur2 = i32.atomic.load vt2
  vme2 = i32.const {me}
  veq2 = i32.eq vcur2 vme2
  br_if veq2 4(vn2, vw2, vh2, vi2, vx2, vw2) 3(vn2, vw2, vh2, vi2, vx2, vcur2)
}}
block 3 (vn3: i64, vw3: i64, vh3: i32, vi3: i64, vx3: i64, vcur3: i32) {{
{wait}  br 2(vn3, vw3, vh3, vi3, vx3)
}}
block 4 (vn4: i64, vw4: i64, vh4: i32, vi4: i64, vx4: i64, vk4: i64) {{
  vz4 = i64.const 0
  vdone4 = i64.eq vk4 vz4
  br_if vdone4 6(vn4, vw4, vh4, vi4, vx4) 5(vn4, vw4, vh4, vi4, vx4, vk4)
}}
block 5 (vn5: i64, vw5: i64, vh5: i32, vi5: i64, vx5: i64, vk5: i64) {{
  vm5 = i64.const 6364136223846793005
  vy5 = i64.mul vx5 vm5
  vone5 = i64.const 1
  vx5n = i64.add vy5 vone5
  vk5n = i64.sub vk5 vone5
  br 4(vn5, vw5, vh5, vi5, vx5n, vk5n)
}}
block 6 (vn6: i64, vw6: i64, vh6: i32, vi6: i64, vx6: i64) {{
  vs6 = i64.const 16384
  i64.store vs6 vx6
  vt6 = i64.const 16400
  vnext6 = i32.const {next}
  i32.atomic.store vt6 vnext6
{notify}  vone6 = i64.const 1
  vin6 = i64.add vi6 vone6
  br 1(vn6, vw6, vh6, vin6, vx6)
}}
{end}
}}
"#
    )
}

/// #2985 — the always-block pingpong ([`crate::kernel_spike::run_always`]): `func 0 (n, w)` spawns
/// its partner (`func 1`) and the two take `n` turns each on one word, each turn `w` steps of work
/// before its store and notify. With the work far longer than a sleep takes to reach, the waiter is
/// asleep before every notify: each hand-off blocks.
///
/// #3078 slice 3, exit row 2 — `fiber`: the root's waits are made in a fiber: each turn it makes one
/// that waits on the turn word and `cont.resume.block`s it, so the wake reaches a task idling on a
/// parked fiber.
pub fn always_ir_of(fiber: bool) -> String {
    let root = always_side(
        0,
        1,
        "block 0 (vn: i64, vw: i64) {\n  vh = thread.spawn 1 vn vw\n  vz = i64.const 0\n  br 1(vn, vw, vh, vz, vz)\n}",
        "block 7 (vh9: i32, vx9: i64) {\n  vst9, vr9 = thread.join vh9\n  return vx9\n}",
        true,
        fiber,
    );
    let partner = always_side(
        1,
        0,
        "block 0 (vn: i64, vw: i64) {\n  vh = i32.const 0\n  vz = i64.const 0\n  br 1(vn, vw, vh, vz, vz)\n}",
        "block 7 (vh9: i32, vx9: i64) {\n  return vx9\n}",
        true,
        false,
    );
    let waits = if fiber { ALWAYS_FIBER } else { "" };
    format!("memory 16\n{root}{partner}{waits}")
}

/// #3078 slice 3 — [`always_ir_of`]'s fiber, function 2: it waits on the turn word for it to leave the
/// value its maker passed (`vexp`), with no deadline, and returns.
const ALWAYS_FIBER: &str = "func (i64, i64) -> (i64) {
block 0 (vexp: i64, varg: i64) {
  vt = i64.const 16400
  ve = i32.wrap_i64 vexp
  vinf = i64.const -1
  vs = i32.atomic.wait vt ve vinf
  vr = i64.const 0
  return vr
  }
}
";

/// #2985 — [`always_ir_of`]'s work alone, to take out of its hand-off: `func 0 (n, w)` joins one trivial
/// thread (so every lane runs it as it runs the pingpong), then takes `n` turns of `w` steps each,
/// with the same stores and no notify: what is left of a hand-off once this is taken out is the
/// stops, the park and the wake.
pub fn always_work_ir() -> String {
    let root = always_side(
        0,
        0,
        "block 0 (vn: i64, vw: i64) {\n  vz0 = i64.const 0\n  vh = thread.spawn 1 vz0 vz0\n  vst0, vr0 = thread.join vh\n  vz = i64.const 0\n  br 1(vn, vw, vh, vz, vz)\n}",
        "block 7 (vh9: i32, vx9: i64) {\n  return vx9\n}",
        false,
        false,
    );
    let idle = "func (i64, i64) -> (i64) {\nblock 0 (va: i64, vb: i64) {\n  return va\n  }\n}\n";
    format!("memory 16\n{root}{idle}")
}

/// #3078 phase 2 slice 1 — [`NOWAKE_IR`]'s no-wake ops beside `k` **fibers**: `func 0 (n, k, m, s)`
/// parks `k` fibers of the root, each waiting for good, then makes `n` no-wake ops on word 16448,
/// `m` choosing which as in [`NOWAKE_IR`] (0: a notify with no waiter; 1: a wait that answers
/// not-equal). With `s` 0 the fibers wait on another word (16392); with `s` 1 they wait on 16448 itself
/// and leave before the loop (a notify wakes them all and the root resumes each to its return), the
/// case a fiber wait that never counted out kept on the slow path. Handles go at 20000 + 8i.
pub const NOWAKE_FIBERS_IR: &str = r#"memory 16
func (i64, i64, i64, i64) -> (i64) {
block 0 (vn: i64, vk: i64, vm: i64, vs: i64) {
  vz = i64.const 0
  br 1(vn, vk, vm, vs, vz)
}
block 1 (vn1: i64, vk1: i64, vm1: i64, vs1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vk1
  br_if vlt1 2(vn1, vk1, vm1, vs1, vi1) 3(vn1, vk1, vm1, vs1)
}
block 2 (vn2: i64, vk2: i64, vm2: i64, vs2: i64, vi2: i64) {
  vb2 = i64.const 16392
  vd2 = i64.const 56
  vo2 = i64.mul vs2 vd2
  vw2 = i64.add vb2 vo2
  vf2 = ref.func 1
  vc2 = cont.new vf2 vw2
  vz2 = i64.const 0
  vr2, vv2 = cont.resume vc2 vz2
  vh2 = i64.const 20000
  ve2 = i64.const 8
  vx2 = i64.mul vi2 ve2
  va2 = i64.add vh2 vx2
  i64.store va2 vc2
  vone2 = i64.const 1
  vin2 = i64.add vi2 vone2
  br 1(vn2, vk2, vm2, vs2, vin2)
}
block 3 (vn3: i64, vk3: i64, vm3: i64, vs3: i64) {
  vz3 = i64.const 0
  vne3 = i64.ne vs3 vz3
  br_if vne3 4(vn3, vk3, vm3) 7(vn3, vm3)
}
block 4 (vn4: i64, vk4: i64, vm4: i64) {
  vx4 = i64.const 16448
  vk324 = i32.wrap_i64 vk4
  vw4 = atomic.notify vx4 vk324
  vz4 = i64.const 0
  br 5(vn4, vk4, vm4, vz4)
}
block 5 (vn5: i64, vk5: i64, vm5: i64, vi5: i64) {
  vlt5 = i64.lt_u vi5 vk5
  br_if vlt5 6(vn5, vk5, vm5, vi5) 7(vn5, vm5)
}
block 6 (vn6: i64, vk6: i64, vm6: i64, vi6: i64) {
  vh6 = i64.const 20000
  ve6 = i64.const 8
  vx6 = i64.mul vi6 ve6
  va6 = i64.add vh6 vx6
  vc6 = i64.load va6
  vz6 = i64.const 0
  vr6, vv6 = cont.resume vc6 vz6
  vone6 = i64.const 1
  vin6 = i64.add vi6 vone6
  br 5(vn6, vk6, vm6, vin6)
}
block 7 (vn7: i64, vm7: i64) {
  vzero7 = i64.const 0
  br 8(vn7, vm7, vzero7, vzero7)
}
block 8 (vn8: i64, vm8: i64, vi8: i64, vacc8: i64) {
  vlt8 = i64.lt_u vi8 vn8
  br_if vlt8 9(vn8, vm8, vi8, vacc8) 12(vacc8)
}
block 9 (vn9: i64, vm9: i64, vi9: i64, vacc9: i64) {
  vzero9 = i64.const 0
  veq9 = i64.eq vm9 vzero9
  br_if veq9 10(vn9, vm9, vi9, vacc9) 11(vn9, vm9, vi9, vacc9)
}
block 10 (vn10: i64, vm10: i64, vi10: i64, vacc10: i64) {
  vx10 = i64.const 16448
  vone10 = i32.const 1
  vw10 = atomic.notify vx10 vone10
  vw6410 = i64.extend_i32_u vw10
  vacc210 = i64.add vacc10 vw6410
  vinc10 = i64.const 1
  vin10 = i64.add vi10 vinc10
  br 8(vn10, vm10, vin10, vacc210)
}
block 11 (vn11: i64, vm11: i64, vi11: i64, vacc11: i64) {
  vx11 = i64.const 16448
  vone11 = i32.const 1
  vto11 = i64.const -1
  vs11 = i32.atomic.wait vx11 vone11 vto11
  vs6411 = i64.extend_i32_u vs11
  vacc211 = i64.add vacc11 vs6411
  vinc11 = i64.const 1
  vin11 = i64.add vi11 vinc11
  br 8(vn11, vm11, vin11, vacc211)
}
block 12 (vr12: i64) {
  return vr12
  }
}
func (i64, i64) -> (i64) {
block 0 (vw: i64, varg: i64) {
  vz = i32.const 0
  vinf = i64.const -1
  vs = i32.atomic.wait vw vz vinf
  vr = i64.extend_i32_s vs
  return vr
  }
}
"#;

/// The root [`FIBER_ROUNDTRIP_IR`] and [`POLL_PARKED_IR`] share: `func 0 (n, w)` spawns `w` threads
/// of function 1, the `i`th with `(n, i)`, and joins them all, answering the sum of their values.
macro_rules! w_threads_root {
    () => {
        r#"func (i64, i64) -> (i64) {
block 0 (vn: i64, vw: i64) {
  vz = i64.const 0
  br 1(vn, vw, vz)
}
block 1 (vn1: i64, vw1: i64, vi1: i64) {
  vmore1 = i64.lt_u vi1 vw1
  br_if vmore1 2(vn1, vw1, vi1) 3(vw1)
}
block 2 (vn2: i64, vw2: i64, vi2: i64) {
  vz2 = i64.const 0
  vh2 = thread.spawn 1 vn2 vi2
  veight2 = i64.const 8
  voff2 = i64.mul vi2 veight2
  vbase2 = i64.const 16384
  vat2 = i64.add vbase2 voff2
  vh642 = i64.extend_i32_s vh2
  i64.store vat2 vh642
  vone2 = i64.const 1
  vnext2 = i64.add vi2 vone2
  br 1(vn2, vw2, vnext2)
}
block 3 (vw3: i64) {
  vz3 = i64.const 0
  br 4(vw3, vz3, vz3)
}
block 4 (vw4: i64, vj4: i64, vacc4: i64) {
  vmore4 = i64.lt_u vj4 vw4
  br_if vmore4 5(vw4, vj4, vacc4) 6(vacc4)
}
block 5 (vw5: i64, vj5: i64, vacc5: i64) {
  veight5 = i64.const 8
  voff5 = i64.mul vj5 veight5
  vbase5 = i64.const 16384
  vat5 = i64.add vbase5 voff5
  vh5 = i64.load vat5
  vh325 = i32.wrap_i64 vh5
  vst5, vv5 = thread.join vh325
  vacc25 = i64.add vacc5 vv5
  vone5 = i64.const 1
  vnext5 = i64.add vj5 vone5
  br 4(vw5, vnext5, vacc25)
}
block 6 (vr6: i64) {
  return vr6
  }
}
"#
    };
}

/// #3078 slice 2 — exit row 1: `w` threads, each resuming a fiber of its own `n` times, the fiber
/// `suspend`ing each time; the root joins them all. Entry `(n, w)`; answers `w * n`. No fiber parks,
/// so the kernel route admits it unmetered.
pub const FIBER_ROUNDTRIP_IR: &str = concat!(
    "memory 16\n",
    w_threads_root!(),
    r#"func (i64, i64) -> (i64) {
block 0 (vn: i64, vb: i64) {
  vf = ref.func 2
  vz = i64.const 0
  vk = cont.new vf vz
  br 1(vn, vk, vz)
}
block 1 (vn1: i64, vk1: i64, vi1: i64) {
  vmore1 = i64.lt_u vi1 vn1
  br_if vmore1 2(vn1, vk1, vi1) 3(vn1)
}
block 2 (vn2: i64, vk2: i64, vi2: i64) {
  vz2 = i64.const 0
  vs2, vv2 = cont.resume vk2 vz2
  vone2 = i64.const 1
  vnext2 = i64.add vi2 vone2
  br 1(vn2, vk2, vnext2)
}
block 3 (vr3: i64) {
  return vr3
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  br 1()
}
block 1 () {
  vz = i64.const 0
  vx = suspend vz
  br 1()
  }
}
"#
);

/// #3078 slice 3 (review Q2) — many fibers round-robin, `mn_sched`-shaped: each of `w` threads makes
/// 512 fibers that suspend in a loop, keeps their handles in memory, and resumes them in turn, `n`
/// resumes in all, so every resume reaches a slot (and its parked word) the last 511 did not. Entry
/// `(n, w)`; answers 0.
pub const ROUND_ROBIN_IR: &str = concat!(
    "memory 16\n",
    w_threads_root!(),
    r#"func (i64, i64) -> (i64) {
block 0 (vn: i64, vi: i64) {
  vf = ref.func 2
  vk4096 = i64.const 4096
  voff = i64.mul vi vk4096
  vbase0 = i64.const 32768
  vbase = i64.add vbase0 voff
  vz = i64.const 0
  br 1(vn, vbase, vz)
}
block 1 (vn1: i64, vb1: i64, vj1: i64) {
  vfibers1 = i64.const 512
  vmore1 = i64.lt_u vj1 vfibers1
  br_if vmore1 2(vn1, vb1, vj1) 3(vn1, vb1)
}
block 2 (vn2: i64, vb2: i64, vj2: i64) {
  vf2 = ref.func 2
  vz2 = i64.const 0
  vk2 = cont.new vf2 vz2
  veight2 = i64.const 8
  vo2 = i64.mul vj2 veight2
  vat2 = i64.add vb2 vo2
  i64.store vat2 vk2
  vone2 = i64.const 1
  vnext2 = i64.add vj2 vone2
  br 1(vn2, vb2, vnext2)
}
block 3 (vn3: i64, vb3: i64) {
  vz3 = i64.const 0
  br 4(vn3, vb3, vz3, vz3)
}
block 4 (vn4: i64, vb4: i64, vi4: i64, vx4: i64) {
  vmore4 = i64.lt_u vi4 vn4
  br_if vmore4 5(vn4, vb4, vi4, vx4) 6()
}
block 5 (vn5: i64, vb5: i64, vi5: i64, vx5: i64) {
  veight5 = i64.const 8
  vo5 = i64.mul vx5 veight5
  vat5 = i64.add vb5 vo5
  vk5 = i64.load vat5
  vz5 = i64.const 0
  vs5, vv5 = cont.resume vk5 vz5
  vone5 = i64.const 1
  vnext5 = i64.add vi5 vone5
  vx15 = i64.add vx5 vone5
  vmask5 = i64.const 511
  vnx5 = i64.and vx15 vmask5
  br 4(vn5, vb5, vnext5, vnx5)
}
block 6 () {
  vz6 = i64.const 0
  return vz6
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  br 1()
}
block 1 () {
  vz = i64.const 0
  vx = suspend vz
  br 1()
  }
}
"#
);

/// #3078 slice 2 — exit row 4: a thread makes `k` fibers and leaves each suspended, then makes `n`
/// `memory.notify`s on a word nothing waits on; the root joins it. Entry `(n, k)`; answers 0.
pub const NOWAKE_SUSPENDED_IR: &str = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (vn: i64, vk: i64) {
  vh = thread.spawn 1 vn vk
  vst, vv = thread.join vh
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (vn: i64, vk: i64) {
  vz = i64.const 0
  br 1(vn, vk, vz)
}
block 1 (vn1: i64, vk1: i64, vi1: i64) {
  vmore1 = i64.lt_u vi1 vk1
  br_if vmore1 2(vn1, vk1, vi1) 3(vn1)
}
block 2 (vn2: i64, vk2: i64, vi2: i64) {
  vf2 = ref.func 2
  vz2 = i64.const 0
  vh2 = cont.new vf2 vz2
  vs2, vv2 = cont.resume vh2 vz2
  vone2 = i64.const 1
  vnext2 = i64.add vi2 vone2
  br 1(vn2, vk2, vnext2)
}
block 3 (vn3: i64) {
  vz3 = i64.const 0
  br 4(vn3, vz3, vz3)
}
block 4 (vn4: i64, vi4: i64, vacc4: i64) {
  vmore4 = i64.lt_u vi4 vn4
  br_if vmore4 5(vn4, vi4, vacc4) 6(vacc4)
}
block 5 (vn5: i64, vi5: i64, vacc5: i64) {
  vq5 = i64.const 16400
  vc5 = i32.const 1
  vw5 = atomic.notify vq5 vc5
  vw645 = i64.extend_i32_u vw5
  vacc25 = i64.add vacc5 vw645
  vone5 = i64.const 1
  vnext5 = i64.add vi5 vone5
  br 4(vn5, vnext5, vacc25)
}
block 6 (vr6: i64) {
  return vr6
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz = i64.const 0
  vx = suspend vz
  return vz
  }
}
"#;

/// #3078 slice 3 — exit row 3: `w` threads (entry `(n, w)`, [`w_threads_root`]), the `i`th of which
/// makes a fiber that waits for good on a word of its own (32768 + 64i, a line apart), resumes it once
/// so it parks, then polls it `n` times with `cont.resume`: each poll of a still-parked fiber.
pub const POLL_PARKED_IR: &str = concat!(
    "memory 16\n",
    w_threads_root!(),
    r#"func (i64, i64) -> (i64) {
block 0 (vn: i64, vi: i64) {
  vbase = i64.const 32768
  vsz = i64.const 64
  voff = i64.mul vi vsz
  vat = i64.add vbase voff
  vf = ref.func 2
  vk = cont.new vf vat
  vz = i64.const 0
  vs0, vv0 = cont.resume vk vz
  br 1(vn, vk, vz)
}
block 1 (vn1: i64, vk1: i64, vi1: i64) {
  vmore1 = i64.lt_u vi1 vn1
  br_if vmore1 2(vn1, vk1, vi1) 3(vn1)
}
block 2 (vn2: i64, vk2: i64, vi2: i64) {
  vz2 = i64.const 0
  vs2, vv2 = cont.resume vk2 vz2
  vone2 = i64.const 1
  vnext2 = i64.add vi2 vone2
  br 1(vn2, vk2, vnext2)
}
block 3 (vr3: i64) {
  return vr3
  }
}
func (i64, i64) -> (i64) {
block 0 (vat: i64, varg: i64) {
  vz = i32.const 0
  vinf = i64.const -1
  vw = i32.atomic.wait vat vz vinf
  vr = i64.const 0
  return vr
  }
}
"#
);

/// #3078 slice 3 — exit row 4 on the kernel route: [`NOWAKE_FIBERS_IR`]'s program run by one spawned
/// thread, since the kernel keeps a run with no spawn off its route (`fiber only`). Its entry is the
/// returned function, `(n, k, m, s)`: it stores `m` and `s` at 24000 and 24008, spawns a thread that
/// reads them and calls function 0 with `(n, k, m, s)`, and joins it.
pub fn nowake_fibers_in_a_thread_ir() -> (String, u32) {
    let n = NOWAKE_FIBERS_IR.matches("\nfunc ").count() as u32;
    let ir = format!(
        "{NOWAKE_FIBERS_IR}func (i64, i64) -> (i64) {{
block 0 (vn: i64, vk: i64) {{
  vam = i64.const 24000
  vm = i64.load vam
  vas = i64.const 24008
  vs = i64.load vas
  vr = call 0(vn, vk, vm, vs)
  return vr
  }}
}}
func (i64, i64, i64, i64) -> (i64) {{
block 0 (vn: i64, vk: i64, vm: i64, vs: i64) {{
  vam = i64.const 24000
  i64.store vam vm
  vas = i64.const 24008
  i64.store vas vs
  vt = thread.spawn {n} vn vk
  vst, vv = thread.join vt
  return vv
  }}
}}
"
    );
    (ir, n + 1)
}
