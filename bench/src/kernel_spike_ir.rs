//! #2948 spike — the park-and-wake probes' programs, shared by `--kernel-spike` and `spike_mem`.

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

/// [`run_pairs`]'s program: `func 0 (n, p)` spawns `p` pairs (`func 1` and `func 2`, each taking
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
