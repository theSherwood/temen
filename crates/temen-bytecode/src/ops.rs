//! Pure scalar, float, conversion, and SIMD value semantics matching `temen-interp` bit-for-bit.

use temen_ir::{
    AtomicRmwOp, BinOp, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, IToF, IntTy,
    IntUnOp, VBitBinOp, VFloatBinOp, VIntBinOp, VShape,
};
use temen_kernel::Trap;
use temen_mem::RmwOp;

/// One 128-bit register-file slot (holds `i32`, `i64`, `f32`, `f64`, `ref`, `cap`, or `v128`).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Reg(pub u128);

impl Reg {
    #[inline]
    pub const fn from_i32(v: i32) -> Reg {
        Reg(v as u32 as u128)
    }
    #[inline]
    pub const fn from_i64(v: i64) -> Reg {
        Reg(v as u64 as u128)
    }
    #[inline]
    pub fn from_f32(v: f32) -> Reg {
        Reg(v.to_bits() as u128)
    }
    #[inline]
    pub fn from_f64(v: f64) -> Reg {
        Reg(v.to_bits() as u128)
    }
    #[inline]
    pub const fn from_f32_bits(bits: u32) -> Reg {
        Reg(bits as u128)
    }
    #[inline]
    pub const fn from_f64_bits(bits: u64) -> Reg {
        Reg(bits as u128)
    }
    #[inline]
    pub const fn from_v128(bytes: [u8; 16]) -> Reg {
        Reg(u128::from_le_bytes(bytes))
    }

    #[inline]
    pub const fn i32(self) -> i32 {
        self.0 as u32 as i32
    }
    #[inline]
    pub const fn i64(self) -> i64 {
        self.0 as u64 as i64
    }
    #[inline]
    pub const fn u64(self) -> u64 {
        self.0 as u64
    }
    #[inline]
    pub fn f32(self) -> f32 {
        f32::from_bits(self.0 as u32)
    }
    #[inline]
    pub fn f64(self) -> f64 {
        f64::from_bits(self.0 as u64)
    }
    #[inline]
    pub const fn v128(self) -> [u8; 16] {
        self.0.to_le_bytes()
    }
}

#[inline]
fn check_div(by_zero: bool, overflow: bool) -> Result<(), Trap> {
    if by_zero {
        Err(Trap::DivByZero)
    } else if overflow {
        Err(Trap::IntOverflow)
    } else {
        Ok(())
    }
}

#[inline]
pub fn bin32(op: BinOp, a: i32, b: i32) -> Result<i32, Trap> {
    Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::DivS => {
            check_div(b == 0, a == i32::MIN && b == -1)?;
            a.wrapping_div(b)
        }
        BinOp::DivU => {
            check_div(b == 0, false)?;
            ((a as u32) / (b as u32)) as i32
        }
        BinOp::RemS => {
            check_div(b == 0, false)?;
            a.wrapping_rem(b)
        }
        BinOp::RemU => {
            check_div(b == 0, false)?;
            ((a as u32) % (b as u32)) as i32
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::ShrS => a.wrapping_shr(b as u32),
        BinOp::ShrU => ((a as u32).wrapping_shr(b as u32)) as i32,
        BinOp::Rotl => a.rotate_left(b as u32),
        BinOp::Rotr => a.rotate_right(b as u32),
    })
}

#[inline]
pub fn bin64(op: BinOp, a: i64, b: i64) -> Result<i64, Trap> {
    Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::DivS => {
            check_div(b == 0, a == i64::MIN && b == -1)?;
            a.wrapping_div(b)
        }
        BinOp::DivU => {
            check_div(b == 0, false)?;
            ((a as u64) / (b as u64)) as i64
        }
        BinOp::RemS => {
            check_div(b == 0, false)?;
            a.wrapping_rem(b)
        }
        BinOp::RemU => {
            check_div(b == 0, false)?;
            ((a as u64) % (b as u64)) as i64
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::ShrS => a.wrapping_shr(b as u32),
        BinOp::ShrU => ((a as u64).wrapping_shr(b as u32)) as i64,
        BinOp::Rotl => a.rotate_left(b as u32),
        BinOp::Rotr => a.rotate_right(b as u32),
    })
}

#[inline]
pub fn cmp32(op: CmpOp, a: i32, b: i32) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::LtS => a < b,
        CmpOp::LtU => (a as u32) < (b as u32),
        CmpOp::LeS => a <= b,
        CmpOp::LeU => (a as u32) <= (b as u32),
        CmpOp::GtS => a > b,
        CmpOp::GtU => (a as u32) > (b as u32),
        CmpOp::GeS => a >= b,
        CmpOp::GeU => (a as u32) >= (b as u32),
    }
}

#[inline]
pub fn cmp64(op: CmpOp, a: i64, b: i64) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::LtS => a < b,
        CmpOp::LtU => (a as u64) < (b as u64),
        CmpOp::LeS => a <= b,
        CmpOp::LeU => (a as u64) <= (b as u64),
        CmpOp::GtS => a > b,
        CmpOp::GtU => (a as u64) > (b as u64),
        CmpOp::GeS => a >= b,
        CmpOp::GeU => (a as u64) >= (b as u64),
    }
}

#[inline]
pub fn intun32(op: IntUnOp, a: i32) -> i32 {
    match op {
        IntUnOp::Clz => (a as u32).leading_zeros() as i32,
        IntUnOp::Ctz => (a as u32).trailing_zeros() as i32,
        IntUnOp::Popcnt => (a as u32).count_ones() as i32,
        IntUnOp::Extend8S => (a as i8) as i32,
        IntUnOp::Extend16S => (a as i16) as i32,
        IntUnOp::Extend32S => a,
    }
}

#[inline]
pub fn intun64(op: IntUnOp, a: i64) -> i64 {
    match op {
        IntUnOp::Clz => (a as u64).leading_zeros() as i64,
        IntUnOp::Ctz => (a as u64).trailing_zeros() as i64,
        IntUnOp::Popcnt => (a as u64).count_ones() as i64,
        IntUnOp::Extend8S => (a as i8) as i64,
        IntUnOp::Extend16S => (a as i16) as i64,
        IntUnOp::Extend32S => (a as i32) as i64,
    }
}

#[inline]
pub fn convert(op: ConvOp, a: Reg) -> Reg {
    match op {
        ConvOp::WrapI64 => Reg::from_i32(a.i64() as i32),
        ConvOp::ExtendI32S => Reg::from_i64(a.i32() as i64),
        ConvOp::ExtendI32U => Reg::from_i64(a.i32() as u32 as i64),
    }
}

#[inline]
fn fmin32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a < b {
        a
    } else {
        b
    }
}

#[inline]
fn fmax32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}

#[inline]
fn fmin64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a < b {
        a
    } else {
        b
    }
}

#[inline]
fn fmax64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}

#[inline]
pub fn fbin32(op: FBinOp, a: f32, b: f32) -> f32 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin32(a, b),
        FBinOp::Max => fmax32(a, b),
        FBinOp::Copysign => a.copysign(b),
    }
}

#[inline]
pub fn fbin64(op: FBinOp, a: f64, b: f64) -> f64 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin64(a, b),
        FBinOp::Max => fmax64(a, b),
        FBinOp::Copysign => a.copysign(b),
    }
}

#[inline]
pub fn fun32(op: FUnOp, a: f32) -> f32 {
    match op {
        FUnOp::Abs => a.abs(),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => a.sqrt(),
        FUnOp::Ceil => a.ceil(),
        FUnOp::Floor => a.floor(),
        FUnOp::Trunc => a.trunc(),
        FUnOp::Nearest => a.round_ties_even(),
    }
}

#[inline]
pub fn fun64(op: FUnOp, a: f64) -> f64 {
    match op {
        FUnOp::Abs => a.abs(),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => a.sqrt(),
        FUnOp::Ceil => a.ceil(),
        FUnOp::Floor => a.floor(),
        FUnOp::Trunc => a.trunc(),
        FUnOp::Nearest => a.round_ties_even(),
    }
}

#[inline]
pub fn fcmp32(op: FCmpOp, a: f32, b: f32) -> bool {
    match op {
        FCmpOp::Eq => a == b,
        FCmpOp::Ne => a != b,
        FCmpOp::Lt => a < b,
        FCmpOp::Le => a <= b,
        FCmpOp::Gt => a > b,
        FCmpOp::Ge => a >= b,
    }
}

#[inline]
pub fn fcmp64(op: FCmpOp, a: f64, b: f64) -> bool {
    match op {
        FCmpOp::Eq => a == b,
        FCmpOp::Ne => a != b,
        FCmpOp::Lt => a < b,
        FCmpOp::Le => a <= b,
        FCmpOp::Gt => a > b,
        FCmpOp::Ge => a >= b,
    }
}

#[inline]
pub fn fto_i_sat(op: FToI, a: Reg) -> Reg {
    match op {
        FToI::F32I32S => Reg::from_i32(a.f32() as i32),
        FToI::F32I32U => Reg::from_i32(a.f32() as u32 as i32),
        FToI::F32I64S => Reg::from_i64(a.f32() as i64),
        FToI::F32I64U => Reg::from_i64(a.f32() as u64 as i64),
        FToI::F64I32S => Reg::from_i32(a.f64() as i32),
        FToI::F64I32U => Reg::from_i32(a.f64() as u32 as i32),
        FToI::F64I64S => Reg::from_i64(a.f64() as i64),
        FToI::F64I64U => Reg::from_i64(a.f64() as u64 as i64),
    }
}

#[inline]
pub fn fto_i_trap(op: FToI, a: Reg) -> Result<Reg, Trap> {
    let (from, to, signed) = op.parts();
    let f: f64 = match from {
        FloatTy::F32 => a.f32() as f64,
        FloatTy::F64 => a.f64(),
    };
    if f.is_nan() {
        return Err(Trap::BadConversion);
    }
    #[allow(clippy::manual_range_contains)]
    let in_range = match (to, signed) {
        (IntTy::I32, true) => f > -2_147_483_649.0 && f < 2_147_483_648.0,
        (IntTy::I32, false) => f > -1.0 && f < 4_294_967_296.0,
        (IntTy::I64, true) => f >= -9_223_372_036_854_775_808.0 && f < 9_223_372_036_854_775_808.0,
        (IntTy::I64, false) => f > -1.0 && f < 18_446_744_073_709_551_616.0,
    };
    if !in_range {
        return Err(Trap::BadConversion);
    }
    Ok(match (to, signed) {
        (IntTy::I32, true) => Reg::from_i32(f as i32),
        (IntTy::I32, false) => Reg::from_i32(f as u32 as i32),
        (IntTy::I64, true) => Reg::from_i64(f as i64),
        (IntTy::I64, false) => Reg::from_i64(f as u64 as i64),
    })
}

#[inline]
pub fn i_to_f(op: IToF, a: Reg) -> Reg {
    match op {
        IToF::I32F32S => Reg::from_f32(a.i32() as f32),
        IToF::I32F32U => Reg::from_f32(a.i32() as u32 as f32),
        IToF::I64F32S => Reg::from_f32(a.i64() as f32),
        IToF::I64F32U => Reg::from_f32(a.i64() as u64 as f32),
        IToF::I32F64S => Reg::from_f64(a.i32() as f64),
        IToF::I32F64U => Reg::from_f64(a.i32() as u32 as f64),
        IToF::I64F64S => Reg::from_f64(a.i64() as f64),
        IToF::I64F64U => Reg::from_f64(a.i64() as u64 as f64),
    }
}

#[inline]
pub fn cast(op: CastOp, a: Reg) -> Reg {
    match op {
        CastOp::Demote => Reg::from_f32(a.f64() as f32),
        CastOp::Promote => Reg::from_f64(a.f32() as f64),
        CastOp::ReinterpI32F32 => Reg::from_f32(f32::from_bits(a.i32() as u32)),
        CastOp::ReinterpF32I32 => Reg::from_i32(a.f32().to_bits() as i32),
        CastOp::ReinterpI64F64 => Reg::from_f64(f64::from_bits(a.i64() as u64)),
        CastOp::ReinterpF64I64 => Reg::from_i64(a.f64().to_bits() as i64),
    }
}

#[inline]
pub fn rmw_op(op: AtomicRmwOp) -> RmwOp {
    match op {
        AtomicRmwOp::Add => RmwOp::Add,
        AtomicRmwOp::Sub => RmwOp::Sub,
        AtomicRmwOp::And => RmwOp::And,
        AtomicRmwOp::Or => RmwOp::Or,
        AtomicRmwOp::Xor => RmwOp::Xor,
        AtomicRmwOp::Xchg => RmwOp::Xchg,
    }
}

// ---- Core SIMD helpers -------------------------------------------------------------------------

pub fn splat(shape: VShape, a: Reg) -> Reg {
    let mut out = [0u8; 16];
    match shape {
        VShape::I8x16 => out.fill(a.i32() as u8),
        VShape::I16x8 => {
            let b = (a.i32() as u16).to_le_bytes();
            for chunk in out.chunks_exact_mut(2) {
                chunk.copy_from_slice(&b);
            }
        }
        VShape::I32x4 | VShape::F32x4 => {
            let b = (a.0 as u32).to_le_bytes();
            for chunk in out.chunks_exact_mut(4) {
                chunk.copy_from_slice(&b);
            }
        }
        VShape::I64x2 | VShape::F64x2 => {
            let b = (a.0 as u64).to_le_bytes();
            for chunk in out.chunks_exact_mut(8) {
                chunk.copy_from_slice(&b);
            }
        }
    }
    Reg::from_v128(out)
}

pub fn extract_lane(shape: VShape, lane: u8, signed: bool, a: Reg) -> Reg {
    let b = a.v128();
    let i = lane as usize;
    match shape {
        VShape::I8x16 => {
            let v = b[i];
            Reg::from_i32(if signed { v as i8 as i32 } else { v as i32 })
        }
        VShape::I16x8 => {
            let v = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
            Reg::from_i32(if signed { v as i16 as i32 } else { v as i32 })
        }
        VShape::I32x4 | VShape::F32x4 => {
            let v = u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
            Reg(v as u128)
        }
        VShape::I64x2 | VShape::F64x2 => {
            let v = u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
            Reg(v as u128)
        }
    }
}

pub fn replace_lane(shape: VShape, lane: u8, a: Reg, val: Reg) -> Reg {
    let mut b = a.v128();
    let i = lane as usize;
    match shape {
        VShape::I8x16 => b[i] = val.i32() as u8,
        VShape::I16x8 => b[2 * i..2 * i + 2].copy_from_slice(&(val.i32() as u16).to_le_bytes()),
        VShape::I32x4 | VShape::F32x4 => {
            b[4 * i..4 * i + 4].copy_from_slice(&(val.0 as u32).to_le_bytes())
        }
        VShape::I64x2 | VShape::F64x2 => {
            b[8 * i..8 * i + 8].copy_from_slice(&(val.0 as u64).to_le_bytes())
        }
    }
    Reg::from_v128(b)
}

pub fn vbit_bin(op: VBitBinOp, a: Reg, b: Reg) -> Reg {
    Reg(match op {
        VBitBinOp::And => a.0 & b.0,
        VBitBinOp::AndNot => a.0 & !b.0,
        VBitBinOp::Or => a.0 | b.0,
        VBitBinOp::Xor => a.0 ^ b.0,
    })
}

pub fn vshuffle(lanes: &[u8; 16], a: Reg, b: Reg) -> Reg {
    let ab_a = a.v128();
    let ab_b = b.v128();
    let mut out = [0u8; 16];
    for (i, &sel) in lanes.iter().enumerate() {
        out[i] = if sel < 16 {
            ab_a[sel as usize]
        } else {
            ab_b[(sel - 16) as usize]
        };
    }
    Reg::from_v128(out)
}

pub fn vswizzle(a: Reg, b: Reg) -> Reg {
    let ba = a.v128();
    let bb = b.v128();
    let mut out = [0u8; 16];
    for i in 0..16 {
        let sel = bb[i] as usize;
        out[i] = if sel < 16 { ba[sel] } else { 0 };
    }
    Reg::from_v128(out)
}

pub fn vint_bin(shape: VShape, op: VIntBinOp, a: Reg, b: Reg) -> Option<Reg> {
    let ba = a.v128();
    let bb = b.v128();
    let mut out = [0u8; 16];
    match shape {
        VShape::I32x4 => {
            for i in 0..4 {
                let x = i32::from_le_bytes(ba[4 * i..4 * i + 4].try_into().unwrap());
                let y = i32::from_le_bytes(bb[4 * i..4 * i + 4].try_into().unwrap());
                let r = match op {
                    VIntBinOp::Add => x.wrapping_add(y),
                    VIntBinOp::Sub => x.wrapping_sub(y),
                    VIntBinOp::Mul => x.wrapping_mul(y),
                    VIntBinOp::MinS => x.min(y),
                    VIntBinOp::MinU => (x as u32).min(y as u32) as i32,
                    VIntBinOp::MaxS => x.max(y),
                    VIntBinOp::MaxU => (x as u32).max(y as u32) as i32,
                };
                out[4 * i..4 * i + 4].copy_from_slice(&r.to_le_bytes());
            }
            Some(Reg::from_v128(out))
        }
        VShape::I64x2 => {
            for i in 0..2 {
                let x = i64::from_le_bytes(ba[8 * i..8 * i + 8].try_into().unwrap());
                let y = i64::from_le_bytes(bb[8 * i..8 * i + 8].try_into().unwrap());
                let r = match op {
                    VIntBinOp::Add => x.wrapping_add(y),
                    VIntBinOp::Sub => x.wrapping_sub(y),
                    VIntBinOp::Mul => x.wrapping_mul(y),
                    VIntBinOp::MinS => x.min(y),
                    VIntBinOp::MinU => (x as u64).min(y as u64) as i64,
                    VIntBinOp::MaxS => x.max(y),
                    VIntBinOp::MaxU => (x as u64).max(y as u64) as i64,
                };
                out[8 * i..8 * i + 8].copy_from_slice(&r.to_le_bytes());
            }
            Some(Reg::from_v128(out))
        }
        VShape::I8x16 => {
            for i in 0..16 {
                let x = ba[i] as i8;
                let y = bb[i] as i8;
                let r = match op {
                    VIntBinOp::Add => x.wrapping_add(y) as u8,
                    VIntBinOp::Sub => x.wrapping_sub(y) as u8,
                    VIntBinOp::Mul => x.wrapping_mul(y) as u8,
                    VIntBinOp::MinS => x.min(y) as u8,
                    VIntBinOp::MinU => ba[i].min(bb[i]),
                    VIntBinOp::MaxS => x.max(y) as u8,
                    VIntBinOp::MaxU => ba[i].max(bb[i]),
                };
                out[i] = r;
            }
            Some(Reg::from_v128(out))
        }
        VShape::I16x8 => {
            for i in 0..8 {
                let x = i16::from_le_bytes([ba[2 * i], ba[2 * i + 1]]);
                let y = i16::from_le_bytes([bb[2 * i], bb[2 * i + 1]]);
                let r = match op {
                    VIntBinOp::Add => x.wrapping_add(y),
                    VIntBinOp::Sub => x.wrapping_sub(y),
                    VIntBinOp::Mul => x.wrapping_mul(y),
                    VIntBinOp::MinS => x.min(y),
                    VIntBinOp::MinU => (x as u16).min(y as u16) as i16,
                    VIntBinOp::MaxS => x.max(y),
                    VIntBinOp::MaxU => (x as u16).max(y as u16) as i16,
                };
                out[2 * i..2 * i + 2].copy_from_slice(&r.to_le_bytes());
            }
            Some(Reg::from_v128(out))
        }
        VShape::F32x4 | VShape::F64x2 => None,
    }
}

pub fn vfloat_bin(shape: VShape, op: VFloatBinOp, a: Reg, b: Reg) -> Option<Reg> {
    let ba = a.v128();
    let bb = b.v128();
    let mut out = [0u8; 16];
    let fop = match op {
        VFloatBinOp::Add => FBinOp::Add,
        VFloatBinOp::Sub => FBinOp::Sub,
        VFloatBinOp::Mul => FBinOp::Mul,
        VFloatBinOp::Div => FBinOp::Div,
        VFloatBinOp::Min => FBinOp::Min,
        VFloatBinOp::Max => FBinOp::Max,
    };
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_le_bytes(ba[4 * i..4 * i + 4].try_into().unwrap());
                let y = f32::from_le_bytes(bb[4 * i..4 * i + 4].try_into().unwrap());
                out[4 * i..4 * i + 4].copy_from_slice(&fbin32(fop, x, y).to_le_bytes());
            }
            Some(Reg::from_v128(out))
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_le_bytes(ba[8 * i..8 * i + 8].try_into().unwrap());
                let y = f64::from_le_bytes(bb[8 * i..8 * i + 8].try_into().unwrap());
                out[8 * i..8 * i + 8].copy_from_slice(&fbin64(fop, x, y).to_le_bytes());
            }
            Some(Reg::from_v128(out))
        }
        _ => None,
    }
}
