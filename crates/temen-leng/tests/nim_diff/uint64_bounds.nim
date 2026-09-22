# 64-bit unsigned bounds and comparisons against them (#1612).
#
# nimsem's overload resolution asks "does this untyped integer literal fit the parameter's type?"
# via `checkIntLitRange`, which compares an `xint` — nimony's `{nan, neg, val: uint64}` extended
# integer — against `lastOrd(T)`. For `uint64` that bound is `high(uint64)`, the one ordinal bound
# that does not fit in an `int64`. Every narrower unsigned type's bound does, which is why a no-C
# nimsem rejected `x < 10` for `uint64` and accepted it for `uint32`.
#
# This is that check, reduced to a program: the `xint` shape, its comparison operators, and the
# bounds themselves.
import std/syncio

type
  Xint = object
    nan: bool
    neg: bool
    val: uint64

proc createXint(x: uint64): Xint = Xint(nan: false, neg: false, val: x)

proc eq(a, b: Xint): bool =
  if a.nan: return b.nan
  elif b.nan: return false
  if a.val == 0'u64 and b.val == 0'u64: return true
  a.neg == b.neg and a.val == b.val

proc lt(a, b: Xint): bool =
  if a.nan or b.nan: return false
  if a.val == 0'u64 and b.val == 0'u64: return false
  if a.neg and not b.neg: return true
  if not a.neg and b.neg: return false
  if a.neg: a.val > b.val
  else: a.val < b.val

proc le(a, b: Xint): bool = lt(a, b) or eq(a, b)

proc go(): string =
  let ten = createXint(10'u64)
  let hi8 = createXint(high(uint8).uint64)
  let hi32 = createXint(high(uint32).uint64)
  let hi64 = createXint(high(uint64))
  # The range check itself, at four widths.
  result = $le(ten, hi8) & $le(ten, hi32) & $le(ten, hi64)
  # The bound values, printed — `high(uint64)` is the one that leaves int64's range.
  result.add "|" & $high(uint32) & "|" & $high(uint64)
  # The bare unsigned comparisons the object fields above go through.
  result.add "|" & $(10'u64 < high(uint64)) & $(10'u64 <= high(uint64))
  # And through a field load, which is how `xint` actually reaches them.
  result.add "|" & $(ten.val < hi64.val) & $(hi64.val > ten.val)

write(stdout, go() & "\n")
