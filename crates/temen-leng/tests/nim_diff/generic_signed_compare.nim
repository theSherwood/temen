# Signed comparisons reached through a **constrained generic** (#1630).
#
# `std/math`'s `euclMod*[T: SomeNumber and SignedArithmetic](x, y: T): T` tests `result < 0` on a
# generic `T`, and a refreshed no-C nimsem rejects that module where native accepts it. The shape
# is one #1612 could plausibly have broken: `TyDesc::Scalar` now carries `unsigned`, and
# `operand_unsigned` asks the declared type for an lvalue. If a generic-instantiated leaf answers
# "unsigned" where its instantiation is signed, `x < 0` lowers to `lt_u` and is **always false** —
# silently, with no type error anywhere. The unsigned rows are the other half: they must NOT come
# out `true` by sign-extension.
#
# The constraint is load-bearing. An *unconstrained* `T` makes `x < T(0)` ambiguous and native
# nimony rejects it with the same candidate dump #1630 quotes, so the case has to be constrained
# to be about codegen rather than overload resolution. (`x mod y` under `SomeInteger` does not
# resolve in nimony either — `math` reaches it through concepts — so this stays on `<`.)
import std/syncio

proc negp[T: SomeInteger](x: T): bool = x < T(0)

proc ltp[T: SomeInteger](a, b: T): bool = a < b

type Holder[T] = object
  val: T

proc fieldNeg[T: SomeInteger](h: Holder[T]): bool = h.val < T(0)

proc main() =
  # A constrained generic at each signed width: the negative case must be `true`.
  stdout.write($negp(-1'i8) & "|" & $negp(1'i8))
  stdout.write("|" & $negp(-1'i16) & "|" & $negp(1'i16))
  stdout.write("|" & $negp(-1'i32) & "|" & $negp(1'i32))
  stdout.write("|" & $negp(-1'i64) & "|" & $negp(1'i64) & "\n")

  # The same generic instantiated *unsigned*: never negative, and the comparison must not
  # sign-extend its way to `true`. The high values are the ones that tell signed from unsigned.
  stdout.write($negp(0'u8) & "|" & $negp(255'u8) & "|" & $negp(0xFFFFFFFF'u32))
  stdout.write("|" & $negp(0xFFFFFFFFFFFFFFFF'u64) & "\n")

  # Two-operand form, at the boundary where signed and unsigned disagree.
  stdout.write($ltp(-1'i64, 1'i64) & "|" & $ltp(1'i64, -1'i64))
  stdout.write("|" & $ltp(0xFFFFFFFFFFFFFFFF'u64, 1'u64) & "|" & $ltp(1'u64, 0xFFFFFFFFFFFFFFFF'u64) & "\n")

  # The same test reached through a field of a generic object — the lvalue route #1612 changed.
  stdout.write($fieldNeg(Holder[int64](val: -5'i64)) & "|" & $fieldNeg(Holder[int64](val: 5'i64)))
  stdout.write("|" & $fieldNeg(Holder[int32](val: -5'i32)))
  stdout.write("|" & $fieldNeg(Holder[uint64](val: 0xFFFFFFFFFFFFFFFF'u64)) & "\n")

main()
