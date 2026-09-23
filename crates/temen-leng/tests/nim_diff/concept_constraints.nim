# Generic constraints expressed as **concepts**, including inheritance (#1630).
#
# `std/math` builds a small hierarchy — `Arithmetic`, then `IntegerArithmetic`/`SignedArithmetic`
# `concept of` it — and constrains its generics on the result (`euclMod*[T: SomeNumber and
# SignedArithmetic]`). A refreshed no-C nimsem rejects that module, printing the ordinary "no
# viable overload" candidate dump — the same one native prints for a generic whose constraint does
# not apply. That is the signature of constraint matching failing, not of a comparison lowering
# wrong (#1635 ruled the comparison out).
#
# So: the hierarchy itself, `concept of` inheritance, an `and` of two constraints, and dispatch
# that has to pick between overloads on the concept — the machinery `math` depends on, at a size
# that runs in seconds instead of a nimsem build.
#
# Builtin types only: a locally-defined `Cents` with its own `+`/`<`/`-` does **not** satisfy these
# concepts under native nimony either ("Cents does not match constraint Addable"), so structural
# matching over user types is a separate question from the one `math` raises.
import std/syncio

type
  Addable = concept
    func `+`(x, y: Self): Self
    func `<`(x, y: Self): bool

  Negatable = concept of Addable ## `Addable` plus negation — `math`'s `SignedArithmetic` shape.
    func `-`(x: Self): Self

proc sum2[T: Addable](a, b: T): T = a + b

proc absLike[T: Negatable](x: T): T =
  # The `euclMod` shape: a comparison against a converted zero, then negation.
  if x < T(0): -x else: x

proc bothConstraints[T: SomeInteger and Negatable](x: T): T = absLike(x) + T(1)

proc main() =
  stdout.write($sum2(2'i32, 3'i32) & "|" & $sum2(2.5'f64, 0.25'f64) & "\n")

  # Signed integer widths and floats through the inherited concept.
  stdout.write($absLike(-7'i8) & "|" & $absLike(7'i8))
  stdout.write("|" & $absLike(-7'i32) & "|" & $absLike(-7'i64) & "\n")
  stdout.write($absLike(-2.5'f64) & "|" & $absLike(2.5'f64) & "\n")

  # An `and` of a builtin constraint with a concept.
  stdout.write($bothConstraints(-4'i64) & "|" & $bothConstraints(4'i32) & "\n")

main()
