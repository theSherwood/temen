# Procs as values: a `{.nimcall.}` proc type, an array of procs, a proc parameter and a proc
# variable.
import std/syncio
type BinOp = proc (a, b: int): int {.nimcall.}
proc plus(a, b: int): int = a + b
proc times(a, b: int): int = a * b
proc apply(op: BinOp, a, b: int): int = op(a, b)
let ops = [plus, times]
var s = ""
for op in ops: s.add $apply(op, 6, 7) & ","
var f: BinOp = plus
s.add $f(1, 2)
write(stdout, s & "\n")
