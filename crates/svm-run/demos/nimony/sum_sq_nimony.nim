# NIM.md Phase 1, step 2 (proper) — a real program compiled by NIMONY itself (not stock Nim),
# whose genuine Leng->C output (via `nimony c`) is on-ramped to SVM and runs there. This is the
# authentic pipeline: nifler -> nimony(sema) -> hexer(lowering, ARC) -> Leng -> lengc(C).
#
# Written to nimony's stricter dialect (v0.4.0): `result` must be explicitly initialized (the
# NJVL "cannot prove initialized" analysis), `ref` types are non-nilable, and I/O comes from
# std/syncio (`echo`). Exercises a heap-managed `seq[int]` (ARC) and a `var object` parameter.
#
# Deterministic output (sum_sq=385 / count=10) so the SVM run diffs against nimony's native run.

import std/[syncio]

type Acc = object
  total: int
  count: int

proc addSquare(a: var Acc; x: int) =
  a.total += x * x
  a.count += 1

proc main() =
  var s: seq[int] = @[]        # ARC-managed dynamic array (grows via the nimony allocator).
  var i = 1
  while i <= 10:
    s.add i
    inc i
  var a = Acc(total: 0, count: 0)
  for x in s:
    addSquare(a, x)            # 1*1 + 2*2 + ... + 10*10 = 385
  echo "sum_sq=" & $a.total
  echo "count=" & $a.count

main()
