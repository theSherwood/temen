# NIM.md Phase 1, step 2 — a real Nim program compiled by the stock Nim compiler's ARC
# backend, on-ramped to Temen (build_nim.sh). This is genuine Nim-compiler codegen — ref
# objects (ARC destructors), a heap-linked list, and a seq grown at runtime — not the
# hand-modeled shapes of arc_probe.c. It stands in for nimony's Leng->C output until a
# nimony bootstrap is available (NIM.md §2 status): stock Nim and nimony share the ARC/ORC
# memory model and the C-ABI codegen shape the on-ramp must handle.
#
# Deterministic stdout + exit code so the Temen run diffs byte-for-byte against a native build.

type Node = ref object   # ref object => ARC-managed; hexer/Nim inject =destroy calls.
  value: int
  next: Node

proc sumList(head: Node): int =
  var n = head
  while n != nil:
    result += n.value
    n = n.next

proc build(k: int): Node =
  # Builds 1*1, 2*2, ... k*k as a linked list (each Node heap-allocated, ARC-owned).
  for i in countdown(k, 1):
    result = Node(value: i * i, next: result)

proc main() =
  let head = build(10)
  let listSum = sumList(head)          # 1+4+9+...+100 = 385

  var s = newSeq[int](0)               # seq: {len,cap,data} grown via realloc.
  for i in 1 .. 10: s.add(i)
  var seqSum = 0
  for x in s: seqSum += x              # 1+2+...+10 = 55

  echo "listSum=", listSum
  echo "seqSum=", seqSum
  quit(if listSum == 385 and seqSum == 55: 0 else: 1)

main()
