# `std/deques` (both ends) and `std/heapqueue` (drained in order, which is the ordering invariant).
import std/syncio
import std/deques
import std/heapqueue

proc go(): string =
  var d = initDeque[int]()
  d.addLast(1)
  d.addLast(2)
  d.addFirst(0)
  result = $d.len & ":" & $d.peekFirst & ":" & $d.peekLast
  result = result & "|" & $d.popFirst & $d.popLast & ":" & $d.len

  var h = initHeapQueue[int]()
  for x in [5, 1, 4, 2]:
    h.push(x)
  result = result & "|"
  while h.len > 0:
    result = result & $h.pop()

write(stdout, go() & "\n")
