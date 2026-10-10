# Threads (`std/rawthreads`): four threads each sum a disjoint range into their own slot, and are
# joined before the total is read, so the result does not depend on how they interleave.
import std/[rawthreads, syncio]
var parts: array[4, int]
var ids: array[4, int]
proc work(p: pointer) =
  let i = cast[ptr int](p)[]
  var t = 0
  for k in i * 1000 ..< (i + 1) * 1000: t += k
  parts[i] = t
proc go(): string =
  var ths {.noinit.}: array[4, RawThread]
  try:
    for i in 0 ..< 4:
      ids[i] = i
      create ths[i], work, addr(ids[i])
  except:
    return "error creating a thread"
  for i in 0 ..< 4: ths[i].join()
  var total = 0
  for p in parts: total += p
  result = $total
write(stdout, go() & "\n")
