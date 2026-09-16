# Tuples: anonymous, named, returned from a proc (an sret aggregate), and destructured at the call site.
import std/syncio

proc divmod2(a, b: int): (int, int) = (a div b, a mod b)

proc named(): tuple[q: int, r: string] = (q: 5, r: "five")

proc go(): string =
  let (q, r) = divmod2(17, 5)
  let n = named()
  $q & "|" & $r & "|" & $n.q & "|" & n.r

write(stdout, go() & "\n")
