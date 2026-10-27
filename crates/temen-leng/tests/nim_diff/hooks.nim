# User lifetime hooks: `=destroy` and `=copy` on an object, logged as they run. The log is the
# order and count nimony's ARC calls them in, which the native run fixes.
import std/syncio
{.feature: "assumeSync".}  # `log` is shared by the hooks that write it, on one thread
var log = ""
type Res = object
  id: int
proc `=destroy`(r: var Res) =
  if r.id != 0: log.add "D" & $r.id
# A written `=destroy` needs a written `=wasMoved`; this one zeroes, as the implicit one did.
proc `=wasMoved`(r: var Res) {.nodestroy, inline.} = zeroMem(addr r, sizeof(r))
proc `=copy`(dst: var Res; src: Res) =
  log.add "C" & $src.id
  dst.id = src.id * 10
proc mk(i: int): Res = Res(id: i)
proc go() =
  var a = mk(1)
  var b = a
  discard b
go()
write(stdout, log & "\n")
