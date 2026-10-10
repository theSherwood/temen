# User lifetime hooks: `=destroy` and `=copy` on an object, logged as they run. The log is the
# order and count nimony's ARC calls them in, which the native run fixes.
import std/syncio
var log = ""
type Res = object
  id: int
proc `=destroy`(r: var Res) =
  if r.id != 0: log.add "D" & $r.id
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
