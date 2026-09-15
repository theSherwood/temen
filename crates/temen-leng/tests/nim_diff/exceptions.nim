# nimony's exception model (#980): a `{.raises.}` proc `raise`s an **ErrorCode**, the caller catches
# with `except ErrorCode`, and hexer lowers it to a returned (code, value) pair. Not standard Nim's
# `newException` / object exceptions.
import std/syncio

proc mayFail(x: int): int {.raises.} =
  if x < 0: raise ValueError
  else: x * 2

proc safe(x: int): int =
  try: mayFail(x)
  except ErrorCode: -1

proc withFinally(x: int): int =
  var acc = 0
  try:
    acc += 1
    acc += mayFail(x)
  except ErrorCode:
    acc += 100
  finally:
    acc += 1000
  acc

write(stdout, $safe(21) & "|" & $safe(-1) & "\n")
write(stdout, $withFinally(5) & "|" & $withFinally(-5) & "\n")
