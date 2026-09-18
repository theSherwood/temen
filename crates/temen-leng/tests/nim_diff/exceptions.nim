# nimony's exception model (#980): a `{.raises.}` proc `raise`s an **ErrorCode**, the caller catches
# with `except ErrorCode`, and hexer lowers it to a returned (code, value) pair. Not standard Nim's
# `newException` / object exceptions.
import std/syncio

proc mayFail(x: int): int {.raises.} =
  # Statement-form `if`, not `if x < 0: raise ValueError else: x * 2`. That expression form regressed
  # in nimony v0.6.2 — nimsem's `xelim` cannot type an `if` whose branch `raise`s
  # (`result.typeKind != AutoT`, xelim.nim:113) and the compile aborts. Upstream, not our lowering:
  # it never reaches Leng. The exception model below is what this case is here to cover, and it is
  # unaffected, so the fixture takes the shape that compiles rather than dropping the coverage.
  if x < 0:
    raise ValueError
  result = x * 2

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
