# The generic `builtin*` atomics (#1499). `std/atomics` declares them `[T: SomeInteger]`, so nimony
# monomorphizes one `importc` instance per width and the name alone can't say which width it is —
# binding by name gave an `i32` atomic the `i64` shim.
#
# Width is only half of it. `__atomic_fetch_add` returns the value **before** the add, while nim's
# `atomicAddFetch` (already in the shim) returns the value after. Both verify; only one is right.
# Every op below is printed with **both** its return value and the resulting cell, so a shim that
# mutates correctly but returns the wrong word cannot pass.
import std/syncio
import std/atomics

proc go(): string =
  # --- 64-bit (int) ---
  var a: int = 10
  result = $atomicFetchAdd(a, 5) & "/" & $atomicLoad(a)      # 10/15 — returns the OLD value
  result = result & "|" & $atomicFetchSub(a, 3) & "/" & $atomicLoad(a)   # 15/12
  result = result & "|" & $atomicExchange(a, 99) & "/" & $atomicLoad(a)  # 12/99
  atomicStore(a, 7)
  result = result & "|" & $atomicLoad(a)                                  # 7

  var exp: int = 7
  let ok = atomicCompareExchange(a, exp, 42)
  result = result & "|" & $ok & "/" & $atomicLoad(a) & "/" & $exp         # true/42/7
  var bad: int = 1
  let no = atomicCompareExchange(a, bad, 5)
  # on failure `expected` is overwritten with the current value, and the cell is untouched
  result = result & "|" & $no & "/" & $atomicLoad(a) & "/" & $bad         # false/42/42

  # --- 32-bit ---
  var b: int32 = 10'i32
  result = result & "||" & $atomicFetchAdd(b, 5'i32) & "/" & $atomicLoad(b)
  result = result & "|" & $atomicFetchSub(b, 3'i32) & "/" & $atomicLoad(b)
  result = result & "|" & $atomicExchange(b, 99'i32) & "/" & $atomicLoad(b)
  atomicStore(b, 7'i32)
  result = result & "|" & $atomicLoad(b)

  var exp32: int32 = 7'i32
  let ok32 = atomicCompareExchange(b, exp32, 42'i32)
  result = result & "|" & $ok32 & "/" & $atomicLoad(b) & "/" & $exp32
  var bad32: int32 = 1'i32
  let no32 = atomicCompareExchange(b, bad32, 5'i32)
  result = result & "|" & $no32 & "/" & $atomicLoad(b) & "/" & $bad32

  # A 32-bit cell must not be touched by a 64-bit access: write a sentinel immediately after `b`
  # and check it survives. This is what a wrong-width bind would corrupt.
  var guard: int32 = 1234'i32
  atomicStore(b, -1'i32)
  result = result & "||" & $atomicLoad(b) & "/" & $guard

write(stdout, go() & "\n")
