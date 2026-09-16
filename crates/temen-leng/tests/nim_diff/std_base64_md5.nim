# `std/base64` round-trip across all three padding cases, and an `std/md5` digest.
import std/syncio
import std/base64
import std/md5

proc go(): string =
  let e = encode("Hello, Temen!")
  result = e & "|" & decode(e)
  result = result & "|" & encode("") & encode("a") & "|" & encode("ab") & "|" & encode("abc")
  result = result & "|" & $toMD5("abc")

write(stdout, go() & "\n")
