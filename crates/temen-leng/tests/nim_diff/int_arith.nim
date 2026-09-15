# Arithmetic, division/modulo signs, and shifts. Nim's `div`/`mod` truncate toward zero.
import std/syncio

let a = 17
let b = -5
write(stdout, $(a div b) & "|" & $(a mod b) & "|" & $(b div a) & "|" & $(b mod a) & "\n")
write(stdout, $(a shl 3) & "|" & $(a shr 2) & "|" & $(b shr 1) & "\n")
let big = 9223372036854775807'i64
write(stdout, $(big - 1) & "|" & $(-big - 1) & "\n")
let ua = 0xF0F0F0F0'u32
write(stdout, $int(ua shr 4) & "|" & $int(ua shl 4) & "|" & $int(not ua) & "\n")
