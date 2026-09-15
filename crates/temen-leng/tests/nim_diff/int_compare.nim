# Unsigned comparison must not use the signed ordering: a uint32 with bit 31 set is the larger value.
import std/syncio

let hi: uint32 = 0x80000000'u32
let lo: uint32 = 0x00000001'u32
write(stdout, $(hi > lo) & "|" & $(lo < hi) & "|" & $(hi == hi) & "\n")
let hi64: uint64 = 0x8000000000000000'u64
let lo64: uint64 = 1'u64
write(stdout, $(hi64 > lo64) & "|" & $(lo64 < hi64) & "\n")
let si: int32 = -1'i32
write(stdout, $(si < 0'i32) & "|" & $(si > 0'i32) & "\n")
