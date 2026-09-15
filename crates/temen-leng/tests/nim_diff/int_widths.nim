# Integer widths and signedness — the #1472 class. Unsigned values with the high bit set must
# zero-extend into wider contexts; signed ones must sign-extend.
import std/syncio

let u8v: uint8 = 0xFF'u8
let u16v: uint16 = 0xFFFF'u16
let u32v: uint32 = 0x81CEB32C'u32
let u64v: uint64 = 0x81CEB32C4B43FCF5'u64
let i8v: int8 = -1'i8
let i16v: int16 = -1'i16
let i32v: int32 = -1'i32

write(stdout, $int(u8v) & "|" & $int(u16v) & "|" & $int(u32v) & "\n")
write(stdout, $int(i8v) & "|" & $int(i16v) & "|" & $int(i32v) & "\n")
write(stdout, $int(u64v shr 32) & "|" & $int(u64v and 0xFFFFFFFF'u64) & "\n")
write(stdout, $int(uint64(u32v) * 3'u64) & "|" & $int(uint64(u32v) + 1'u64) & "\n")
