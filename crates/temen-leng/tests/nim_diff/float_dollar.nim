# `$` on floats — the shortest-round-trip dtoa (#1472). Exact powers of two, values needing the
# general digit generation, and the float32 overload.
import std/syncio

write(stdout, $0.5 & "|" & $2.0 & "|" & $0.25 & "|" & $100.0 & "\n")
write(stdout, $1.5 & "|" & $1.25 & "|" & $0.1 & "|" & $3.14 & "|" & $7.5 & "\n")
write(stdout, $1e10 & "|" & $1.0e-5 & "|" & $(-2.75) & "\n")
let f: float32 = 0.5'f32
let g: float32 = 3.25'f32
write(stdout, $f & "|" & $g & "\n")
