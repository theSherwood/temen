# libm through the guest libc (#1375) — powers, roots, logs, rounding, and the trig family.
import std/syncio
import std/strutils
import std/math

proc f(x: float): string = formatFloat(x, ffDecimal, 6)

write(stdout, f(sqrt(2.0)) & "|" & f(pow(2.0, 10.0)) & "|" & f(exp(1.0)) & "\n")
write(stdout, f(ln(100.0)) & "|" & f(log10(1000.0)) & "|" & f(log2(8.0)) & "\n")
write(stdout, f(floor(-1.5)) & "|" & f(ceil(-1.5)) & "|" & f(round(2.5)) & "|" & f(trunc(-2.7)) & "\n")
write(stdout, f(sin(0.5)) & "|" & f(cos(0.5)) & "|" & f(arctan2(1.0, 2.0)) & "|" & f(hypot(3.0, 4.0)) & "\n")
