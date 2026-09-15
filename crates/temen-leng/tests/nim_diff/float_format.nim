# strutils float formatting — the `snprintf` path, across the three format modes and precisions.
import std/syncio
import std/strutils

write(stdout, formatFloat(3.14159, ffDecimal, 3) & "|" & formatFloat(3.14159, ffScientific, 2) & "\n")
write(stdout, formatFloat(0.000123, ffDecimal, 8) & "|" & formatFloat(1234567.0, ffDecimal, 1) & "\n")
write(stdout, formatFloat(-0.5, ffDecimal, 4) & "|" & formatFloat(0.0, ffDecimal, 2) & "\n")
