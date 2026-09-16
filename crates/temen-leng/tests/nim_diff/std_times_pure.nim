# `std/times` over **fixed** instants — no clock is read, so both engines see the same input. Covers
# the int64 second/nanosecond arithmetic, Duration normalization, and the civil-date decomposition.
import std/syncio
import std/times

proc go(): string =
  let t = fromUnix(1_000_000_000'i64)   # 2001-09-09T01:46:40Z
  result = $t.toUnix
  let d = initDuration(seconds = 90, nanoseconds = 500_000_000)
  result = result & "|" & $d.inSeconds & "," & $d.inMilliseconds
  let t2 = t + d
  result = result & "|" & $t2.toUnix & "," & $t2.nanosecond
  result = result & "|" & $((t2 - t).inMilliseconds)
  result = result & "|" & $(t < t2) & $(t2 <= t)
  let dt = utc(t)
  result = result & "|" & $dt.year & "-" & $ord(dt.month) & "-" & $dt.monthday
  result = result & "|" & $isLeapYear(2000) & $isLeapYear(1900) & $isLeapYear(2024)
  result = result & "|" & $getDaysInMonth(mFeb, 2024) & "," & $getDaysInMonth(mFeb, 2023)

write(stdout, go() & "\n")
