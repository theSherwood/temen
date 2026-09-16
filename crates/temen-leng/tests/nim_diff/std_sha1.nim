# `std/sha1` over the canonical FIPS-180 vectors, through the incremental `update`/`finalize` state.
import std/syncio
import std/sha1

proc digest(s: string): string =
  var st = newSha1State()
  st.update(s)
  result = $SecureHash(st.finalize())

proc go(): string =
  result = digest("abc")
  result = result & "|" & digest("")
  result = result & "|" & digest("The quick brown fox jumps over the lazy dog")

write(stdout, go() & "\n")
