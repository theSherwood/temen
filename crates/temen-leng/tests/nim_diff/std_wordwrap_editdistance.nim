# Two pure string algorithms: `std/wordwrap`'s greedy wrap and `std/editdistance`'s Levenshtein.
import std/syncio
import std/wordwrap
import std/editdistance

proc go(): string =
  result = wrapWords("the quick brown fox jumps over the lazy dog", 12)
  result = result & "|" & $editDistance("kitten", "sitting")
  result = result & "|" & $editDistance("", "abc") & "|" & $editDistance("same", "same")

write(stdout, go() & "\n")
