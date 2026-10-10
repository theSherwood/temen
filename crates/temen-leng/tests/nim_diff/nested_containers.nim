# Nested containers: a two-dimensional fixed array and a `seq[seq[string]]` built row by row.
import std/syncio
var grid: array[3, array[4, int]]
for r in 0 ..< 3:
  for c in 0 ..< 4:
    grid[r][c] = r * 4 + c
var rows: seq[seq[string]] = @[]
for r in 0 ..< 3:
  var row: seq[string] = @[]
  for c in 0 .. r: row.add $(r + c)
  rows.add row
var s = $grid[2][3] & "|"
for row in rows:
  for v in row: s.add v
  s.add ";"
write(stdout, s & "\n")
