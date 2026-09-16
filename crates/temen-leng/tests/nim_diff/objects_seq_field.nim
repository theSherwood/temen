# An object with a `seq` field — the container twin of objects_variant_string.nim's `string` field
# (#1444/#1480): a non-scalar field built and grown inside a proc, then read back through the object.
import std/syncio

type Bag = object
  name: string
  items: seq[int]

proc total(b: Bag): int =
  result = 0
  for x in b.items:
    result = result + x

proc go(): string =
  var b = Bag(name: "bag", items: @[])
  for i in 1 .. 4:
    b.items.add(i * i)
  b.name & ":" & $total(b) & ":" & $b.items.len

write(stdout, go() & "\n")
