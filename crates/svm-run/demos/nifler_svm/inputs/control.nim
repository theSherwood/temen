type Color = enum cRed, cGreen, cBlue

proc classify(n: int): string =
  if n < 0:
    result = "neg"
  elif n == 0:
    result = "zero"
  else:
    result = "pos"

proc sumTo(n: int): int =
  var total = 0
  for i in 0 ..< n:
    total += i
  while total > 100:
    total = total div 2
  total

let c = cGreen
let s = classify(-5)
let t = sumTo(20)
