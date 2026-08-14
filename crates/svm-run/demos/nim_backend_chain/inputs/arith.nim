proc fib(n: int): int =
  if n < 2:
    result = n
  else:
    result = fib(n - 1) + fib(n - 2)

proc sumTo(n: int): int =
  var total = 0
  var i = 0
  while i < n:
    total = total + i
    i = i + 1
  total

let a = fib(10)
let b = sumTo(100)
let c = a + b
