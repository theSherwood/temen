# A parallel for (`std/parfor`'s `||`): the iterations run as chunks on the
# thread pool, and the loop ends when all of them have. Each iteration writes
# only its own slot, so the result does not depend on the schedule.
import std/[parfor, syncio]

proc isPrime(n: int): bool =
  if n < 2: return false
  var d = 2
  while d * d <= n:
    if n mod d == 0: return false
    inc d
  result = true

# The infix form, with the default grain and join (the card's program).
proc countPrimes(n: int): int =
  var flags = newSeq[bool](n)
  for i in 0 || (n - 1):
    flags[i] = isPrime(i)
  result = 0
  for f in flags:
    if f: inc result

# An explicit grain, over ranges of one iteration and of none.
proc sumSquares(n, chunk: int): int =
  var s = newSeq[int](n)
  for i in `||`(0, n - 1, chunkSize = chunk):
    s[i] = i * i
  result = 0
  for x in s: result += x

# A stride, and a join that only helps the pool (no I/O poll).
proc everyThird(n: int): int =
  var hit = newSeq[int](n)
  for i in `||`(0, n - 1, 3, workload = CpuBound):
    hit[i] = 1
  result = 0
  for x in hit: result += x

echo "primes below 10000: ", countPrimes(10_000)
echo "squares: ", sumSquares(100, 7), " ", sumSquares(1, 1), " ", sumSquares(0, 1)
echo "every third: ", everyThird(100), " ", everyThird(3)
