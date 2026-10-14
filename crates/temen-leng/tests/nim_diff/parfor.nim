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

proc countPrimes(n: int): int =
  var flags = newSeq[bool](n)
  for i in `||`(0, n - 1, workload = CpuBound):
    flags[i] = isPrime(i)
  result = 0
  for f in flags:
    if f: inc result

proc sumSquares(n, chunk: int): int =
  var s = newSeq[int](n)
  for i in `||`(0, n - 1, chunkSize = chunk):
    s[i] = i * i
  result = 0
  for x in s: result += x

echo "primes below 10000: ", countPrimes(10_000)
echo "squares: ", sumSquares(100, 7), " ", sumSquares(1, 1), " ", sumSquares(0, 1)
