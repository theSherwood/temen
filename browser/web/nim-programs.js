// The Nim card's programs (#2374): the dropdown's entries, and what each prints. play.js loads the
// picked one into the card's editor; `browser-nim-card-tierup-test.mjs` builds every one with the
// committed toolchain and checks it prints `expect`, which is native nimony's output for the same
// source. A program that imports a module the card's library pack lacks still builds, but compiles
// that module first: keep the pack's prelude (`scripts/rebuild-assets.sh`, step `nim_card`) covering
// what these import.
export const NIM_PROGRAMS = [
  {
    name: 'Hello',
    src: `# Edit this Nim, then Run. nimony's own toolchain compiles it in your browser:
# nifler2 (parse) -> nimsem (check) -> hexer (lower) -> temen-link, and the
# result runs on the Temen. The text below is your program's real stdout.
import std/[syncio, strutils]

proc greet(name: string): string =
  "hello, " & name

var words = 0
for word in "the quick brown fox jumps over the lazy dog".split(' '):
  inc words

echo greet("Nim")
echo "words: ", words
echo toUpperAscii("temen")
`,
    expect: 'hello, Nim\nwords: 9\nTEMEN\n',
  },
  {
    name: 'Collections',
    src: `# Count words with a table, collect the distinct ones in a set, keep the
# repeated ones with sequtils, and sort them with std/algorithm.
import std/[syncio, strutils, tables, sets, sequtils, algorithm]

let words = "the cat sat on the mat and the dog sat on the log".split(' ')

proc tally(words: seq[string]): Table[string, int] =
  result = initTable[string, int]()
  for w in words: result[w] = result.getOrDefault(w) + 1

# A \`let\`, so the procs below may read it: a proc may not touch a mutable
# global unless it says how that is synchronized.
let counts = tally(words)
var seen = initHashSet[string]()
var order: seq[string] = @[]
for w in words:
  if not seen.contains(w):
    seen.incl(w)
    order.add w

proc repeated(w: string): bool = counts.getOrDefault(w) > 1
proc byCount(a, b: string): int =
  let d = counts.getOrDefault(b) - counts.getOrDefault(a)
  if d != 0: d else: cmp(a, b)

echo "words: ", words.len, ", distinct: ", seen.len
for w in order.filter(repeated).sorted(byCount):
  echo w, ": ", counts.getOrDefault(w)
`,
    expect: 'words: 13, distinct: 8\nthe: 4\non: 2\nsat: 2\n',
  },
  {
    name: 'JSON',
    src: `# Parse a JSON document, look keys up with {}, and walk an array.
# nimony's std/json is a cursor API over a JsonTree.
import std/[syncio, json]

var doc = parseJson("""{"name": "temen", "langs": ["C", "Nim", "Lua"], "stars": 42}""")
echo "name: ", doc{"name"}.getStr
echo "stars: ", doc{"stars"}.getInt
var n = 0
for lang in items(doc{"langs"}):
  inc n
  echo "lang ", n, ": ", lang.getStr
`,
    expect: 'name: temen\nstars: 42\nlang 1: C\nlang 2: Nim\nlang 3: Lua\n',
  },
  {
    name: 'Macros',
    src: `# A macro runs at compile time and writes code. nimony builds the macro as a
# plugin program and runs it during the build, all in your browser.
import std/[syncio, macros]

macro squareSum(a, b: untyped): untyped =
  # Expands squareSum(a, b) to a*a + b*b.
  result = newTree(nnkInfix, [ident("+"),
    newTree(nnkInfix, [ident("*"), a, a]),
    newTree(nnkInfix, [ident("*"), b, b])])

echo "3*3 + 4*4 = ", squareSum(3, 4)
echo "5*5 + 12*12 = ", squareSum(5, 12)
`,
    expect: '3*3 + 4*4 = 25\n5*5 + 12*12 = 169\n',
  },
  {
    name: 'Compile-time evaluation',
    src: `# A const is evaluated while the program compiles: nimony builds a small
# program that computes it, runs it, and bakes the answer into yours.
import std/syncio

proc fib(n: int): int =
  if n < 2: n else: fib(n - 1) + fib(n - 2)

const F25 = fib(25)

proc collatzSteps(n: int): int =
  result = 0
  var x = n
  while x != 1:
    x = (if x mod 2 == 0: x div 2 else: 3 * x + 1)
    inc result

const Steps = collatzSteps(27)

echo "fib(25) = ", F25, " (computed at compile time)"
echo "collatz(27) takes ", Steps, " steps"
`,
    expect: 'fib(25) = 75025 (computed at compile time)\ncollatz(27) takes 111 steps\n',
  },
  {
    name: 'Threads',
    src: `# Four threads each count the primes in their own range; the main thread
# joins them and adds up their counts. On this card every thread runs on the
# one Worker, taking turns.
import std/[syncio, rawthreads]

const Threads = 4
const Span = 25_000

type Job = object
  lo, hi: int   # the range a thread counts in
  primes: int   # its count, read once the thread is joined

proc isPrime(n: int): bool =
  if n < 2: return false
  var d = 2
  while d * d <= n:
    if n mod d == 0: return false
    inc d
  true

proc work(p: pointer) =
  let job = cast[ptr Job](p)
  for n in job.lo ..< job.hi:
    if isPrime(n): inc job.primes

proc main() =
  var jobs = default(array[Threads, Job])
  var ths {.noinit.}: array[Threads, RawThread]
  try:
    for i in 0 ..< Threads:
      jobs[i] = Job(lo: i * Span, hi: (i + 1) * Span, primes: 0)
      create ths[i], work, addr(jobs[i])
  except:
    echo "could not start a thread"
    return
  for i in 0 ..< Threads: ths[i].join()
  var total = 0
  for i in 0 ..< Threads:
    echo "thread ", i, ": ", jobs[i].primes, " primes below ", jobs[i].hi
    total += jobs[i].primes
  echo "total: ", total

main()
`,
    expect: 'thread 0: 2762 primes below 25000\nthread 1: 2371 primes below 50000\n' +
      'thread 2: 2260 primes below 75000\nthread 3: 2199 primes below 100000\ntotal: 9592\n',
  },
  {
    name: 'Processes',
    src: `# The program starts three copies of itself as worker processes, each with a
# number to sum up to, and reads back their output and exit codes.
import std/[syncio, osproc, cmdline, strutils]

if paramCount() == 2 and paramStr(1) == "worker":
  var n = 0
  try: n = parseInt(paramStr(2))
  except: quit(2)
  var sum = 0
  for k in 1 .. n: sum += k
  echo "worker ", n, ": sum = ", sum
  quit(n mod 7)

for n in [10, 100, 1000]:
  try:
    let (output, code) = execCmdEx(paramStr(0) & " worker " & $n)
    echo output.strip, " (exit code ", code, ")"
  except:
    echo "could not start a worker"
`,
    expect: 'worker 10: sum = 55 (exit code 3)\nworker 100: sum = 5050 (exit code 2)\n' +
      'worker 1000: sum = 500500 (exit code 6)\n',
  },
];
