-- Lua (PUC 5.3+, native 64-bit integers + bitwise ops) mirror of kernels.c — i32-LCG masked to 32
-- bits, same computations as the C/Temen/Python kernels. Beside `python` it is the second scripting-
-- language interpreter reference for `temen-bytecode` (Lua's register VM with unboxed integers is the
-- classic "good bytecode interpreter" bar); like `python`, it interprets a transliteration, not the
-- same compiled IR — Pulley remains the apples-to-apples row.
local clock = os.clock
local MASK = 0xffffffff
local function lcg(a, i) return (a * 1103515245 + 12345 + i) & MASK end

local function alu(n)
  local a = 0
  for i = 0, n - 1 do a = lcg(a, i) end
  return a
end
local function xorshift(n)
  local a = 1
  for i = 0, n - 1 do
    a = a ~ ((a << 13) & MASK); a = a ~ (a >> 17); a = (a + i) & MASK
  end
  return a
end
local function step(a, i) return lcg(a, i) end
local function call(n)
  local a = 0
  for i = 0, n - 1 do a = step(a, i) end
  return a
end
local fp = step
local function call_indirect(n)
  local a = 0; local f = fp
  for i = 0, n - 1 do a = f(a, i) end
  return a
end
local function mem(n)
  local cell = 0; local a = 0
  for i = 0, n - 1 do cell = a; a = lcg(cell, i) end
  return a
end

local CN = 4096
local function chase(n)
  local carr = {}
  for i = 0, CN - 1 do carr[i] = (i + 1789) & (CN - 1) end
  local x = 0; local h = 0
  for _ = 1, n do x = carr[x]; h = (h + x) & MASK end
  return h
end
local RN = 1 << 20
local function chase_rand(n)
  local rarr = {}
  for i = 0, RN - 1 do rarr[i] = (i * 1103515245 + 12345) & (RN - 1) end
  local x = 0; local h = 0
  for _ = 1, n do x = rarr[x]; h = (h + x) & MASK end
  return h
end
local FBUF = 4096
local function fnv(n)
  local fbuf = {}
  for i = 0, FBUF - 1 do fbuf[i] = (i * 7 + 1) & 0xff end
  local h = 2166136261
  for k = 0, n - 1 do h = ((h ~ fbuf[k & (FBUF - 1)]) * 16777619) & MASK end
  return h
end
local function fma(n)
  local a = 1.0
  for _ = 1, n do a = a * 0.9999999 + 1.0 end
  return math.floor(a)
end
local function vadd(n)
  local seed = (n * 2654435761) & MASK
  local s = 0
  for k = 0, n - 1 do s = (s + (k ~ seed)) & MASK end
  return s
end

local function min_run(k, n)
  k(n)
  local best = math.huge
  for _ = 1, 7 do
    local a = clock(); k(n); local b = clock()
    if b - a < best then best = b - a end
  end
  return best
end
for _, e in ipairs({ { "alu", alu }, { "xorshift", xorshift }, { "call", call }, { "call_indirect", call_indirect },
  { "mem", mem }, { "chase", chase }, { "chase_rand", chase_rand }, { "fnv", fnv }, { "fma", fma }, { "vadd", vadd } }) do
  local name, k = e[1], e[2]
  local s = min_run(k, 1000); local l = min_run(k, 201000)
  print(string.format("lua,%s,%.4f", name, (l - s) * 1e9 / 200000.0))
end
