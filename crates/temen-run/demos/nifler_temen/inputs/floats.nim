# Float literals that miss `parseBiggestFloat`'s fast path and so reach `c_strtod` — the shim
# `nifler_shim.c` provides. A nifler built before that shim traps `Unreachable` on every line here,
# which is exactly how a stale asset shipped and made `import std/math` uncompilable in the
# playground (#1364). These are real constants from `std/fenv` and `std/math`.
const
  FLT_MIN = 1.17549435e-38'f32
  FLT_EPSILON = 1.19209290e-07'f32
  DBL_MIN = 2.2250738585072014E-308
  DBL_EPSILON = 2.2204460492503131E-16
  DBL_MAX = 1.7976931348623157E+308
  big = 1.5e100
  small = 1.5e-100
  longMantissa = 2.2204460492503131

proc use(): float =
  result = DBL_EPSILON + big + small + longMantissa + DBL_MIN + DBL_MAX
