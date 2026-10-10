# KNOWN GAP — #2260: a derived `ref object` with no method override of its own. nimony lifts a
# `=copy` hook for `Dog.Obj` whose inlined call binds the `var` dest without `haddr`. The hook is
# ill-formed and dead (nimony's C backend drops it), but leng translates every proc, so the link fails.
#
# Object inheritance without methods: `of` tests the runtime type, and a checked conversion
# (`Dog(a)`) reads the derived field.
import std/syncio
type
  Animal = ref object of RootObj
    name: string
  Dog = ref object of Animal
    tricks: int
  Cat = ref object of Animal
proc kind(a: Animal): string =
  if a of Dog: "dog(" & $Dog(a).tricks & ")"
  elif a of Cat: "cat"
  else: "animal"
proc go(): string =
  let zoo: seq[Animal] = @[Animal(Dog(name: "rex", tricks: 3)), Animal(Cat(name: "tom")), Animal(name: "x")]
  result = ""
  for a in zoo: result.add a.name & ":" & kind(a) & ";"
write(stdout, go() & "\n")
