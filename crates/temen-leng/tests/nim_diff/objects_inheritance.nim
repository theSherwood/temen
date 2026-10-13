# Object inheritance without methods: `of` tests the runtime type, and a checked conversion
# (`Dog(a)`) reads the derived field. nimony lifts an ill-formed `=copy` hook for `Dog.Obj` (#2260),
# but it is dead, and the program's dead-code pass drops it before the link sees it.
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
