# `std/paths` + `std/pathnorm` — pure path algebra. Nothing here consults a filesystem.
import std/syncio
import std/paths
import std/pathnorm

proc go(): string =
  result = normalizePath("a/b/../c/./d.txt")
  let sf = path("x/y/z.nif").splitFile()
  result = result & "|" & $sf.dir & "," & $sf.name & "," & sf.ext
  result = result & "|" & $(path("x/y") / path("z.nif"))
  result = result & "|" & $path("rel/p").isAbsolute & $path("/abs/p").isAbsolute
  result = result & "|" & $path("a/b/c.txt").parentDir
  result = result & "|" & $path("a/b/c.txt").changeFileExt("nif")

write(stdout, go() & "\n")
