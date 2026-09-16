# `std/lexbase` — the buffered line-tracking lexer base, driven over a `StringStream`. Walks the
# whole input a character at a time, folding CR/LF through the handlers that advance `lineNumber`.
import std/syncio
import std/lexbase
import std/streams

proc go(): string =
  try:
    var L: BaseLexer = default(BaseLexer)
    L.open(newStringStream("ab\ncd\r\nef"))
    var chars = 0
    var lines = ""
    while true:
      let ch = L.buf[L.bufpos]
      if ch == '\0': break
      elif ch == '\r':
        let p = L.bufpos          # copied out: passing `L.bufpos` while `L` is the `var`
        L.bufpos = L.handleCR(p)  # receiver is a mutable/immutable alias nimony rejects
      elif ch == '\n':
        let p = L.bufpos
        L.bufpos = L.handleLF(p)
      else:
        chars = chars + 1
        L.bufpos = L.bufpos + 1
      lines = $L.lineNumber
    result = $chars & "|" & lines & "|" & $L.ioError
    L.close()
  except:
    result = "raised"

write(stdout, go() & "\n")
