// The JS capabilities the embedder cases stage (#2579, #2645), by slot. One module, so a page's run
// (`onCap`) and executor 2's Workers (`capModule`) serve the same functions:
// - 0 `js.upper(ptr, len)`: uppercases the guest's bytes in place and returns `len` (-1 when it can't);
// - 1 `js.add(a, b)`: `a + b`;
// - 2 `js.waitFor(ptr, more)`: waits until the i64 at `ptr` in the guest's window has grown by `more`
//   since the call began, and returns 1, or 0 once `WAIT_MS` have passed without it.
export const WAIT_MS = 5000;

export function onCap(slot, op, args, mem) {
  if (slot === 0) return mem.write(args[0], mem.str(args[0], args[1]).toUpperCase()) ? args[1] : -1n;
  if (slot === 1) return args[0] + args[1];
  const word = () => {
    const b = mem.bytes(args[0], 8);
    return b ? new DataView(b.buffer).getBigInt64(0, true) : null;
  };
  const from = word();
  const until = performance.now() + WAIT_MS;
  while (from !== null && performance.now() < until) {
    if (word() - from >= args[1]) return 1n;
  }
  return 0n;
}
