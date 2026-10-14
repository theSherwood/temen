// The playground's **reactor** (#1414 S4-4): an interactive guest opened once, then driven one `tick`
// per frame. A frame is a pump run over the reactor's live window, pumped by the driver every emitted
// run uses (`driveCoopTierupRun`), and the tier is a parameter of the frame: interpreted, or with
// `tick` offered whole to the emitted tier, where it runs as a leaf when the engine can emit it. So both
// tiers run the one guest, and a moment or a save-state taken on either serves both.
import { driveCoopTierupRun, endSession } from './coop-driver.js';

// Open a reactor over the module `bytes`, serving `file` ({ name, data }) through the `fs` capability
// when given, and rebuilt from the save-state `artifact` rather than from `_start` when given. `ex` and
// `memory` are the engine's exports and its linear memory. `null` when the engine refuses, with why in
// `temen_status`.
export function openReactor(ex, memory, bytes, file = null, artifact = null) {
  const put = (b) => {
    const p = Number(ex.temen_alloc(b.length));
    new Uint8Array(memory.buffer).set(b, p);
    return [p, b.length];
  };
  const empty = new Uint8Array(0);
  const bufs = [bytes, file ? new TextEncoder().encode(file.name) : empty, file ? file.data : empty, artifact ?? empty]
    .map(put);
  const [[mp, ml], [np, nl], [dp, dl], [ap, al]] = bufs;
  const opened = artifact
    ? ex.temen_onramp_thaw(ap, al, mp, ml, np, nl, dp, dl)
    : file
      ? ex.temen_onramp_open_fs(mp, ml, np, nl, dp, dl)
      : ex.temen_onramp_open(mp, ml);
  for (const [p, n] of bufs) ex.temen_dealloc(p, n);
  if (opened !== 0) return null;
  // The emitted tier's host and the leaf `tick` runs as, kept from frame to frame.
  const session = {};
  const shared = memory.buffer instanceof SharedArrayBuffer ? 1 : 0;
  return {
    // Run one frame — `tick` offered to the emitted tier when `jit`, else interpreted — and resolve to
    // its status: `0` to keep going, `5` when the guest exited, else a trap (`temen_onramp_trap_len`
    // names it). The frame it presented and what it wrote are in the engine's `temen_framebuffer_*` and
    // `temen_stdout_*` slots. `counts` is filled as `driveCoopTierupRun` fills it: its `leaves` is `1`
    // when `tick` ran emitted.
    async frame(jit, counts = {}) {
      if (ex.temen_onramp_frame_begin(jit ? 1 : 0, shared) !== 0) return ex.temen_status();
      return driveCoopTierupRun(ex, memory, { session, counts, trapDeclines: false });
    },
    close() {
      endSession(session);
      ex.temen_onramp_close();
    },
  };
}
