// The import object every instantiation of the **engine** wasm (`temen_browser.wasm`) supplies, for
// the Node harnesses: the one the page and its Workers build (`web/engine-imports.js`). A harness that
// drives a seam installs its `globalThis` hook as the page does — `driveCoopTierupRun` installs the
// output sink and the JS-capability servicer — and with none a seam does nothing: `webgpu_op` answers
// -1, `js_cap_call` -ENOSYS, and a refused `x2_spawn` ends a parallel run with `ThreadFault`.
export { engineImports } from './web/engine-imports.js';
