// How the Node harnesses instantiate the **engine** wasm (`temen_browser.wasm`): as the page and its
// Workers do (`web/engine-imports.js`), with the same imports, and adopted. A harness that drives a
// seam installs its `globalThis` hook as the page does — `driveCoopTierupRun` installs the output sink
// and the JS-capability servicer — and with none a seam does nothing: `webgpu_op` answers -1,
// `js_cap_call` -ENOSYS, and a refused `x2_spawn` ends a parallel run with `ThreadFault`.
export { instantiateEngine } from './web/engine-imports.js';
