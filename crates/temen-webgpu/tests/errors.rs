//! Guest input never panics the host (#2330 A). Each case that once reached wgpu's panicking default
//! error handler (a buffer over the limit, a misaligned write, a bind group short of a binding, a
//! dispatch over the limit) comes back to the guest from its `error` op, and the device stays
//! usable: the guest then runs a compute pass and checks it. On all three engines.

mod common;

#[test]
fn guest_errors_are_values() {
    common::require_adapter();
    let inst = common::guest("errors");
    for backend in common::ENGINES {
        let caps = [(
            "webgpu",
            temen_webgpu::webgpu_cap(temen_webgpu::Surface::None),
        )];
        let stdout = common::run(&inst, backend, &caps);
        for case in [
            "a buffer over the limit: reported",
            "a misaligned write: reported",
            "pipeline: OK",
            "a bind group without a binding: reported",
            "a dispatch over the limit: reported",
            "after the errors: OK",
            "errors: OK",
        ] {
            assert!(
                stdout.contains(case),
                "{case:?} ({backend:?}); stdout:\n{stdout}"
            );
        }
    }
}
