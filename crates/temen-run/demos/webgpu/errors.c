/* errors.c — guest input never panics the host (#2330 A; temen-gpu, #2665). Each case below once
 * reached wgpu's default error handler, which panics. Now each comes back to the guest as a value
 * from `gpu_error`, the device stays usable, and the run goes on: the program ends with a compute
 * pass whose result it checks.
 *
 * A stream the protocol itself refuses runs nothing (`render.c` checks that); these streams are
 * well-formed, so they run, and WebGPU reports what it rejects. */

#include <stdio.h>

#ifdef TEMEN_GUEST
#include "temen_gpu.h"
#endif

static unsigned char stream[8 << 10];
static char why[512];
static unsigned out[64];

enum { HUGE = 1, SMALL, MOD, PIPE, LAYOUT, GROUP, PARTIAL, BUF, OUT };

static const char shader[] =
    "@group(0) @binding(0) var<storage, read> a: array<u32>;\n"
    "@group(0) @binding(1) var<storage, read_write> b: array<u32>;\n"
    "@compute @workgroup_size(64)\n"
    "fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n"
    "  if (g.x < arrayLength(&b)) { b[g.x] = a[g.x] + 1u; }\n"
    "}\n";

/* Submits the stream, and says whether it ran and WebGPU then reported an error. */
static int reported(gpu_stream *s, const char *what) {
  long r = gpu_submit(s);
  long n = gpu_error(why, sizeof why);
  int ok = r == 0 && n > 0;
  printf("%s: %s (%s)\n", what, ok ? "reported" : "NOT REPORTED", why);
  return ok;
}

int main(void) {
  int ok = 1;
  gpu_stream s;
  gpu_stream_init(&s, stream, sizeof stream);
  const unsigned storage = GPU_BUFFER_STORAGE | GPU_BUFFER_COPY_DST | GPU_BUFFER_COPY_SRC;

  /* A buffer over the device's limit. */
  gpu_create_buffer(&s, HUGE, 1ull << 40, storage);
  ok &= reported(&s, "a buffer over the limit");

  /* A write whose size is not a multiple of 4. */
  gpu_create_buffer(&s, SMALL, 16, storage);
  gpu_write_buffer(&s, SMALL, 0, "abc", 3);
  ok &= reported(&s, "a misaligned write");

  /* The pipeline the rest uses. */
  gpu_create_shader_module(&s, MOD, shader);
  gpu_create_compute_pipeline(&s, PIPE, MOD, "main");
  gpu_bind_group_layout_of(&s, LAYOUT, PIPE, 0);
  gpu_create_buffer(&s, BUF, sizeof out, storage);
  gpu_create_buffer(&s, OUT, sizeof out, storage);
  gpu_bind_entry e[] = {{0, GPU_BIND_BUFFER, BUF, 0, 0}, {1, GPU_BIND_BUFFER, OUT, 0, 0}};
  gpu_create_bind_group(&s, GROUP, LAYOUT, 2, e);
  long r = gpu_submit(&s);
  long n = gpu_error(why, sizeof why);
  printf("pipeline: %s%s%s\n", r == 0 && n == 0 ? "OK" : "FAIL ", n > 0 ? why : "",
         r != 0 ? " (refused)" : "");
  ok &= r == 0 && n == 0;

  /* A bind group that leaves out a binding the shader declares. */
  gpu_create_bind_group(&s, PARTIAL, LAYOUT, 1, e);
  ok &= reported(&s, "a bind group without a binding");

  /* A dispatch over the per-dimension limit. */
  gpu_begin_compute_pass(&s);
  gpu_set_pipeline(&s, PIPE);
  gpu_set_bind_group(&s, 0, GROUP);
  gpu_dispatch(&s, 70000, 1, 1);
  gpu_end_pass(&s);
  ok &= reported(&s, "a dispatch over the limit");

  /* The device is still usable. */
  unsigned in[64];
  for (int i = 0; i < 64; i++) in[i] = (unsigned)i * 3;
  gpu_write_buffer(&s, BUF, 0, in, sizeof in);
  gpu_begin_compute_pass(&s);
  gpu_set_pipeline(&s, PIPE);
  gpu_set_bind_group(&s, 0, GROUP);
  gpu_dispatch(&s, 1, 1, 1);
  gpu_end_pass(&s);
  gpu_readback(&s, OUT, 0, sizeof out);
  r = gpu_submit(&s);
  n = r == 0 ? gpu_read_wait(OUT, out, sizeof out) : r;
  int after = n == (long)sizeof out && gpu_error(why, sizeof why) == 0;
  for (int i = 0; i < 64 && after; i++) after = out[i] == in[i] + 1;
  printf("after the errors: %s\n", after ? "OK" : "FAIL");
  ok &= after;

  printf("errors: %s\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
