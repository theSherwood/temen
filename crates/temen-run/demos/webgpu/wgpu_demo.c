/* wgpu_demo.c — the headless WebGPU **compute → readback → assert-vs-CPU** demo (LLVM.md WebGPU
 * demo 1), on the `webgpu` capability's command streams (temen-gpu, #2665). Proves the whole data
 * plane — upload → WGSL compute → readback — with zero windowing. Two kernels:
 *   (1) an in-place `u32` map (`b[i] = b[i]*3 + 7`) over one storage buffer;
 *   (2) a two-buffer kernel (`c[i] = a[i]*a[i] + i`) — a read-only input at binding 0 and a
 *       read_write output at binding 1 — exercising multi-buffer bind groups.
 * Each result is re-computed on the CPU **in-guest** and compared element-by-element, so a wrong
 * dispatch or a mis-marshaled buffer is caught here; the program prints OK/FAIL + an FNV digest and
 * exits 0 only if both kernels match. There is no native oracle (the capability is TEMEN-only, like the
 * async/JIT demos), so the test asserts this self-check. */

#include <stdio.h>

#ifdef TEMEN_GUEST
#include "temen_gpu.h"
#endif

#define N 1024

static unsigned fnv(const unsigned *a, int n) {
  unsigned h = 2166136261u;
  for (int i = 0; i < n; i++) h = (h ^ a[i]) * 16777619u;
  return h;
}

static unsigned in[N], out[N], a[N], c[N];
static unsigned char stream[16 << 10];
static char why[256];

/* The ids this program gives its objects. */
enum { BUF = 1, MOD1, PIPE1, LAYOUT1, GROUP1, BA, BC, MOD2, PIPE2, LAYOUT2, GROUP2 };

/* Runs `pipe` over `groups` workgroups with `group` bound, then reads `buf` back into `dst`. */
static int run(gpu_stream *s, unsigned pipe, unsigned group, unsigned groups, unsigned buf,
               unsigned *dst) {
  gpu_begin_compute_pass(s);
  gpu_set_pipeline(s, pipe);
  gpu_set_bind_group(s, 0, group);
  gpu_dispatch(s, groups, 1, 1);
  gpu_end_pass(s);
  gpu_readback(s, buf, 0, N * 4);
  long r = gpu_submit(s);
  long n = r == 0 ? gpu_read_wait(buf, dst, N * 4) : r;
  if (gpu_error(why, sizeof why) > 0) printf("webgpu error: %s\n", why);
  return r == 0 && n == N * 4;
}

int main(void) {
  int fail = 0;
  gpu_stream s;
  gpu_stream_init(&s, stream, sizeof stream);
  const unsigned storage = GPU_BUFFER_STORAGE | GPU_BUFFER_COPY_DST | GPU_BUFFER_COPY_SRC;

  /* --- kernel 1: in-place mul-add over a single storage buffer --------------------------------- */
  for (int i = 0; i < N; i++) in[i] = (unsigned)i * 2654435761u >> 8;
  static const char sh1[] =
      "@group(0) @binding(0) var<storage, read_write> b: array<u32>;\n"
      "@compute @workgroup_size(64)\n"
      "fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n"
      "  let i = g.x;\n"
      "  if (i < arrayLength(&b)) { b[i] = b[i] * 3u + 7u; }\n"
      "}\n";
  gpu_create_buffer(&s, BUF, N * 4, storage);
  gpu_write_buffer(&s, BUF, 0, in, N * 4);
  gpu_create_shader_module(&s, MOD1, sh1);
  gpu_create_compute_pipeline(&s, PIPE1, MOD1, "main");
  gpu_bind_group_layout_of(&s, LAYOUT1, PIPE1, 0);
  gpu_bind_entry e1[] = {{0, GPU_BIND_BUFFER, BUF, 0, 0}};
  gpu_create_bind_group(&s, GROUP1, LAYOUT1, 1, e1);
  int ok1 = run(&s, PIPE1, GROUP1, (N + 63) / 64, BUF, out);
  for (int i = 0; i < N && ok1; i++) {
    if (out[i] != in[i] * 3u + 7u) ok1 = 0;
  }
  printf("compute1 (mul-add, 1 buffer): %s digest=%08x\n", ok1 ? "OK" : "FAIL", fnv(out, N));
  fail |= !ok1;

  /* --- kernel 2: two-buffer kernel c[i] = a[i]*a[i] + i ---------------------------------------- */
  for (int i = 0; i < N; i++) a[i] = (unsigned)(i * 7 + 1);
  static const char sh2[] =
      "@group(0) @binding(0) var<storage, read> a: array<u32>;\n"
      "@group(0) @binding(1) var<storage, read_write> c: array<u32>;\n"
      "@compute @workgroup_size(64)\n"
      "fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n"
      "  let i = g.x;\n"
      "  if (i < arrayLength(&a)) { c[i] = a[i] * a[i] + i; }\n"
      "}\n";
  gpu_create_buffer(&s, BA, N * 4, storage);
  gpu_create_buffer(&s, BC, N * 4, storage);
  gpu_write_buffer(&s, BA, 0, a, N * 4);
  gpu_create_shader_module(&s, MOD2, sh2);
  gpu_create_compute_pipeline(&s, PIPE2, MOD2, "main");
  gpu_bind_group_layout_of(&s, LAYOUT2, PIPE2, 0);
  gpu_bind_entry e2[] = {{0, GPU_BIND_BUFFER, BA, 0, 0}, {1, GPU_BIND_BUFFER, BC, 0, 0}};
  gpu_create_bind_group(&s, GROUP2, LAYOUT2, 2, e2);
  int ok2 = run(&s, PIPE2, GROUP2, (N + 63) / 64, BC, c);
  for (int i = 0; i < N && ok2; i++) {
    if (c[i] != a[i] * a[i] + (unsigned)i) ok2 = 0;
  }
  printf("compute2 (a*a+i, 2 buffers): %s digest=%08x\n", ok2 ? "OK" : "FAIL", fnv(c, N));
  fail |= !ok2;

  printf("webgpu compute: %s\n", fail ? "FAIL" : "ALL MATCH cpu");
  return fail ? 1 : 0;
}
