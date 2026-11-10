/* render.c — WebGPU **render → present → readback**, checked pixel for pixel (LLVM.md WebGPU demo 2;
 * temen-gpu, #2665). A guest renders one frame into the host's 64x64 surface, reads it back, and
 * compares every pixel with a model computed here on the CPU:
 *   - two rects, as instanced quads from an index buffer and a per-instance vertex buffer, added
 *     to a black clear (red, green, and yellow where they overlap);
 *   - a full-surface blue quad, scissored to the top-right 16x16;
 *   - a 4x4 texture, written from the guest and sampled nearest, scaled 4x into the bottom-left.
 * Every edge lies on a pixel boundary and every colour is a whole step of 1/255, so any conformant
 * WebGPU gives exactly these bytes, natively and in a page alike. The frame is presented, and a host
 * that sees it can compare it with the digest printed here.
 *
 * It also checks the surface, that a malformed stream is refused whole and says why, and that a WGSL
 * error comes back as a value. The surface may be RGBA or BGRA; the readback is put in RGBA order.
 *
 * One program for every host. Built with `-DGPU_REACTOR` it is a reactor: `main()` submits the frame,
 * and `tick()` asks for the readback until it arrives, as a page must, since the browser resolves a
 * readback, and reports WebGPU's errors, only between ticks. Without it, `main()` asks in a loop. */

#include <stdio.h>

#ifdef TEMEN_GUEST
#include "temen_gpu.h"
#endif

#define S 64

static gpu_stream s;
static unsigned char stream[16 << 10];
static unsigned char px[S * S * 4];
static char why[256];
static int ok = 1, stage, waited, bgra;

enum {
  UBO = 1, RECTS, INDICES, TEX, TEX_VIEW, SAMPLER, READBACK,
  SOLID_MOD, SOLID, SOLID_LAYOUT, SOLID_GROUP,
  TEXT_MOD, TEXTURED, TEXT_LAYOUT, TEXT_GROUP,
  FRAME, FRAME_VIEW, BROKEN,
};

static const char solid_wgsl[] =
    "struct U { size: vec2f };\n"
    "@group(0) @binding(0) var<uniform> u: U;\n"
    "struct Out { @builtin(position) pos: vec4f, @location(0) color: vec4f };\n"
    "@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) rect: vec4f,\n"
    "              @location(1) color: vec4f) -> Out {\n"
    "  let x = select(rect.x, rect.z, (vi & 1u) == 1u);\n"
    "  let y = select(rect.y, rect.w, (vi & 2u) == 2u);\n"
    "  var o: Out;\n"
    "  o.pos = vec4f(x / u.size.x * 2.0 - 1.0, 1.0 - y / u.size.y * 2.0, 0.0, 1.0);\n"
    "  o.color = color;\n"
    "  return o;\n"
    "}\n"
    "@fragment fn fs(i: Out) -> @location(0) vec4f { return i.color; }\n";

static const char textured_wgsl[] =
    "@group(0) @binding(0) var t: texture_2d<f32>;\n"
    "@group(0) @binding(1) var s: sampler;\n"
    "struct Out { @builtin(position) pos: vec4f, @location(0) uv: vec2f };\n"
    "@vertex fn vs(@builtin(vertex_index) vi: u32) -> Out {\n"
    "  var c = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),\n"
    "                          vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));\n"
    "  let k = c[vi];\n"
    "  let p = vec2f(0.0, 48.0) + k * 16.0;\n"
    "  var o: Out;\n"
    "  o.pos = vec4f(p.x / 32.0 - 1.0, 1.0 - p.y / 32.0, 0.0, 1.0);\n"
    "  o.uv = k;\n"
    "  return o;\n"
    "}\n"
    "@fragment fn fs(i: Out) -> @location(0) vec4f { return textureSample(t, s, i.uv); }\n";

/* One rect: its corners in pixels, then its colour as unorm8x4. */
typedef struct {
  float x0, y0, x1, y1;
  unsigned char rgba[4];
} rect;

static const rect rects[3] = {
    {8, 8, 40, 40, {255, 0, 0, 255}},
    {24, 24, 56, 56, {0, 255, 0, 255}},
    {0, 0, 64, 64, {0, 0, 255, 255}},
};

/* The pixel at (x, y), as the model says. */
static void model(int x, int y, unsigned char out[4]) {
  unsigned char r = 0, g = 0, b = 0, a = 255;
  if (x >= 8 && x < 40 && y >= 8 && y < 40) r = 255;
  if (x >= 24 && x < 56 && y >= 24 && y < 56) g = 255;
  if (x >= 48 && y < 16) b = 255;
  if (x < 16 && y >= 48) {
    r = (unsigned char)(x / 4 * 64);
    g = (unsigned char)((y - 48) / 4 * 64);
    b = 128;
  }
  out[0] = r;
  out[1] = g;
  out[2] = b;
  out[3] = a;
}

static unsigned fnv(const unsigned char *p, int n) {
  unsigned h = 2166136261u;
  for (int i = 0; i < n; i++) h = (h ^ p[i]) * 16777619u;
  return h;
}

static int contains(const char *s, const char *sub) {
  for (; *s; s++) {
    const char *a = s, *b = sub;
    while (*a && *b && *a == *b) a++, b++;
    if (!*b) return 1;
  }
  return 0;
}

/* The surface, the refused stream, the resources, and the frame with its readback. */
static void setup(void) {
  gpu_stream_init(&s, stream, sizeof stream);

  /* The surface the host bound. */
  gpu_surface_info info;
  long r = gpu_surface(&info);
  bgra = info.format == GPU_BGRA8UNORM;
  int surface_ok = r == 0 && info.width == S && info.height == S &&
                   (info.format == GPU_RGBA8UNORM || bgra);
  printf("surface: %ldx%ld format %ld: %s\n", (long)info.width, (long)info.height,
         (long)info.format, surface_ok ? "OK" : "FAIL");
  ok &= surface_ok;
  if (!surface_ok) return;

  /* A malformed stream runs nothing, and says why. */
  gpu_create_buffer(&s, UBO, 16, GPU_BUFFER_UNIFORM | GPU_BUFFER_COPY_DST);
  gpu_draw(&s, 3, 1, 0, 0);
  r = gpu_submit(&s);
  gpu_error(why, sizeof why);
  int refused = r == GPU_EINVAL && contains(why, "only inside a render pass");
  printf("refused: %s (%s)\n", refused ? "OK" : "FAIL", why);
  ok &= refused;

  /* The resources. The refused stream above made none, so UBO is free. */
  float size[4] = {S, S, 0, 0};
  unsigned short indices[6] = {0, 1, 2, 2, 1, 3};
  unsigned char texels[4 * 4 * 4];
  for (int j = 0; j < 4; j++)
    for (int i = 0; i < 4; i++) {
      unsigned char *t = &texels[(j * 4 + i) * 4];
      t[0] = (unsigned char)(i * 64);
      t[1] = (unsigned char)(j * 64);
      t[2] = 128;
      t[3] = 255;
    }
  gpu_create_buffer(&s, UBO, 16, GPU_BUFFER_UNIFORM | GPU_BUFFER_COPY_DST);
  gpu_write_buffer(&s, UBO, 0, size, sizeof size);
  gpu_create_buffer(&s, RECTS, sizeof rects, GPU_BUFFER_VERTEX | GPU_BUFFER_COPY_DST);
  gpu_write_buffer(&s, RECTS, 0, rects, sizeof rects);
  gpu_create_buffer(&s, INDICES, 16, GPU_BUFFER_INDEX | GPU_BUFFER_COPY_DST);
  gpu_write_buffer(&s, INDICES, 0, indices, 12);
  gpu_create_texture(&s, TEX, 4, 4, GPU_RGBA8UNORM, GPU_TEXTURE_BINDING | GPU_TEXTURE_COPY_DST);
  gpu_write_texture(&s, TEX, 0, 0, 4, 4, 16, texels, sizeof texels);
  gpu_create_view(&s, TEX_VIEW, TEX);
  gpu_create_sampler(&s, SAMPLER, GPU_NEAREST);
  gpu_create_buffer(&s, READBACK, S * S * 4, GPU_BUFFER_COPY_DST | GPU_BUFFER_COPY_SRC);

  /* The solid pipeline: instanced quads, added to what is there. */
  gpu_create_shader_module(&s, SOLID_MOD, solid_wgsl);
  gpu_attribute attrs[2] = {{GPU_FLOAT32X4, 0, 0}, {GPU_UNORM8X4, 1, 16}};
  gpu_vertex_buffer vb = {sizeof(rect), GPU_STEP_INSTANCE, 2, attrs};
  gpu_target add = {info.format, GPU_WRITE_ALL, 1,
                    {GPU_ONE, GPU_ONE, GPU_ADD}, {GPU_ONE, GPU_ZERO, GPU_ADD}};
  gpu_render_pipeline solid = {SOLID, SOLID_MOD, "vs", "fs", GPU_TRIANGLE_LIST, 0, GPU_CCW,
                               GPU_CULL_NONE, 1, 1, &vb, 1, &add};
  gpu_create_render_pipeline(&s, &solid);
  gpu_bind_group_layout_of(&s, SOLID_LAYOUT, SOLID, 0);
  gpu_bind_entry se[] = {{0, GPU_BIND_BUFFER, UBO, 0, 0}};
  gpu_create_bind_group(&s, SOLID_GROUP, SOLID_LAYOUT, 1, se);

  /* The textured pipeline: one quad, replacing what is there. */
  gpu_create_shader_module(&s, TEXT_MOD, textured_wgsl);
  gpu_target replace = {info.format, GPU_WRITE_ALL, 0, {0, 0, 0}, {0, 0, 0}};
  gpu_render_pipeline textured = {TEXTURED, TEXT_MOD, "vs", "fs", GPU_TRIANGLE_LIST, 0, GPU_CCW,
                                  GPU_CULL_NONE, 1, 0, 0, 1, &replace};
  gpu_create_render_pipeline(&s, &textured);
  gpu_bind_group_layout_of(&s, TEXT_LAYOUT, TEXTURED, 0);
  gpu_bind_entry te[] = {{0, GPU_BIND_VIEW, TEX_VIEW, 0, 0}, {1, GPU_BIND_SAMPLER, SAMPLER, 0, 0}};
  gpu_create_bind_group(&s, TEXT_GROUP, TEXT_LAYOUT, 2, te);

  /* The frame. */
  gpu_acquire(&s, FRAME, FRAME_VIEW);
  gpu_color_attachment color = {FRAME_VIEW, 0, GPU_LOAD_CLEAR, GPU_STORE_STORE, {0, 0, 0, 1}};
  gpu_begin_render_pass(&s, 1, &color);
  gpu_set_pipeline(&s, SOLID);
  gpu_set_bind_group(&s, 0, SOLID_GROUP);
  gpu_set_vertex_buffer(&s, 0, RECTS, 0, 0);
  gpu_set_index_buffer(&s, INDICES, GPU_INDEX_UINT16, 0, 12);
  gpu_draw_indexed(&s, 6, 2, 0, 0, 0);
  gpu_set_scissor(&s, 48, 0, 16, 16);
  gpu_draw_indexed(&s, 6, 1, 0, 0, 2);
  gpu_set_scissor(&s, 0, 0, S, S);
  gpu_set_pipeline(&s, TEXTURED);
  gpu_set_bind_group(&s, 0, TEXT_GROUP);
  gpu_draw(&s, 6, 1, 0, 0);
  gpu_end_pass(&s);
  gpu_copy_texture_to_buffer(&s, FRAME, 0, 0, S, S, READBACK, 0, S * 4);
  gpu_readback(&s, READBACK, 0, S * S * 4);
  gpu_present(&s, FRAME, FRAME_VIEW);
  r = gpu_submit(&s);
  if (r != 0) {
    gpu_error(why, sizeof why);
    printf("frame: FAIL (%ld: %s)\n", r, why);
    ok = 0;
    return;
  }
  stage = 1;
}

/* Compares the frame read back with the model, in RGBA order. */
static void check(long n) {
  int frame_ok = n == (long)sizeof px;
  int bad = 0;
  for (int i = 0; bgra && frame_ok && i < S * S; i++) {
    unsigned char b = px[i * 4];
    px[i * 4] = px[i * 4 + 2];
    px[i * 4 + 2] = b;
  }
  for (int y = 0; y < S && frame_ok; y++)
    for (int x = 0; x < S; x++) {
      unsigned char want[4];
      model(x, y, want);
      const unsigned char *got = &px[(y * S + x) * 4];
      if (got[0] != want[0] || got[1] != want[1] || got[2] != want[2] || got[3] != want[3]) {
        if (bad++ < 4)
          printf("pixel (%d, %d): got %d,%d,%d,%d want %d,%d,%d,%d\n", x, y, got[0], got[1],
                 got[2], got[3], want[0], want[1], want[2], want[3]);
      }
    }
  frame_ok &= bad == 0;
  printf("frame: %s, %d pixels differ, digest=%08x\n", frame_ok ? "OK" : "FAIL", bad,
         fnv(px, sizeof px));
  ok &= frame_ok;
}

/* One step: 1 once the program is done. Stage 1 waits for the readback and checks the frame, then
 * submits a broken WGSL module; stage 2 waits for its error. Nothing else may have reported one. */
static int step(void) {
  if (stage == 1) {
    long n = gpu_read(READBACK, px, sizeof px);
    if (n == GPU_EAGAIN) return 0;
    if (gpu_error(why, sizeof why) > 0) {
      printf("an error before the WGSL case: %s\n", why);
      ok = 0;
    }
    check(n);
    gpu_create_shader_module(&s, BROKEN, "fn broken(");
    gpu_destroy(&s, BROKEN);
    long r = gpu_submit(&s);
    ok &= r == 0;
    stage = 2;
    waited = 0;
  }
  if (stage == 2) {
    if (gpu_error(why, sizeof why) == 0 && ++waited < 600) return 0;
    printf("wgsl error reported: %s\n", waited < 600 ? "OK" : "NOT REPORTED");
    ok &= waited < 600;
    stage = 3;
  }
  if (stage == 3) {
    printf("render: %s\n", ok ? "OK" : "FAIL");
    stage = 4;
  }
  return stage >= 3;
}

int main(void) {
  setup();
  if (stage == 0) {
    printf("render: FAIL\n");
    return 1;
  }
#ifdef GPU_REACTOR
  return 0;
#else
  while (!step()) {
  }
  return ok ? 0 : 1;
#endif
}

#ifdef GPU_REACTOR
int tick(void) {
  if (stage == 1 || stage == 2) step();
  return 0;
}
#endif
