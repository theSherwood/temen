/* GPU fragment-shader demo for the playground — a Mandelbrot zoom rendered on the **GPU** through the
 * `webgpu` capability, presented to the page's <canvas>. Unlike `mandelzoom.c` (which computes every
 * pixel on the CPU in-guest), here the guest builds one render pipeline from a **WGSL fragment shader**
 * and each tick draws a frame with an updated time — the massively-parallel escape-time loop runs on
 * the GPU, so it stays smooth at full resolution.
 *
 * The capability's protocol is the one every host shares (`crates/temen-gpu/temen_gpu.h`, #2665): the
 * guest names objects by ids it chooses and submits command streams, and never holds a GPU object —
 * validation, not raw access, is the boundary (§2a). `main()` makes the pipeline for the surface's
 * size and format; the host calls `tick()` once per animation frame, which writes the frame counter
 * to a uniform, then acquires the canvas's texture, draws, and presents. Only the stream crosses the
 * boundary per frame. */

#include "temen_gpu.h"

enum { UNIFORM = 1, MODULE, PIPELINE, LAYOUT, GROUP, FRAME, FRAME_VIEW };

static gpu_stream s;
static unsigned char stream[8 << 10];
static gpu_surface_info surface;
static int live;
static int frame;

/* A fullscreen-triangle vertex shader + a Mandelbrot-zoom fragment shader. `p.frame` drives the zoom
 * (exp-decay scale toward a seahorse-valley point) and a smooth cosine-palette color cycle. All the
 * math (`exp`/`cos`/the escape loop) runs on the GPU. */
static const char shader[] =
    "struct P { frame: f32, w: f32, h: f32, pad: f32 };\n"
    "@group(0) @binding(0) var<uniform> p: P;\n"
    "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
    "  var v = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));\n"
    "  return vec4f(v[i], 0.0, 1.0);\n"
    "}\n"
    "@fragment fn fs(@builtin(position) c: vec4f) -> @location(0) vec4f {\n"
    "  let t = p.frame;\n"
    "  let scale = 1.4 * exp(-t * 0.02);\n"
    "  let center = vec2f(-0.743643887037151, 0.131825904205330);\n"
    "  let uv = (c.xy - vec2f(p.w, p.h) * 0.5) / (p.h * 0.5);\n"
    "  let cc = center + uv * scale;\n"
    "  var z = vec2f(0.0, 0.0);\n"
    "  var it = 0.0;\n"
    "  let maxit = 240.0;\n"
    "  loop {\n"
    "    if (it >= maxit) { break; }\n"
    "    z = vec2f(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + cc;\n"
    "    if (dot(z, z) > 4.0) { break; }\n"
    "    it = it + 1.0;\n"
    "  }\n"
    "  if (it >= maxit) { return vec4f(0.0, 0.0, 0.0, 1.0); }\n"
    "  let m = it / maxit;\n"
    "  let col = 0.5 + 0.5 * cos(6.28318 * (m + vec3f(0.0, 0.33, 0.67)) + t * 0.15);\n"
    "  return vec4f(col, 1.0);\n"
    "}\n";

int main(void) {
  gpu_stream_init(&s, stream, sizeof stream);
  live = gpu_surface(&surface) == 0;
  if (!live) return 0;
  gpu_create_buffer(&s, UNIFORM, 16, GPU_BUFFER_UNIFORM | GPU_BUFFER_COPY_DST);
  gpu_create_shader_module(&s, MODULE, shader);
  gpu_target target = {surface.format, GPU_WRITE_ALL, 0, {0, 0, 0}, {0, 0, 0}};
  gpu_render_pipeline p = {PIPELINE, MODULE, "vs", "fs", GPU_TRIANGLE_LIST, 0, GPU_CCW,
                           GPU_CULL_NONE, 1, 0, 0, 1, &target};
  gpu_create_render_pipeline(&s, &p);
  gpu_bind_group_layout_of(&s, LAYOUT, PIPELINE, 0);
  gpu_bind_entry e[] = {{0, GPU_BIND_BUFFER, UNIFORM, 0, 0}};
  gpu_create_bind_group(&s, GROUP, LAYOUT, 1, e);
  live = gpu_submit(&s) == 0;
  frame = 0;
  return 0;
}

int tick(void) {
  if (live) {
    float u[4] = {(float)frame, (float)surface.width, (float)surface.height, 0};
    gpu_write_buffer(&s, UNIFORM, 0, u, sizeof u);
    gpu_acquire(&s, FRAME, FRAME_VIEW);
    gpu_color_attachment color = {FRAME_VIEW, 0, GPU_LOAD_CLEAR, GPU_STORE_STORE, {0, 0, 0, 1}};
    gpu_begin_render_pass(&s, 1, &color);
    gpu_set_pipeline(&s, PIPELINE);
    gpu_set_bind_group(&s, 0, GROUP);
    gpu_draw(&s, 3, 1, 0, 0);
    gpu_end_pass(&s);
    gpu_present(&s, FRAME, FRAME_VIEW);
    live = gpu_submit(&s) == 0;
  }
  frame++;
  return 0;
}
