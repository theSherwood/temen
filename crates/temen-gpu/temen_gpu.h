/* temen_gpu.h — the `webgpu` capability's protocol for a C guest (crates/temen-gpu, #2665).
 *
 * A guest names GPU objects by ids it chooses (1 to 65535) and builds command streams in its own
 * memory, which `gpu_submit` hands to the host whole. Each `gpu_*` command below appends one record,
 * laid out exactly as `temen_gpu::command` describes; the host decodes and checks the whole stream
 * before any of it runs, so a stream with a bad id or field runs nothing, and `gpu_error` says why.
 * WebGPU validates the rest (formats, usages, layouts, WGSL) and reports through `gpu_error` too.
 *
 * Commands take effect in stream order. A readback arrives some time after its submit: `gpu_read`
 * says GPU_EAGAIN until then, on every host.
 *
 * `#include` it in a guest built for temen (`-DTEMEN_GUEST`). */
#ifndef TEMEN_GPU_H
#define TEMEN_GPU_H

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

/* The ops. */
enum { GPU_OP_SUBMIT = 0, GPU_OP_READ = 1, GPU_OP_ERROR = 2, GPU_OP_SURFACE = 3 };

/* What the ops return besides a length or 0. */
enum {
  GPU_ENOENT = -2,
  GPU_E2BIG = -7, /* the stream outgrew its buffer: nothing was submitted */
  GPU_EAGAIN = -11,
  GPU_EFAULT = -14,
  GPU_ENODEV = -19,
  GPU_EINVAL = -22,
};

/* GPUBufferUsage, GPUTextureUsage and GPUColorWrite, as WebGPU numbers them. */
enum {
  GPU_BUFFER_COPY_SRC = 0x0004,
  GPU_BUFFER_COPY_DST = 0x0008,
  GPU_BUFFER_INDEX = 0x0010,
  GPU_BUFFER_VERTEX = 0x0020,
  GPU_BUFFER_UNIFORM = 0x0040,
  GPU_BUFFER_STORAGE = 0x0080,
};
enum {
  GPU_TEXTURE_COPY_SRC = 0x01,
  GPU_TEXTURE_COPY_DST = 0x02,
  GPU_TEXTURE_BINDING = 0x04,
  GPU_TEXTURE_STORAGE = 0x08,
  GPU_TEXTURE_RENDER_ATTACHMENT = 0x10,
};
enum { GPU_WRITE_RED = 1, GPU_WRITE_GREEN = 2, GPU_WRITE_BLUE = 4, GPU_WRITE_ALPHA = 8, GPU_WRITE_ALL = 15 };

/* The enumerations' codes (temen_gpu::enums). Where 0 is allowed it means none. */
enum {
  GPU_R8UNORM = 1, GPU_RG8UNORM, GPU_RGBA8UNORM, GPU_RGBA8UNORM_SRGB, GPU_BGRA8UNORM,
  GPU_BGRA8UNORM_SRGB, GPU_R16FLOAT, GPU_RG16FLOAT, GPU_RGBA16FLOAT, GPU_R32FLOAT, GPU_RG32FLOAT,
  GPU_RGBA32FLOAT, GPU_R32UINT, GPU_RGB10A2UNORM,
};
enum { GPU_DIM_1D = 1, GPU_DIM_2D, GPU_DIM_3D };
enum {
  GPU_FLOAT32 = 1, GPU_FLOAT32X2, GPU_FLOAT32X3, GPU_FLOAT32X4, GPU_UINT32, GPU_UINT32X2,
  GPU_UINT32X3, GPU_UINT32X4, GPU_SINT32, GPU_SINT32X2, GPU_SINT32X3, GPU_SINT32X4, GPU_UNORM8X4,
  GPU_UINT8X4, GPU_UINT16X2, GPU_SINT16X2, GPU_UNORM16X2, GPU_FLOAT16X2, GPU_FLOAT16X4,
};
enum { GPU_STEP_VERTEX = 1, GPU_STEP_INSTANCE };
enum { GPU_POINT_LIST = 1, GPU_LINE_LIST, GPU_LINE_STRIP, GPU_TRIANGLE_LIST, GPU_TRIANGLE_STRIP };
enum { GPU_INDEX_UINT16 = 1, GPU_INDEX_UINT32 };
enum { GPU_CCW = 1, GPU_CW };
enum { GPU_CULL_NONE = 0, GPU_CULL_FRONT, GPU_CULL_BACK };
enum {
  GPU_ZERO = 1, GPU_ONE, GPU_SRC, GPU_ONE_MINUS_SRC, GPU_SRC_ALPHA, GPU_ONE_MINUS_SRC_ALPHA, GPU_DST,
  GPU_ONE_MINUS_DST, GPU_DST_ALPHA, GPU_ONE_MINUS_DST_ALPHA, GPU_SRC_ALPHA_SATURATED,
};
enum { GPU_ADD = 1, GPU_SUBTRACT, GPU_REVERSE_SUBTRACT, GPU_MIN, GPU_MAX };
enum { GPU_LOAD_CLEAR = 1, GPU_LOAD_LOAD };
enum { GPU_STORE_STORE = 1, GPU_STORE_DISCARD };
enum { GPU_CLAMP_TO_EDGE = 1, GPU_REPEAT, GPU_MIRROR_REPEAT };
enum { GPU_NEAREST = 1, GPU_LINEAR };
enum { GPU_BIND_BUFFER = 1, GPU_BIND_SAMPLER, GPU_BIND_VIEW };

/* A stream being built, in a buffer the guest owns. */
typedef struct {
  unsigned char *p;
  unsigned long len, cap;
  int overflow;
} gpu_stream;

static void gpu_stream_init(gpu_stream *s, void *buf, unsigned long cap) {
  s->p = (unsigned char *)buf;
  s->len = 0;
  s->cap = cap;
  s->overflow = 0;
}

static int gpu__h = -2;
static int gpu_cap(void) {
  if (gpu__h == -2) gpu__h = __vm_cap_resolve("webgpu", 6);
  return gpu__h;
}

/* The encoder's only code, kept out of line: a command is its record's C struct and one or a few
 * calls. So a guest that issues many commands stays small, which is what the JIT compiles before a
 * run (a guest encoding field by field was half as large again). temen is little-endian, so a
 * struct's bytes are its wire form. */

/* Appends `n` bytes and zeros to the next multiple of 8, or marks the stream overflowed. */
__attribute__((noinline)) static void gpu__put(gpu_stream *s, const void *p, unsigned long n) {
  unsigned long pad = (8 - n % 8) % 8;
  if (s->overflow || n + pad > s->cap - s->len) {
    s->overflow = 1;
    return;
  }
  __builtin_memcpy(s->p + s->len, p, n);
  __builtin_memset(s->p + s->len + n, 0, pad);
  s->len += n + pad;
}
/* Opens a record: its header, with the size `gpu__close` patches in. */
__attribute__((noinline)) static unsigned long gpu__open(gpu_stream *s, unsigned cmd) {
  unsigned h[2] = {cmd, 0};
  unsigned long at = s->len;
  gpu__put(s, h, sizeof h);
  return at;
}
__attribute__((noinline)) static void gpu__close(gpu_stream *s, unsigned long at) {
  if (s->overflow) return;
  unsigned size = (unsigned)(s->len - at - 8);
  __builtin_memcpy(s->p + at + 4, &size, 4);
}
/* A record that is one fixed part, the most of them. */
__attribute__((noinline)) static void gpu__record(gpu_stream *s, unsigned cmd, const void *p,
                                                  unsigned long n) {
  unsigned long at = gpu__open(s, cmd);
  gpu__put(s, p, n);
  gpu__close(s, at);
}
static unsigned long gpu__strlen(const char *z) {
  unsigned long n = 0;
  while (z[n]) n++;
  return n;
}

/* --- Objects --- */

static void gpu_create_buffer(gpu_stream *s, unsigned id, unsigned long long size, unsigned usage) {
  struct { unsigned id, usage; unsigned long long size; } r = {id, usage, size};
  gpu__record(s, 1, &r, sizeof r);
}

/* A 2D texture of one mip and one sample, which is what most guests want. */
static void gpu_create_texture(gpu_stream *s, unsigned id, unsigned width, unsigned height,
                               unsigned format, unsigned usage) {
  unsigned r[10] = {id, format, usage, GPU_DIM_2D, width, height, 1, 1, 1, 0};
  gpu__record(s, 2, r, sizeof r);
}

static void gpu_create_view(gpu_stream *s, unsigned id, unsigned texture) {
  unsigned r[2] = {id, texture};
  gpu__record(s, 3, r, sizeof r);
}

/* A sampler that clamps to the edge, with one filter for magnifying and minifying. */
static void gpu_create_sampler(gpu_stream *s, unsigned id, unsigned filter) {
  struct { unsigned v[9]; float lod_min, lod_max; unsigned pad; } r = {
      {id, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE, filter, filter, GPU_NEAREST, 0, 1},
      0.0f, 32.0f, 0};
  gpu__record(s, 4, &r, sizeof r);
}

static void gpu_create_shader_module(gpu_stream *s, unsigned id, const char *wgsl) {
  unsigned long n = gpu__strlen(wgsl);
  unsigned r[2] = {id, (unsigned)n};
  unsigned long at = gpu__open(s, 5);
  gpu__put(s, r, sizeof r);
  gpu__put(s, wgsl, n);
  gpu__close(s, at);
}

/* An attribute is laid out as the wire's `{format, location, u64 offset}`. */
typedef struct {
  unsigned format, location;
  unsigned long long offset;
} gpu_attribute;

typedef struct {
  unsigned long long stride;
  unsigned step, n_attributes;
  const gpu_attribute *attributes;
} gpu_vertex_buffer;

typedef struct {
  unsigned src, dst, op;
} gpu_blend_component;

/* A colour target; `blend` 0 writes the colour as it is, 1 blends with `color` and `alpha`. */
typedef struct {
  unsigned format, write_mask, blend;
  gpu_blend_component color, alpha;
} gpu_target;

typedef struct {
  unsigned id, module;
  const char *vertex, *fragment;
  unsigned topology, strip_index_format, front_face, cull, samples;
  unsigned n_buffers;
  const gpu_vertex_buffer *buffers;
  unsigned n_targets;
  const gpu_target *targets;
} gpu_render_pipeline;

static void gpu_create_render_pipeline(gpu_stream *s, const gpu_render_pipeline *p) {
  unsigned long vn = gpu__strlen(p->vertex), fn = gpu__strlen(p->fragment);
  unsigned v[12] = {p->id, p->module, p->topology, p->strip_index_format, p->front_face, p->cull,
                    p->samples, p->n_buffers, p->n_targets, (unsigned)vn, (unsigned)fn, 0};
  unsigned long at = gpu__open(s, 6);
  gpu__put(s, v, sizeof v);
  gpu__put(s, p->vertex, vn);
  gpu__put(s, p->fragment, fn);
  for (unsigned i = 0; i < p->n_buffers; i++) {
    const gpu_vertex_buffer *vb = &p->buffers[i];
    struct { unsigned long long stride; unsigned step, n; } b = {vb->stride, vb->step,
                                                               vb->n_attributes};
    gpu__put(s, &b, sizeof b);
    gpu__put(s, vb->attributes, vb->n_attributes * sizeof(gpu_attribute));
  }
  for (unsigned i = 0; i < p->n_targets; i++) {
    const gpu_target *g = &p->targets[i];
    unsigned on = g->blend ? 1 : 0;
    unsigned w[10] = {g->format, g->write_mask, on,
                      on ? g->color.src : 0, on ? g->color.dst : 0, on ? g->color.op : 0,
                      on ? g->alpha.src : 0, on ? g->alpha.dst : 0, on ? g->alpha.op : 0, 0};
    gpu__put(s, w, sizeof w);
  }
  gpu__close(s, at);
}

static void gpu_create_compute_pipeline(gpu_stream *s, unsigned id, unsigned module,
                                        const char *entry) {
  unsigned long n = gpu__strlen(entry);
  unsigned r[4] = {id, module, (unsigned)n, 0};
  unsigned long at = gpu__open(s, 7);
  gpu__put(s, r, sizeof r);
  gpu__put(s, entry, n);
  gpu__close(s, at);
}

/* Names the layout `pipeline` has for its bind group `group` (WebGPU's "auto" layout). */
static void gpu_bind_group_layout_of(gpu_stream *s, unsigned id, unsigned pipeline, unsigned group) {
  unsigned r[4] = {id, pipeline, group, 0};
  gpu__record(s, 8, r, sizeof r);
}

/* A bind group entry: a buffer range (a size of 0 is the rest), a sampler or a view. */
typedef struct {
  unsigned binding, kind, resource;
  unsigned long long offset, size;
} gpu_bind_entry;

static void gpu_create_bind_group(gpu_stream *s, unsigned id, unsigned layout, unsigned n,
                                  const gpu_bind_entry *e) {
  unsigned r[4] = {id, layout, n, 0};
  unsigned long at = gpu__open(s, 9);
  gpu__put(s, r, sizeof r);
  for (unsigned i = 0; i < n; i++) {
    int buffer = e[i].kind == GPU_BIND_BUFFER;
    struct { unsigned binding, kind, resource, pad; unsigned long long offset, size; } w = {
        e[i].binding, e[i].kind, e[i].resource, 0, buffer ? e[i].offset : 0,
        buffer ? e[i].size : 0};
    gpu__put(s, &w, sizeof w);
  }
  gpu__close(s, at);
}

static void gpu_destroy(gpu_stream *s, unsigned id) {
  unsigned r[2] = {id, 0};
  gpu__record(s, 29, r, sizeof r);
}

/* --- The queue --- */

static void gpu_write_buffer(gpu_stream *s, unsigned buffer, unsigned long long offset,
                             const void *data, unsigned long n) {
  struct { unsigned buffer, len; unsigned long long offset; } r = {buffer, (unsigned)n, offset};
  unsigned long at = gpu__open(s, 10);
  gpu__put(s, &r, sizeof r);
  gpu__put(s, data, n);
  gpu__close(s, at);
}

/* Writes `width` x `height` texels of a 2D texture's first mip at (x, y). */
static void gpu_write_texture(gpu_stream *s, unsigned texture, unsigned x, unsigned y,
                              unsigned width, unsigned height, unsigned bytes_per_row,
                              const void *data, unsigned long n) {
  unsigned r[12] = {texture, 0, x, y, 0, width, height, 1, bytes_per_row, height, (unsigned)n, 0};
  unsigned long at = gpu__open(s, 11);
  gpu__put(s, r, sizeof r);
  gpu__put(s, data, n);
  gpu__close(s, at);
}

/* Copies `size` bytes of `buffer` back to the host after everything before it; `gpu_read` takes
 * them. The buffer needs GPU_BUFFER_COPY_SRC, and the offset and size are multiples of 4. */
static void gpu_readback(gpu_stream *s, unsigned buffer, unsigned long long offset,
                         unsigned long long size) {
  struct { unsigned buffer, pad; unsigned long long offset, size; } r = {buffer, 0, offset, size};
  gpu__record(s, 26, &r, sizeof r);
}

/* --- Passes --- */

/* A colour attachment is laid out as the wire's `{view, resolve, load, store, f64 clear[4]}`. */
typedef struct {
  unsigned view, resolve, load, store;
  double clear[4];
} gpu_color_attachment;

static void gpu_begin_render_pass(gpu_stream *s, unsigned n, const gpu_color_attachment *a) {
  unsigned r[2] = {n, 0};
  unsigned long at = gpu__open(s, 12);
  gpu__put(s, r, sizeof r);
  gpu__put(s, a, n * sizeof *a);
  gpu__close(s, at);
}

static void gpu_begin_compute_pass(gpu_stream *s) { gpu__record(s, 13, 0, 0); }

static void gpu_set_pipeline(gpu_stream *s, unsigned pipeline) {
  unsigned r[2] = {pipeline, 0};
  gpu__record(s, 14, r, sizeof r);
}

static void gpu_set_bind_group(gpu_stream *s, unsigned index, unsigned group) {
  unsigned r[2] = {index, group};
  gpu__record(s, 15, r, sizeof r);
}

static void gpu_set_vertex_buffer(gpu_stream *s, unsigned slot, unsigned buffer,
                                  unsigned long long offset, unsigned long long size) {
  struct { unsigned slot, buffer; unsigned long long offset, size; } r = {slot, buffer, offset, size};
  gpu__record(s, 16, &r, sizeof r);
}

static void gpu_set_index_buffer(gpu_stream *s, unsigned buffer, unsigned format,
                                 unsigned long long offset, unsigned long long size) {
  struct { unsigned buffer, format; unsigned long long offset, size; } r = {buffer, format, offset,
                                                                            size};
  gpu__record(s, 17, &r, sizeof r);
}

static void gpu_set_scissor(gpu_stream *s, unsigned x, unsigned y, unsigned w, unsigned h) {
  unsigned r[4] = {x, y, w, h};
  gpu__record(s, 18, r, sizeof r);
}

static void gpu_draw(gpu_stream *s, unsigned vertices, unsigned instances, unsigned first_vertex,
                     unsigned first_instance) {
  unsigned r[4] = {vertices, instances, first_vertex, first_instance};
  gpu__record(s, 19, r, sizeof r);
}

static void gpu_draw_indexed(gpu_stream *s, unsigned indices, unsigned instances,
                             unsigned first_index, int base_vertex, unsigned first_instance) {
  unsigned r[6] = {indices, instances, first_index, (unsigned)base_vertex, first_instance, 0};
  gpu__record(s, 20, r, sizeof r);
}

static void gpu_dispatch(gpu_stream *s, unsigned x, unsigned y, unsigned z) {
  unsigned r[4] = {x, y, z, 0};
  gpu__record(s, 21, r, sizeof r);
}

static void gpu_end_pass(gpu_stream *s) { gpu__record(s, 22, 0, 0); }

/* --- Copies --- */

static void gpu_copy_buffer_to_buffer(gpu_stream *s, unsigned src, unsigned long long src_offset,
                                      unsigned dst, unsigned long long dst_offset,
                                      unsigned long long size) {
  struct { unsigned src, dst; unsigned long long src_offset, dst_offset, size; } r = {
      src, dst, src_offset, dst_offset, size};
  gpu__record(s, 23, &r, sizeof r);
}

/* A buffer copy `{buffer, bytes_per_row, rows_per_image, 0, u64 offset}`, a texture copy
 * `{texture, mip, x, y, z, 0}` and an extent `{width, height, depth, 0}`, as the wire has them. */
static void gpu__copy(gpu_stream *s, unsigned cmd, unsigned buffer, unsigned long long offset,
                      unsigned bytes_per_row, unsigned texture, unsigned x, unsigned y,
                      unsigned width, unsigned height) {
  struct { unsigned buffer, bytes_per_row, rows_per_image, pad; unsigned long long offset;
           unsigned t[10]; } r = {buffer, bytes_per_row, height, 0, offset,
                                  {texture, 0, x, y, 0, 0, width, height, 1, 0}};
  gpu__record(s, cmd, &r, sizeof r);
}

/* Copies a `width` x `height` region of a 2D texture's first mip, at (x, y), to or from a buffer
 * whose rows are `bytes_per_row` apart (a multiple of 256, as WebGPU asks). */
static void gpu_copy_buffer_to_texture(gpu_stream *s, unsigned buffer, unsigned long long offset,
                                       unsigned bytes_per_row, unsigned texture, unsigned x,
                                       unsigned y, unsigned width, unsigned height) {
  gpu__copy(s, 24, buffer, offset, bytes_per_row, texture, x, y, width, height);
}
static void gpu_copy_texture_to_buffer(gpu_stream *s, unsigned texture, unsigned x, unsigned y,
                                       unsigned width, unsigned height, unsigned buffer,
                                       unsigned long long offset, unsigned bytes_per_row) {
  gpu__copy(s, 25, buffer, offset, bytes_per_row, texture, x, y, width, height);
}

/* --- The surface --- */

/* The surface's texture for this frame as `texture`, and its view as `view`. */
static void gpu_acquire(gpu_stream *s, unsigned texture, unsigned view) {
  unsigned r[2] = {texture, view};
  gpu__record(s, 27, r, sizeof r);
}

/* Presents the acquired texture; its two ids are free again. */
static void gpu_present(gpu_stream *s, unsigned texture, unsigned view) {
  unsigned r[2] = {texture, view};
  gpu__record(s, 28, r, sizeof r);
}

/* --- The ops --- */

/* Submits the stream and empties it: 0, or a negative errno (GPU_E2BIG if it overflowed). */
static long gpu_submit(gpu_stream *s) {
  long r;
  if (s->overflow) {
    r = GPU_E2BIG;
  } else {
    r = __vm_host_call(gpu_cap(), GPU_OP_SUBMIT, (long)s->p, (long)s->len, 0, 0);
  }
  s->len = 0;
  s->overflow = 0;
  return r;
}

/* The readback of `buffer`: its length, with up to `len` bytes copied to `dst`; GPU_EAGAIN until it
 * arrives; GPU_ENOENT with none asked for. */
static long gpu_read(unsigned buffer, void *dst, long len) {
  return __vm_host_call(gpu_cap(), GPU_OP_READ, buffer, (long)dst, len, 0);
}

/* For a guest that may wait where it runs: asks until the readback arrives. (A reactor in a page
 * asks on its next tick instead; #2838 makes the wait a park.) */
static long gpu_read_wait(unsigned buffer, void *dst, long len) {
  long r;
  while ((r = gpu_read(buffer, dst, len)) == GPU_EAGAIN) {
  }
  return r;
}

/* The first error since the last call, as text, NUL-terminated in `dst`: its length, or 0. */
static long gpu_error(char *dst, long cap) {
  long n = __vm_host_call(gpu_cap(), GPU_OP_ERROR, (long)dst, cap > 0 ? cap - 1 : 0, 0, 0);
  if (cap > 0) dst[n < 0 ? 0 : (n < cap - 1 ? n : cap - 1)] = 0;
  return n;
}

typedef struct {
  unsigned width, height, format, reserved;
} gpu_surface_info;

/* The surface the host bound: 0, or GPU_ENODEV with none. */
static long gpu_surface(gpu_surface_info *out) {
  return __vm_host_call(gpu_cap(), GPU_OP_SURFACE, (long)out, 0, 0, 0);
}

#endif
