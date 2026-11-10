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

/* Appends `n` bytes, or marks the stream overflowed. temen is little-endian, so a scalar's bytes are
 * its wire form; one bulk copy keeps a large upload cheap on every engine. */
static void gpu__put(gpu_stream *s, const void *p, unsigned long n) {
  if (s->overflow || n > s->cap - s->len) {
    s->overflow = 1;
    return;
  }
  __builtin_memcpy(s->p + s->len, p, n);
  s->len += n;
}
static void gpu__u32(gpu_stream *s, unsigned v) { gpu__put(s, &v, 4); }
static void gpu__u64(gpu_stream *s, unsigned long long v) { gpu__put(s, &v, 8); }
static void gpu__f32(gpu_stream *s, float v) { gpu__put(s, &v, 4); }
static void gpu__f64(gpu_stream *s, double v) { gpu__put(s, &v, 8); }
static void gpu__pad(gpu_stream *s) {
  static const unsigned char zero[8];
  gpu__put(s, zero, (8 - s->len % 8) % 8);
}
static void gpu__bytes(gpu_stream *s, const void *p, unsigned long n) {
  gpu__put(s, p, n);
  gpu__pad(s);
}
static unsigned long gpu__strlen(const char *z) {
  unsigned long n = 0;
  while (z[n]) n++;
  return n;
}
/* Opens a record: its header, with the size patched in by `gpu__close`. */
static unsigned long gpu__open(gpu_stream *s, unsigned cmd) {
  unsigned long at = s->len;
  gpu__u32(s, cmd);
  gpu__u32(s, 0);
  return at;
}
static void gpu__close(gpu_stream *s, unsigned long at) {
  if (s->overflow) return;
  unsigned size = (unsigned)(s->len - at - 8);
  for (int i = 0; i < 4; i++) s->p[at + 4 + i] = (unsigned char)(size >> (8 * i));
}

/* --- Objects --- */

static void gpu_create_buffer(gpu_stream *s, unsigned id, unsigned long long size, unsigned usage) {
  unsigned long at = gpu__open(s, 1);
  gpu__u32(s, id);
  gpu__u32(s, usage);
  gpu__u64(s, size);
  gpu__close(s, at);
}

/* A 2D texture of one mip and one sample, which is what most guests want. */
static void gpu_create_texture(gpu_stream *s, unsigned id, unsigned width, unsigned height,
                               unsigned format, unsigned usage) {
  unsigned long at = gpu__open(s, 2);
  unsigned v[10] = {id, format, usage, GPU_DIM_2D, width, height, 1, 1, 1, 0};
  for (int i = 0; i < 10; i++) gpu__u32(s, v[i]);
  gpu__close(s, at);
}

static void gpu_create_view(gpu_stream *s, unsigned id, unsigned texture) {
  unsigned long at = gpu__open(s, 3);
  gpu__u32(s, id);
  gpu__u32(s, texture);
  gpu__close(s, at);
}

/* A sampler that clamps to the edge, with one filter for magnifying and minifying. */
static void gpu_create_sampler(gpu_stream *s, unsigned id, unsigned filter) {
  unsigned long at = gpu__open(s, 4);
  unsigned v[9] = {id, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE,
                   filter, filter, GPU_NEAREST, 0, 1};
  for (int i = 0; i < 9; i++) gpu__u32(s, v[i]);
  gpu__f32(s, 0.0f);
  gpu__f32(s, 32.0f);
  gpu__u32(s, 0);
  gpu__close(s, at);
}

static void gpu_create_shader_module(gpu_stream *s, unsigned id, const char *wgsl) {
  unsigned long n = gpu__strlen(wgsl);
  unsigned long at = gpu__open(s, 5);
  gpu__u32(s, id);
  gpu__u32(s, (unsigned)n);
  gpu__bytes(s, wgsl, n);
  gpu__close(s, at);
}

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
  unsigned long at = gpu__open(s, 6);
  unsigned v[12] = {p->id, p->module, p->topology, p->strip_index_format, p->front_face, p->cull,
                    p->samples, p->n_buffers, p->n_targets, (unsigned)vn, (unsigned)fn, 0};
  for (int i = 0; i < 12; i++) gpu__u32(s, v[i]);
  gpu__bytes(s, p->vertex, vn);
  gpu__bytes(s, p->fragment, fn);
  for (unsigned b = 0; b < p->n_buffers; b++) {
    const gpu_vertex_buffer *vb = &p->buffers[b];
    gpu__u64(s, vb->stride);
    gpu__u32(s, vb->step);
    gpu__u32(s, vb->n_attributes);
    for (unsigned a = 0; a < vb->n_attributes; a++) {
      gpu__u32(s, vb->attributes[a].format);
      gpu__u32(s, vb->attributes[a].location);
      gpu__u64(s, vb->attributes[a].offset);
    }
  }
  for (unsigned t = 0; t < p->n_targets; t++) {
    const gpu_target *g = &p->targets[t];
    unsigned on = g->blend ? 1 : 0;
    unsigned w[10] = {g->format, g->write_mask, on,
                      on ? g->color.src : 0, on ? g->color.dst : 0, on ? g->color.op : 0,
                      on ? g->alpha.src : 0, on ? g->alpha.dst : 0, on ? g->alpha.op : 0, 0};
    for (int i = 0; i < 10; i++) gpu__u32(s, w[i]);
  }
  gpu__close(s, at);
}

static void gpu_create_compute_pipeline(gpu_stream *s, unsigned id, unsigned module,
                                        const char *entry) {
  unsigned long n = gpu__strlen(entry);
  unsigned long at = gpu__open(s, 7);
  gpu__u32(s, id);
  gpu__u32(s, module);
  gpu__u32(s, (unsigned)n);
  gpu__u32(s, 0);
  gpu__bytes(s, entry, n);
  gpu__close(s, at);
}

/* Names the layout `pipeline` has for its bind group `group` (WebGPU's "auto" layout). */
static void gpu_bind_group_layout_of(gpu_stream *s, unsigned id, unsigned pipeline, unsigned group) {
  unsigned long at = gpu__open(s, 8);
  gpu__u32(s, id);
  gpu__u32(s, pipeline);
  gpu__u32(s, group);
  gpu__u32(s, 0);
  gpu__close(s, at);
}

/* A bind group entry: a buffer range (a size of 0 is the rest), a sampler or a view. */
typedef struct {
  unsigned binding, kind, resource;
  unsigned long long offset, size;
} gpu_bind_entry;

static void gpu_create_bind_group(gpu_stream *s, unsigned id, unsigned layout, unsigned n,
                                  const gpu_bind_entry *e) {
  unsigned long at = gpu__open(s, 9);
  gpu__u32(s, id);
  gpu__u32(s, layout);
  gpu__u32(s, n);
  gpu__u32(s, 0);
  for (unsigned i = 0; i < n; i++) {
    gpu__u32(s, e[i].binding);
    gpu__u32(s, e[i].kind);
    gpu__u32(s, e[i].resource);
    gpu__u32(s, 0);
    gpu__u64(s, e[i].kind == GPU_BIND_BUFFER ? e[i].offset : 0);
    gpu__u64(s, e[i].kind == GPU_BIND_BUFFER ? e[i].size : 0);
  }
  gpu__close(s, at);
}

static void gpu_destroy(gpu_stream *s, unsigned id) {
  unsigned long at = gpu__open(s, 29);
  gpu__u32(s, id);
  gpu__u32(s, 0);
  gpu__close(s, at);
}

/* --- The queue --- */

static void gpu_write_buffer(gpu_stream *s, unsigned buffer, unsigned long long offset,
                             const void *data, unsigned long n) {
  unsigned long at = gpu__open(s, 10);
  gpu__u32(s, buffer);
  gpu__u32(s, (unsigned)n);
  gpu__u64(s, offset);
  gpu__bytes(s, data, n);
  gpu__close(s, at);
}

/* Writes `width` x `height` texels of a 2D texture's first mip at (x, y). */
static void gpu_write_texture(gpu_stream *s, unsigned texture, unsigned x, unsigned y,
                              unsigned width, unsigned height, unsigned bytes_per_row,
                              const void *data, unsigned long n) {
  unsigned long at = gpu__open(s, 11);
  unsigned v[12] = {texture, 0, x, y, 0, width, height, 1, bytes_per_row, height, (unsigned)n, 0};
  for (int i = 0; i < 12; i++) gpu__u32(s, v[i]);
  gpu__bytes(s, data, n);
  gpu__close(s, at);
}

/* Copies `size` bytes of `buffer` back to the host after everything before it; `gpu_read` takes
 * them. The buffer needs GPU_BUFFER_COPY_SRC, and the offset and size are multiples of 4. */
static void gpu_readback(gpu_stream *s, unsigned buffer, unsigned long long offset,
                         unsigned long long size) {
  unsigned long at = gpu__open(s, 26);
  gpu__u32(s, buffer);
  gpu__u32(s, 0);
  gpu__u64(s, offset);
  gpu__u64(s, size);
  gpu__close(s, at);
}

/* --- Passes --- */

typedef struct {
  unsigned view, resolve, load, store;
  double clear[4];
} gpu_color_attachment;

static void gpu_begin_render_pass(gpu_stream *s, unsigned n, const gpu_color_attachment *a) {
  unsigned long at = gpu__open(s, 12);
  gpu__u32(s, n);
  gpu__u32(s, 0);
  for (unsigned i = 0; i < n; i++) {
    gpu__u32(s, a[i].view);
    gpu__u32(s, a[i].resolve);
    gpu__u32(s, a[i].load);
    gpu__u32(s, a[i].store);
    for (int k = 0; k < 4; k++) gpu__f64(s, a[i].clear[k]);
  }
  gpu__close(s, at);
}

static void gpu_begin_compute_pass(gpu_stream *s) { gpu__close(s, gpu__open(s, 13)); }

static void gpu_set_pipeline(gpu_stream *s, unsigned pipeline) {
  unsigned long at = gpu__open(s, 14);
  gpu__u32(s, pipeline);
  gpu__u32(s, 0);
  gpu__close(s, at);
}

static void gpu_set_bind_group(gpu_stream *s, unsigned index, unsigned group) {
  unsigned long at = gpu__open(s, 15);
  gpu__u32(s, index);
  gpu__u32(s, group);
  gpu__close(s, at);
}

static void gpu_set_vertex_buffer(gpu_stream *s, unsigned slot, unsigned buffer,
                                  unsigned long long offset, unsigned long long size) {
  unsigned long at = gpu__open(s, 16);
  gpu__u32(s, slot);
  gpu__u32(s, buffer);
  gpu__u64(s, offset);
  gpu__u64(s, size);
  gpu__close(s, at);
}

static void gpu_set_index_buffer(gpu_stream *s, unsigned buffer, unsigned format,
                                 unsigned long long offset, unsigned long long size) {
  unsigned long at = gpu__open(s, 17);
  gpu__u32(s, buffer);
  gpu__u32(s, format);
  gpu__u64(s, offset);
  gpu__u64(s, size);
  gpu__close(s, at);
}

static void gpu_set_scissor(gpu_stream *s, unsigned x, unsigned y, unsigned w, unsigned h) {
  unsigned long at = gpu__open(s, 18);
  gpu__u32(s, x);
  gpu__u32(s, y);
  gpu__u32(s, w);
  gpu__u32(s, h);
  gpu__close(s, at);
}

static void gpu_draw(gpu_stream *s, unsigned vertices, unsigned instances, unsigned first_vertex,
                     unsigned first_instance) {
  unsigned long at = gpu__open(s, 19);
  gpu__u32(s, vertices);
  gpu__u32(s, instances);
  gpu__u32(s, first_vertex);
  gpu__u32(s, first_instance);
  gpu__close(s, at);
}

static void gpu_draw_indexed(gpu_stream *s, unsigned indices, unsigned instances,
                             unsigned first_index, int base_vertex, unsigned first_instance) {
  unsigned long at = gpu__open(s, 20);
  gpu__u32(s, indices);
  gpu__u32(s, instances);
  gpu__u32(s, first_index);
  gpu__u32(s, (unsigned)base_vertex);
  gpu__u32(s, first_instance);
  gpu__u32(s, 0);
  gpu__close(s, at);
}

static void gpu_dispatch(gpu_stream *s, unsigned x, unsigned y, unsigned z) {
  unsigned long at = gpu__open(s, 21);
  gpu__u32(s, x);
  gpu__u32(s, y);
  gpu__u32(s, z);
  gpu__u32(s, 0);
  gpu__close(s, at);
}

static void gpu_end_pass(gpu_stream *s) { gpu__close(s, gpu__open(s, 22)); }

/* --- Copies --- */

static void gpu_copy_buffer_to_buffer(gpu_stream *s, unsigned src, unsigned long long src_offset,
                                      unsigned dst, unsigned long long dst_offset,
                                      unsigned long long size) {
  unsigned long at = gpu__open(s, 23);
  gpu__u32(s, src);
  gpu__u32(s, dst);
  gpu__u64(s, src_offset);
  gpu__u64(s, dst_offset);
  gpu__u64(s, size);
  gpu__close(s, at);
}

static void gpu__copy(gpu_stream *s, unsigned cmd, unsigned buffer, unsigned long long offset,
                      unsigned bytes_per_row, unsigned texture, unsigned x, unsigned y,
                      unsigned width, unsigned height) {
  unsigned long at = gpu__open(s, cmd);
  gpu__u32(s, buffer);
  gpu__u32(s, bytes_per_row);
  gpu__u32(s, height);
  gpu__u32(s, 0);
  gpu__u64(s, offset);
  unsigned v[10] = {texture, 0, x, y, 0, 0, width, height, 1, 0};
  for (int i = 0; i < 10; i++) gpu__u32(s, v[i]);
  gpu__close(s, at);
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
  unsigned long at = gpu__open(s, 27);
  gpu__u32(s, texture);
  gpu__u32(s, view);
  gpu__close(s, at);
}

/* Presents the acquired texture; its two ids are free again. */
static void gpu_present(gpu_stream *s, unsigned texture, unsigned view) {
  unsigned long at = gpu__open(s, 28);
  gpu__u32(s, texture);
  gpu__u32(s, view);
  gpu__close(s, at);
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
