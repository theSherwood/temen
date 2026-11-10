//! The command stream: what a guest submits, how it is laid out, and its total decoder.
//!
//! A stream is a run of records, each an 8-byte header `{u32 command, u32 size}` and `size` bytes of
//! payload. Every field is little-endian, at its natural alignment in a C struct, and every record
//! and every variable-length part is padded to 8 bytes with zeros. So a guest may build a record by
//! copying a C struct or byte by byte, and the layouts below are the C ones. A padding byte that is
//! not zero, a code no table names, or a payload of the wrong size refuses the whole stream: nothing
//! in it runs.

use alloc::vec::Vec;

use crate::enums::*;
use crate::Id;

/// The largest stream one submit takes: room for a 4K RGBA upload several times over, and a bound on
/// what the host copies out of the window.
pub const MAX_STREAM: usize = 1 << 28;
/// The colour attachments of a pass and the targets of a pipeline, as WebGPU's
/// `maxColorAttachments`.
pub const MAX_COLORS: usize = 8;
/// A pipeline's vertex buffers, as `maxVertexBuffers`.
pub const MAX_VERTEX_BUFFERS: usize = 8;
/// A vertex buffer's attributes, as `maxVertexAttributes`.
pub const MAX_ATTRIBUTES: usize = 16;
/// A bind group's entries: far above what a pipeline uses, and a bound on the record.
pub const MAX_BIND_ENTRIES: usize = 64;
/// The bind groups of a pipeline, as `maxBindGroups`.
pub const MAX_BIND_GROUPS: u32 = 4;

/// One command. Bytes and strings borrow from the stream, which the host has already copied out
/// of the window, so nothing a guest still writes is read twice.
#[derive(Clone, Debug, PartialEq)]
pub enum Command<'a> {
    /// `{id, usage, u64 size}`
    CreateBuffer { id: Id, size: u64, usage: u32 },
    /// `{id, format, usage, dimension, width, height, layers, mips, samples, 0}`
    CreateTexture(Texture),
    /// `{id, texture}`: the texture's whole view.
    CreateView { id: Id, texture: Id },
    /// `{id, address u, v, w, mag, min, mipmap, compare, anisotropy, f32 lod_min, f32 lod_max, 0}`
    CreateSampler(Sampler),
    /// `{id, len}`, then the WGSL source.
    CreateShaderModule { id: Id, wgsl: &'a str },
    /// See [`RenderPipeline`].
    CreateRenderPipeline(RenderPipeline<'a>),
    /// `{id, module, entry_len, 0}`, then the entry point's name.
    CreateComputePipeline { id: Id, module: Id, entry: &'a str },
    /// `{id, pipeline, group, 0}`: names a pipeline's own layout for its bind group `group`.
    BindGroupLayoutOf { id: Id, pipeline: Id, group: u32 },
    /// `{id, layout, n, 0}`, then `n` × `{binding, kind, resource, 0, u64 offset, u64 size}`.
    CreateBindGroup {
        id: Id,
        layout: Id,
        entries: Vec<BindEntry>,
    },
    /// `{buffer, len, u64 offset}`, then the bytes.
    WriteBuffer {
        buffer: Id,
        offset: u64,
        data: &'a [u8],
    },
    /// `{texture, mip, x, y, z, width, height, depth, bytes_per_row, rows_per_image, len, 0}`, then
    /// the bytes.
    WriteTexture {
        texture: TextureCopy,
        size: [u32; 3],
        bytes_per_row: u32,
        rows_per_image: u32,
        data: &'a [u8],
    },
    /// `{n, 0}`, then `n` × `{view, resolve, load, store, f64 clear[4]}`.
    BeginRenderPass { colors: Vec<ColorAttachment> },
    /// No payload.
    BeginComputePass,
    /// `{pipeline, 0}`
    SetPipeline { pipeline: Id },
    /// `{index, group}`
    SetBindGroup { index: u32, group: Id },
    /// `{slot, buffer, u64 offset, u64 size}`; a size of 0 is the rest of the buffer.
    SetVertexBuffer {
        slot: u32,
        buffer: Id,
        offset: u64,
        size: u64,
    },
    /// `{buffer, format, u64 offset, u64 size}`; a size of 0 is the rest of the buffer.
    SetIndexBuffer {
        buffer: Id,
        format: IndexFormat,
        offset: u64,
        size: u64,
    },
    /// `{x, y, width, height}`
    SetScissor {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    /// `{vertices, instances, first_vertex, first_instance}`
    Draw {
        vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    /// `{indices, instances, first_index, i32 base_vertex, first_instance, 0}`
    DrawIndexed {
        indices: u32,
        instances: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    },
    /// `{x, y, z, 0}`
    Dispatch { x: u32, y: u32, z: u32 },
    /// No payload: ends the open pass.
    EndPass,
    /// `{src, dst, u64 src_offset, u64 dst_offset, u64 size}`
    CopyBufferToBuffer {
        src: Id,
        dst: Id,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    },
    /// `{BufferCopy}{TextureCopy}{width, height, depth, 0}`
    CopyBufferToTexture {
        buffer: BufferCopy,
        texture: TextureCopy,
        size: [u32; 3],
    },
    /// The same layout as [`Command::CopyBufferToTexture`], copying the other way.
    CopyTextureToBuffer {
        buffer: BufferCopy,
        texture: TextureCopy,
        size: [u32; 3],
    },
    /// `{buffer, 0, u64 offset, u64 size}`: copies the range back to the host, after every command
    /// before it. The guest takes it with the `read` op, which says `-EAGAIN` until it arrives.
    Readback { buffer: Id, offset: u64, size: u64 },
    /// `{texture, view}`: the surface's texture for this frame, and its whole view.
    Acquire { texture: Id, view: Id },
    /// `{texture, view}`, the two ids `Acquire` gave: presents the texture, and frees both ids.
    Present { texture: Id, view: Id },
    /// `{id, 0}`
    Destroy { id: Id },
}

/// The wire codes of the commands, in the order of [`Command`]'s variants.
pub mod code {
    pub const CREATE_BUFFER: u32 = 1;
    pub const CREATE_TEXTURE: u32 = 2;
    pub const CREATE_VIEW: u32 = 3;
    pub const CREATE_SAMPLER: u32 = 4;
    pub const CREATE_SHADER_MODULE: u32 = 5;
    pub const CREATE_RENDER_PIPELINE: u32 = 6;
    pub const CREATE_COMPUTE_PIPELINE: u32 = 7;
    pub const BIND_GROUP_LAYOUT_OF: u32 = 8;
    pub const CREATE_BIND_GROUP: u32 = 9;
    pub const WRITE_BUFFER: u32 = 10;
    pub const WRITE_TEXTURE: u32 = 11;
    pub const BEGIN_RENDER_PASS: u32 = 12;
    pub const BEGIN_COMPUTE_PASS: u32 = 13;
    pub const SET_PIPELINE: u32 = 14;
    pub const SET_BIND_GROUP: u32 = 15;
    pub const SET_VERTEX_BUFFER: u32 = 16;
    pub const SET_INDEX_BUFFER: u32 = 17;
    pub const SET_SCISSOR: u32 = 18;
    pub const DRAW: u32 = 19;
    pub const DRAW_INDEXED: u32 = 20;
    pub const DISPATCH: u32 = 21;
    pub const END_PASS: u32 = 22;
    pub const COPY_BUFFER_TO_BUFFER: u32 = 23;
    pub const COPY_BUFFER_TO_TEXTURE: u32 = 24;
    pub const COPY_TEXTURE_TO_BUFFER: u32 = 25;
    pub const READBACK: u32 = 26;
    pub const ACQUIRE: u32 = 27;
    pub const PRESENT: u32 = 28;
    pub const DESTROY: u32 = 29;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Texture {
    pub id: Id,
    pub format: TextureFormat,
    pub usage: u32,
    pub dimension: TextureDimension,
    /// Width, height, and depth or array layers.
    pub size: [u32; 3],
    pub mips: u32,
    pub samples: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sampler {
    pub id: Id,
    pub address: [AddressMode; 3],
    pub mag: FilterMode,
    pub min: FilterMode,
    pub mipmap: FilterMode,
    pub compare: Option<CompareFunction>,
    pub anisotropy: u32,
    pub lod_min: f32,
    pub lod_max: f32,
}

/// `{id, module, topology, strip_index_format, front_face, cull, samples, n_buffers, n_targets,
/// vertex_len, fragment_len, 0}`, the two entry points' names, `n_buffers` × ([`VertexBuffer`]
/// `{u64 stride, step, n_attributes}` then its attributes `{format, location, u64 offset}`), and
/// `n_targets` × [`ColorTarget`] `{format, write_mask, blend, color src, dst, op, alpha src, dst,
/// op, 0}`. Its layout is the pipeline's own (WebGPU's `"auto"`), which
/// [`Command::BindGroupLayoutOf`] names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderPipeline<'a> {
    pub id: Id,
    pub module: Id,
    pub vertex: &'a str,
    pub fragment: &'a str,
    pub topology: Topology,
    pub strip_index_format: Option<IndexFormat>,
    pub front_face: FrontFace,
    pub cull: Option<Face>,
    pub samples: u32,
    pub buffers: Vec<VertexBuffer>,
    pub targets: Vec<ColorTarget>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexBuffer {
    pub stride: u64,
    pub step: StepMode,
    pub attributes: Vec<VertexAttribute>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub location: u32,
    pub offset: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorTarget {
    pub format: TextureFormat,
    pub write_mask: u32,
    pub blend: Option<Blend>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blend {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendComponent {
    pub src: BlendFactor,
    pub dst: BlendFactor,
    pub op: BlendOperation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindEntry {
    pub binding: u32,
    pub resource: Resource,
}

/// What a bind group entry binds. On the wire its kind is 1, 2 or 3; a sampler's and a view's
/// offset and size are 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resource {
    /// A range of a buffer; a size of 0 is the rest of it.
    Buffer {
        id: Id,
        offset: u64,
        size: u64,
    },
    Sampler(Id),
    View(Id),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorAttachment {
    pub view: Id,
    pub resolve: Option<Id>,
    pub load: LoadOp,
    pub store: StoreOp,
    /// RGBA, as WebGPU's `GPUColor`: finite doubles.
    pub clear: [f64; 4],
}

/// `{buffer, bytes_per_row, rows_per_image, 0, u64 offset}`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferCopy {
    pub buffer: Id,
    pub offset: u64,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
}

/// `{texture, mip, x, y, z, 0}`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureCopy {
    pub texture: Id,
    pub mip: u32,
    pub origin: [u32; 3],
}

impl Command<'_> {
    /// The command's name, as errors give it.
    pub fn name(&self) -> &'static str {
        match self {
            Command::CreateBuffer { .. } => "create_buffer",
            Command::CreateTexture(_) => "create_texture",
            Command::CreateView { .. } => "create_view",
            Command::CreateSampler(_) => "create_sampler",
            Command::CreateShaderModule { .. } => "create_shader_module",
            Command::CreateRenderPipeline(_) => "create_render_pipeline",
            Command::CreateComputePipeline { .. } => "create_compute_pipeline",
            Command::BindGroupLayoutOf { .. } => "bind_group_layout_of",
            Command::CreateBindGroup { .. } => "create_bind_group",
            Command::WriteBuffer { .. } => "write_buffer",
            Command::WriteTexture { .. } => "write_texture",
            Command::BeginRenderPass { .. } => "begin_render_pass",
            Command::BeginComputePass => "begin_compute_pass",
            Command::SetPipeline { .. } => "set_pipeline",
            Command::SetBindGroup { .. } => "set_bind_group",
            Command::SetVertexBuffer { .. } => "set_vertex_buffer",
            Command::SetIndexBuffer { .. } => "set_index_buffer",
            Command::SetScissor { .. } => "set_scissor",
            Command::Draw { .. } => "draw",
            Command::DrawIndexed { .. } => "draw_indexed",
            Command::Dispatch { .. } => "dispatch",
            Command::EndPass => "end_pass",
            Command::CopyBufferToBuffer { .. } => "copy_buffer_to_buffer",
            Command::CopyBufferToTexture { .. } => "copy_buffer_to_texture",
            Command::CopyTextureToBuffer { .. } => "copy_texture_to_buffer",
            Command::Readback { .. } => "readback",
            Command::Acquire { .. } => "acquire",
            Command::Present { .. } => "present",
            Command::Destroy { .. } => "destroy",
        }
    }
}

// --- Encoding ---------------------------------------------------------------------------------

/// Appends commands to a stream. A Rust guest builds its submits with it; the host's tests build
/// theirs. `temen_gpu.h` is the same for a C guest.
#[derive(Default)]
pub struct Encoder {
    pub bytes: Vec<u8>,
}

impl Encoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn pad(&mut self) {
        while !self.bytes.len().is_multiple_of(8) {
            self.bytes.push(0);
        }
    }
    fn raw(&mut self, b: &[u8]) {
        self.bytes.extend_from_slice(b);
        self.pad();
    }
    fn texture_copy(&mut self, t: &TextureCopy) {
        self.u32(t.texture);
        self.u32(t.mip);
        self.u32(t.origin[0]);
        self.u32(t.origin[1]);
        self.u32(t.origin[2]);
        self.u32(0);
    }
    fn buffer_copy(&mut self, b: &BufferCopy) {
        self.u32(b.buffer);
        self.u32(b.bytes_per_row);
        self.u32(b.rows_per_image);
        self.u32(0);
        self.u64(b.offset);
    }
    fn extent(&mut self, s: [u32; 3]) {
        self.u32(s[0]);
        self.u32(s[1]);
        self.u32(s[2]);
        self.u32(0);
    }

    /// Appends one command.
    pub fn push(&mut self, c: &Command<'_>) {
        use code::*;
        let at = self.bytes.len();
        self.u32(0);
        self.u32(0);
        let cmd = match c {
            Command::CreateBuffer { id, size, usage } => {
                self.u32(*id);
                self.u32(*usage);
                self.u64(*size);
                CREATE_BUFFER
            }
            Command::CreateTexture(t) => {
                for v in [
                    t.id,
                    t.format.code(),
                    t.usage,
                    t.dimension.code(),
                    t.size[0],
                    t.size[1],
                    t.size[2],
                    t.mips,
                    t.samples,
                    0,
                ] {
                    self.u32(v);
                }
                CREATE_TEXTURE
            }
            Command::CreateView { id, texture } => {
                self.u32(*id);
                self.u32(*texture);
                CREATE_VIEW
            }
            Command::CreateSampler(s) => {
                for v in [
                    s.id,
                    s.address[0].code(),
                    s.address[1].code(),
                    s.address[2].code(),
                    s.mag.code(),
                    s.min.code(),
                    s.mipmap.code(),
                    s.compare.map_or(0, |c| c.code()),
                    s.anisotropy,
                ] {
                    self.u32(v);
                }
                self.f32(s.lod_min);
                self.f32(s.lod_max);
                self.u32(0);
                CREATE_SAMPLER
            }
            Command::CreateShaderModule { id, wgsl } => {
                self.u32(*id);
                self.u32(wgsl.len() as u32);
                self.raw(wgsl.as_bytes());
                CREATE_SHADER_MODULE
            }
            Command::CreateRenderPipeline(p) => {
                for v in [
                    p.id,
                    p.module,
                    p.topology.code(),
                    p.strip_index_format.map_or(0, |f| f.code()),
                    p.front_face.code(),
                    p.cull.map_or(0, |f| f.code()),
                    p.samples,
                    p.buffers.len() as u32,
                    p.targets.len() as u32,
                    p.vertex.len() as u32,
                    p.fragment.len() as u32,
                    0,
                ] {
                    self.u32(v);
                }
                self.raw(p.vertex.as_bytes());
                self.raw(p.fragment.as_bytes());
                for b in &p.buffers {
                    self.u64(b.stride);
                    self.u32(b.step.code());
                    self.u32(b.attributes.len() as u32);
                    for a in &b.attributes {
                        self.u32(a.format.code());
                        self.u32(a.location);
                        self.u64(a.offset);
                    }
                }
                for t in &p.targets {
                    let (c, a) = match t.blend {
                        Some(b) => (b.color, b.alpha),
                        None => {
                            let z = BlendComponent {
                                src: BlendFactor::One,
                                dst: BlendFactor::Zero,
                                op: BlendOperation::Add,
                            };
                            (z, z)
                        }
                    };
                    let blend = t.blend.is_some() as u32;
                    let off = |v: u32| if blend == 1 { v } else { 0 };
                    for v in [
                        t.format.code(),
                        t.write_mask,
                        blend,
                        off(c.src.code()),
                        off(c.dst.code()),
                        off(c.op.code()),
                        off(a.src.code()),
                        off(a.dst.code()),
                        off(a.op.code()),
                        0,
                    ] {
                        self.u32(v);
                    }
                }
                CREATE_RENDER_PIPELINE
            }
            Command::CreateComputePipeline { id, module, entry } => {
                self.u32(*id);
                self.u32(*module);
                self.u32(entry.len() as u32);
                self.u32(0);
                self.raw(entry.as_bytes());
                CREATE_COMPUTE_PIPELINE
            }
            Command::BindGroupLayoutOf {
                id,
                pipeline,
                group,
            } => {
                self.u32(*id);
                self.u32(*pipeline);
                self.u32(*group);
                self.u32(0);
                BIND_GROUP_LAYOUT_OF
            }
            Command::CreateBindGroup {
                id,
                layout,
                entries,
            } => {
                self.u32(*id);
                self.u32(*layout);
                self.u32(entries.len() as u32);
                self.u32(0);
                for e in entries {
                    let (kind, res, offset, size) = match e.resource {
                        Resource::Buffer { id, offset, size } => (1, id, offset, size),
                        Resource::Sampler(id) => (2, id, 0, 0),
                        Resource::View(id) => (3, id, 0, 0),
                    };
                    self.u32(e.binding);
                    self.u32(kind);
                    self.u32(res);
                    self.u32(0);
                    self.u64(offset);
                    self.u64(size);
                }
                CREATE_BIND_GROUP
            }
            Command::WriteBuffer {
                buffer,
                offset,
                data,
            } => {
                self.u32(*buffer);
                self.u32(data.len() as u32);
                self.u64(*offset);
                self.raw(data);
                WRITE_BUFFER
            }
            Command::WriteTexture {
                texture,
                size,
                bytes_per_row,
                rows_per_image,
                data,
            } => {
                for v in [
                    texture.texture,
                    texture.mip,
                    texture.origin[0],
                    texture.origin[1],
                    texture.origin[2],
                    size[0],
                    size[1],
                    size[2],
                    *bytes_per_row,
                    *rows_per_image,
                    data.len() as u32,
                    0,
                ] {
                    self.u32(v);
                }
                self.raw(data);
                WRITE_TEXTURE
            }
            Command::BeginRenderPass { colors } => {
                self.u32(colors.len() as u32);
                self.u32(0);
                for a in colors {
                    self.u32(a.view);
                    self.u32(a.resolve.unwrap_or(0));
                    self.u32(a.load.code());
                    self.u32(a.store.code());
                    for v in a.clear {
                        self.f64(v);
                    }
                }
                BEGIN_RENDER_PASS
            }
            Command::BeginComputePass => BEGIN_COMPUTE_PASS,
            Command::SetPipeline { pipeline } => {
                self.u32(*pipeline);
                self.u32(0);
                SET_PIPELINE
            }
            Command::SetBindGroup { index, group } => {
                self.u32(*index);
                self.u32(*group);
                SET_BIND_GROUP
            }
            Command::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => {
                self.u32(*slot);
                self.u32(*buffer);
                self.u64(*offset);
                self.u64(*size);
                SET_VERTEX_BUFFER
            }
            Command::SetIndexBuffer {
                buffer,
                format,
                offset,
                size,
            } => {
                self.u32(*buffer);
                self.u32(format.code());
                self.u64(*offset);
                self.u64(*size);
                SET_INDEX_BUFFER
            }
            Command::SetScissor {
                x,
                y,
                width,
                height,
            } => {
                for v in [*x, *y, *width, *height] {
                    self.u32(v);
                }
                SET_SCISSOR
            }
            Command::Draw {
                vertices,
                instances,
                first_vertex,
                first_instance,
            } => {
                for v in [*vertices, *instances, *first_vertex, *first_instance] {
                    self.u32(v);
                }
                DRAW
            }
            Command::DrawIndexed {
                indices,
                instances,
                first_index,
                base_vertex,
                first_instance,
            } => {
                for v in [
                    *indices,
                    *instances,
                    *first_index,
                    *base_vertex as u32,
                    *first_instance,
                    0,
                ] {
                    self.u32(v);
                }
                DRAW_INDEXED
            }
            Command::Dispatch { x, y, z } => {
                for v in [*x, *y, *z, 0] {
                    self.u32(v);
                }
                DISPATCH
            }
            Command::EndPass => END_PASS,
            Command::CopyBufferToBuffer {
                src,
                dst,
                src_offset,
                dst_offset,
                size,
            } => {
                self.u32(*src);
                self.u32(*dst);
                self.u64(*src_offset);
                self.u64(*dst_offset);
                self.u64(*size);
                COPY_BUFFER_TO_BUFFER
            }
            Command::CopyBufferToTexture {
                buffer,
                texture,
                size,
            } => {
                self.buffer_copy(buffer);
                self.texture_copy(texture);
                self.extent(*size);
                COPY_BUFFER_TO_TEXTURE
            }
            Command::CopyTextureToBuffer {
                buffer,
                texture,
                size,
            } => {
                self.buffer_copy(buffer);
                self.texture_copy(texture);
                self.extent(*size);
                COPY_TEXTURE_TO_BUFFER
            }
            Command::Readback {
                buffer,
                offset,
                size,
            } => {
                self.u32(*buffer);
                self.u32(0);
                self.u64(*offset);
                self.u64(*size);
                READBACK
            }
            Command::Acquire { texture, view } => {
                self.u32(*texture);
                self.u32(*view);
                ACQUIRE
            }
            Command::Present { texture, view } => {
                self.u32(*texture);
                self.u32(*view);
                PRESENT
            }
            Command::Destroy { id } => {
                self.u32(*id);
                self.u32(0);
                DESTROY
            }
        };
        let size = (self.bytes.len() - at - 8) as u32;
        self.bytes[at..at + 4].copy_from_slice(&cmd.to_le_bytes());
        self.bytes[at + 4..at + 8].copy_from_slice(&size.to_le_bytes());
    }
}

// --- Decoding ---------------------------------------------------------------------------------

/// Why a stream was refused: the index of the command, its code, and what was wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeError {
    pub index: usize,
    pub command: u32,
    pub what: &'static str,
}

/// A cursor over one record's payload. Every read is bounds-checked, and the payload must be used
/// up exactly.
struct Payload<'a> {
    b: &'a [u8],
    at: usize,
}

type R<T> = Result<T, &'static str>;

impl<'a> Payload<'a> {
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        let end = self.at.checked_add(n).ok_or("a length overflows")?;
        let s = self
            .b
            .get(self.at..end)
            .ok_or("the record is shorter than its fields")?;
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> R<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> R<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> R<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// A padding word, which must be zero.
    fn zero(&mut self) -> R<()> {
        if self.u32()? != 0 {
            return Err("a padding field is not zero");
        }
        Ok(())
    }
    /// `n` bytes of data, then zero padding to 8.
    fn raw(&mut self, n: u32) -> R<&'a [u8]> {
        let s = self.take(n as usize)?;
        let pad = (8 - n as usize % 8) % 8;
        if self.take(pad)?.iter().any(|&b| b != 0) {
            return Err("padding after the data is not zero");
        }
        Ok(s)
    }
    fn str(&mut self, n: u32) -> R<&'a str> {
        core::str::from_utf8(self.raw(n)?).map_err(|_| "a string is not UTF-8")
    }
    fn count(&mut self, max: usize) -> R<usize> {
        let n = self.u32()? as usize;
        if n > max {
            return Err("a count is over its limit");
        }
        Ok(n)
    }
    fn texture_copy(&mut self) -> R<TextureCopy> {
        let t = TextureCopy {
            texture: self.u32()?,
            mip: self.u32()?,
            origin: [self.u32()?, self.u32()?, self.u32()?],
        };
        self.zero()?;
        Ok(t)
    }
    fn buffer_copy(&mut self) -> R<BufferCopy> {
        let buffer = self.u32()?;
        let bytes_per_row = self.u32()?;
        let rows_per_image = self.u32()?;
        self.zero()?;
        Ok(BufferCopy {
            buffer,
            bytes_per_row,
            rows_per_image,
            offset: self.u64()?,
        })
    }
    fn extent(&mut self) -> R<[u32; 3]> {
        let s = [self.u32()?, self.u32()?, self.u32()?];
        self.zero()?;
        Ok(s)
    }
}

fn need<T>(v: Option<T>) -> R<T> {
    v.ok_or("a code no table names")
}

fn optional<T>(c: u32, f: impl Fn(u32) -> Option<T>) -> R<Option<T>> {
    if c == 0 {
        Ok(None)
    } else {
        need(f(c)).map(Some)
    }
}

fn component(p: &mut Payload) -> R<(u32, u32, u32)> {
    Ok((p.u32()?, p.u32()?, p.u32()?))
}

fn finite(v: f64) -> R<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err("a clear value is not finite")
    }
}

/// Decodes a whole stream, or refuses it at its first malformed record.
pub fn decode(stream: &[u8]) -> Result<Vec<Command<'_>>, DecodeError> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < stream.len() {
        let index = out.len();
        let fail = |command, what| DecodeError {
            index,
            command,
            what,
        };
        let header = stream
            .get(at..at + 8)
            .ok_or(fail(0, "a record header is cut short"))?;
        let command = u32::from_le_bytes(header[..4].try_into().unwrap());
        let size = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
        if !size.is_multiple_of(8) {
            return Err(fail(command, "a record's size is not a multiple of 8"));
        }
        let body = stream
            .get(at + 8..at + 8 + size)
            .ok_or(fail(command, "a record runs past the stream"))?;
        let mut p = Payload { b: body, at: 0 };
        let c = one(command, &mut p).map_err(|what| fail(command, what))?;
        if p.at != body.len() {
            return Err(fail(command, "a record is longer than its fields"));
        }
        out.push(c);
        at += 8 + size;
    }
    Ok(out)
}

fn one<'a>(command: u32, p: &mut Payload<'a>) -> R<Command<'a>> {
    use code::*;
    Ok(match command {
        CREATE_BUFFER => Command::CreateBuffer {
            id: p.u32()?,
            usage: p.u32()?,
            size: p.u64()?,
        },
        CREATE_TEXTURE => {
            let id = p.u32()?;
            let format = need(TextureFormat::from_code(p.u32()?))?;
            let usage = p.u32()?;
            let dimension = need(TextureDimension::from_code(p.u32()?))?;
            let size = [p.u32()?, p.u32()?, p.u32()?];
            let mips = p.u32()?;
            let samples = p.u32()?;
            p.zero()?;
            Command::CreateTexture(Texture {
                id,
                format,
                usage,
                dimension,
                size,
                mips,
                samples,
            })
        }
        CREATE_VIEW => Command::CreateView {
            id: p.u32()?,
            texture: p.u32()?,
        },
        CREATE_SAMPLER => {
            let id = p.u32()?;
            let address = [
                need(AddressMode::from_code(p.u32()?))?,
                need(AddressMode::from_code(p.u32()?))?,
                need(AddressMode::from_code(p.u32()?))?,
            ];
            let mag = need(FilterMode::from_code(p.u32()?))?;
            let min = need(FilterMode::from_code(p.u32()?))?;
            let mipmap = need(FilterMode::from_code(p.u32()?))?;
            let compare = optional(p.u32()?, CompareFunction::from_code)?;
            let anisotropy = p.u32()?;
            let lod_min = p.f32()?;
            let lod_max = p.f32()?;
            p.zero()?;
            if !(lod_min.is_finite() && lod_max.is_finite()) {
                return Err("a sampler's LOD clamp is not finite");
            }
            Command::CreateSampler(Sampler {
                id,
                address,
                mag,
                min,
                mipmap,
                compare,
                anisotropy,
                lod_min,
                lod_max,
            })
        }
        CREATE_SHADER_MODULE => {
            let id = p.u32()?;
            let n = p.u32()?;
            Command::CreateShaderModule {
                id,
                wgsl: p.str(n)?,
            }
        }
        CREATE_RENDER_PIPELINE => {
            let id = p.u32()?;
            let module = p.u32()?;
            let topology = need(Topology::from_code(p.u32()?))?;
            let strip_index_format = optional(p.u32()?, IndexFormat::from_code)?;
            let front_face = need(FrontFace::from_code(p.u32()?))?;
            let cull = optional(p.u32()?, Face::from_code)?;
            let samples = p.u32()?;
            let n_buffers = p.count(MAX_VERTEX_BUFFERS)?;
            let n_targets = p.count(MAX_COLORS)?;
            let vertex_len = p.u32()?;
            let fragment_len = p.u32()?;
            p.zero()?;
            let vertex = p.str(vertex_len)?;
            let fragment = p.str(fragment_len)?;
            let mut buffers = Vec::with_capacity(n_buffers);
            for _ in 0..n_buffers {
                let stride = p.u64()?;
                let step = need(StepMode::from_code(p.u32()?))?;
                let n = p.count(MAX_ATTRIBUTES)?;
                let mut attributes = Vec::with_capacity(n);
                for _ in 0..n {
                    attributes.push(VertexAttribute {
                        format: need(VertexFormat::from_code(p.u32()?))?,
                        location: p.u32()?,
                        offset: p.u64()?,
                    });
                }
                buffers.push(VertexBuffer {
                    stride,
                    step,
                    attributes,
                });
            }
            let mut targets = Vec::with_capacity(n_targets);
            for _ in 0..n_targets {
                let format = need(TextureFormat::from_code(p.u32()?))?;
                let write_mask = p.u32()?;
                let blend = p.u32()?;
                let c = component(p)?;
                let a = component(p)?;
                p.zero()?;
                let blend = match blend {
                    0 if c == (0, 0, 0) && a == (0, 0, 0) => None,
                    0 => return Err("a target without blending names blend factors"),
                    1 => {
                        let comp = |(s, d, o): (u32, u32, u32)| -> R<BlendComponent> {
                            Ok(BlendComponent {
                                src: need(BlendFactor::from_code(s))?,
                                dst: need(BlendFactor::from_code(d))?,
                                op: need(BlendOperation::from_code(o))?,
                            })
                        };
                        Some(Blend {
                            color: comp(c)?,
                            alpha: comp(a)?,
                        })
                    }
                    _ => return Err("a target's blend flag is neither 0 nor 1"),
                };
                if write_mask & !color_write::ALL != 0 {
                    return Err("a write mask sets an undefined bit");
                }
                targets.push(ColorTarget {
                    format,
                    write_mask,
                    blend,
                });
            }
            Command::CreateRenderPipeline(RenderPipeline {
                id,
                module,
                vertex,
                fragment,
                topology,
                strip_index_format,
                front_face,
                cull,
                samples,
                buffers,
                targets,
            })
        }
        CREATE_COMPUTE_PIPELINE => {
            let id = p.u32()?;
            let module = p.u32()?;
            let n = p.u32()?;
            p.zero()?;
            Command::CreateComputePipeline {
                id,
                module,
                entry: p.str(n)?,
            }
        }
        BIND_GROUP_LAYOUT_OF => {
            let c = Command::BindGroupLayoutOf {
                id: p.u32()?,
                pipeline: p.u32()?,
                group: p.u32()?,
            };
            p.zero()?;
            c
        }
        CREATE_BIND_GROUP => {
            let id = p.u32()?;
            let layout = p.u32()?;
            let n = p.count(MAX_BIND_ENTRIES)?;
            p.zero()?;
            let mut entries = Vec::with_capacity(n);
            for _ in 0..n {
                let binding = p.u32()?;
                let kind = p.u32()?;
                let res = p.u32()?;
                p.zero()?;
                let offset = p.u64()?;
                let size = p.u64()?;
                let resource = match kind {
                    1 => Resource::Buffer {
                        id: res,
                        offset,
                        size,
                    },
                    2 | 3 if offset != 0 || size != 0 => {
                        return Err("a sampler or a view entry names a buffer range")
                    }
                    2 => Resource::Sampler(res),
                    3 => Resource::View(res),
                    _ => return Err("a bind entry's kind is not 1, 2 or 3"),
                };
                entries.push(BindEntry { binding, resource });
            }
            Command::CreateBindGroup {
                id,
                layout,
                entries,
            }
        }
        WRITE_BUFFER => {
            let buffer = p.u32()?;
            let n = p.u32()?;
            let offset = p.u64()?;
            Command::WriteBuffer {
                buffer,
                offset,
                data: p.raw(n)?,
            }
        }
        WRITE_TEXTURE => {
            let texture = TextureCopy {
                texture: p.u32()?,
                mip: p.u32()?,
                origin: [p.u32()?, p.u32()?, p.u32()?],
            };
            let size = [p.u32()?, p.u32()?, p.u32()?];
            let bytes_per_row = p.u32()?;
            let rows_per_image = p.u32()?;
            let n = p.u32()?;
            p.zero()?;
            Command::WriteTexture {
                texture,
                size,
                bytes_per_row,
                rows_per_image,
                data: p.raw(n)?,
            }
        }
        BEGIN_RENDER_PASS => {
            let n = p.count(MAX_COLORS)?;
            p.zero()?;
            let mut colors = Vec::with_capacity(n);
            for _ in 0..n {
                let view = p.u32()?;
                let resolve = p.u32()?;
                let load = need(LoadOp::from_code(p.u32()?))?;
                let store = need(StoreOp::from_code(p.u32()?))?;
                let clear = [
                    finite(p.f64()?)?,
                    finite(p.f64()?)?,
                    finite(p.f64()?)?,
                    finite(p.f64()?)?,
                ];
                colors.push(ColorAttachment {
                    view,
                    resolve: (resolve != 0).then_some(resolve),
                    load,
                    store,
                    clear,
                });
            }
            Command::BeginRenderPass { colors }
        }
        BEGIN_COMPUTE_PASS => Command::BeginComputePass,
        SET_PIPELINE => {
            let c = Command::SetPipeline { pipeline: p.u32()? };
            p.zero()?;
            c
        }
        SET_BIND_GROUP => Command::SetBindGroup {
            index: p.u32()?,
            group: p.u32()?,
        },
        SET_VERTEX_BUFFER => Command::SetVertexBuffer {
            slot: p.u32()?,
            buffer: p.u32()?,
            offset: p.u64()?,
            size: p.u64()?,
        },
        SET_INDEX_BUFFER => Command::SetIndexBuffer {
            buffer: p.u32()?,
            format: need(IndexFormat::from_code(p.u32()?))?,
            offset: p.u64()?,
            size: p.u64()?,
        },
        SET_SCISSOR => Command::SetScissor {
            x: p.u32()?,
            y: p.u32()?,
            width: p.u32()?,
            height: p.u32()?,
        },
        DRAW => Command::Draw {
            vertices: p.u32()?,
            instances: p.u32()?,
            first_vertex: p.u32()?,
            first_instance: p.u32()?,
        },
        DRAW_INDEXED => {
            let c = Command::DrawIndexed {
                indices: p.u32()?,
                instances: p.u32()?,
                first_index: p.u32()?,
                base_vertex: p.u32()? as i32,
                first_instance: p.u32()?,
            };
            p.zero()?;
            c
        }
        DISPATCH => {
            let c = Command::Dispatch {
                x: p.u32()?,
                y: p.u32()?,
                z: p.u32()?,
            };
            p.zero()?;
            c
        }
        END_PASS => Command::EndPass,
        COPY_BUFFER_TO_BUFFER => Command::CopyBufferToBuffer {
            src: p.u32()?,
            dst: p.u32()?,
            src_offset: p.u64()?,
            dst_offset: p.u64()?,
            size: p.u64()?,
        },
        COPY_BUFFER_TO_TEXTURE | COPY_TEXTURE_TO_BUFFER => {
            let buffer = p.buffer_copy()?;
            let texture = p.texture_copy()?;
            let size = p.extent()?;
            if command == COPY_BUFFER_TO_TEXTURE {
                Command::CopyBufferToTexture {
                    buffer,
                    texture,
                    size,
                }
            } else {
                Command::CopyTextureToBuffer {
                    buffer,
                    texture,
                    size,
                }
            }
        }
        READBACK => {
            let buffer = p.u32()?;
            p.zero()?;
            Command::Readback {
                buffer,
                offset: p.u64()?,
                size: p.u64()?,
            }
        }
        ACQUIRE => Command::Acquire {
            texture: p.u32()?,
            view: p.u32()?,
        },
        PRESENT => Command::Present {
            texture: p.u32()?,
            view: p.u32()?,
        },
        DESTROY => {
            let c = Command::Destroy { id: p.u32()? };
            p.zero()?;
            c
        }
        _ => return Err("no such command"),
    })
}
