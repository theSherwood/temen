//! **The `webgpu` capability natively, over [`wgpu`]** (`DESIGN.md` §17, #2665).
//!
//! The protocol is `temen-gpu`'s, which the browser's capability shares: a guest submits command
//! streams that name objects by its own ids, and [`temen_gpu::Gpu`] decodes and checks each one
//! before any of it runs. This crate is the [`Backend`] that then runs it on a wgpu device, and the
//! [`HostCap`] that grants it. The host holds the device; the guest only ever passes data
//! (validation, not raw access, is the boundary, §2a), and wgpu validates what the protocol leaves
//! to WebGPU.
//!
//! - **The device** is made on first use: a hardware adapter if the host has one, else a software
//!   one (lavapipe), which is what tests and CI run on. With none, every submit says `-ENODEV`.
//! - **Errors** never panic the host. wgpu's own validation errors are recorded for the guest's
//!   `error` op, and a panic inside wgpu, which the protocol's checks exist to rule out, loses the
//!   device rather than the host.
//! - **The surface** is what [`Surface`] binds. A guest presents to an offscreen texture of the
//!   host's, which each present reads back into [`Frames`]: a test sees exactly what was presented.
//!   A window's surface is the embedder's (#2837).
//!
//! Kept in its own workspace-excluded crate (like `temen-llvm`) so the heavy `wgpu` dependency
//! never enters the default build.

use std::num::NonZeroU64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use temen_gpu::command::*;
use temen_gpu::enums::*;
use temen_gpu::{Backend, Fault, Gpu, Id, Readback, SurfaceInfo};
use temen_interp::{GuestMem, RegionMinter, Trap};
use temen_run::HostCap;

/// What a guest's frames are presented to.
#[derive(Clone)]
pub enum Surface {
    /// No surface: the `surface` op says `-ENODEV`, and a stream that acquires one is refused.
    None,
    /// A texture the host owns, of this size and a 4-byte format; each present reads it back into
    /// `frames`.
    Offscreen {
        width: u32,
        height: u32,
        format: TextureFormat,
        frames: Frames,
    },
}

/// The frames a guest has presented to an offscreen surface: the last one, and how many.
#[derive(Clone, Default)]
pub struct Frames(Arc<Mutex<(u64, Option<Frame>)>>);

/// One presented frame: rows of `width` texels of 4 bytes each, top to bottom, in the surface's
/// format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub texels: Vec<u8>,
}

impl Frames {
    /// The number of frames presented, and the last.
    pub fn last(&self) -> (u64, Option<Frame>) {
        self.0.lock().unwrap().clone()
    }
}

/// The `webgpu` capability: grant it under the name `"webgpu"`. Each host instantiation holds its
/// own device, made on first use.
pub fn webgpu_cap(surface: Surface) -> HostCap {
    HostCap::host_proc(0, move |_, _| {
        let mut gpu = Gpu::new(Wgpu::new(surface.clone()));
        let h: temen_interp::HostProc = Box::new(
            move |op: u32,
                  args: &[i64],
                  out: &mut [i64],
                  mem: Option<&mut dyn GuestMem>,
                  _minter: Option<&mut dyn RegionMinter>|
                  -> Result<(), Trap> {
                let r = match mem {
                    Some(m) => gpu.call(op, args, Some(&mut Window(m))),
                    None => gpu.call(op, args, None),
                };
                temen_interp::put_results(out, &[r])
            },
        );
        // The guest's objects live on the GPU, which nothing here serializes (#1699).
        Some((h, temen_interp::CapState::Uncaptured))
    })
}

/// Whether this host has an adapter, hardware or software.
pub fn adapter_available() -> bool {
    Device::open().is_ok()
}

/// The guest's window, as the protocol reaches it.
struct Window<'m>(&'m mut dyn GuestMem);

impl temen_gpu::Window for Window<'_> {
    fn read(&mut self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        self.0.read_bytes(ptr, len)
    }
    fn write(&mut self, ptr: u64, bytes: &[u8]) -> Option<()> {
        self.0.write_bytes(ptr, bytes)
    }
}

struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Device {
    /// A hardware adapter if there is one, else a software one.
    fn open() -> Result<Device, String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let mut adapter = None;
            for fallback in [false, true] {
                adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        force_fallback_adapter: fallback,
                        compatible_surface: None,
                    })
                    .await;
                if adapter.is_some() {
                    break;
                }
            }
            let adapter =
                adapter.ok_or("no WebGPU adapter: install a Vulkan driver, such as lavapipe")?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .map_err(|e| format!("no WebGPU device: {e}"))?;
            Ok(Device { device, queue })
        })
    }
}

/// What an id names on the device.
enum Obj {
    Free,
    Buffer(wgpu::Buffer),
    Texture(wgpu::Texture),
    View(wgpu::TextureView),
    Sampler(wgpu::Sampler),
    Module(wgpu::ShaderModule),
    Render(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline),
    Layout(wgpu::BindGroupLayout),
    Group(wgpu::BindGroup),
}

/// A readback on its way: the host's staging buffer, and whether its map has finished (0 not yet,
/// 1 done, 2 failed).
struct Pending {
    buffer: Id,
    staging: wgpu::Buffer,
    state: Arc<AtomicU8>,
}

struct Wgpu {
    surface: Surface,
    device: Option<Result<Device, String>>,
    objs: Vec<Obj>,
    pending: Vec<Pending>,
    /// The offscreen surface's texture, made on the first acquire.
    offscreen: Option<wgpu::Texture>,
    /// The first error wgpu reported since the guest last asked.
    error: Arc<Mutex<Option<String>>>,
}

impl Wgpu {
    fn new(surface: Surface) -> Self {
        Wgpu {
            surface,
            device: None,
            objs: Vec::new(),
            pending: Vec::new(),
            offscreen: None,
            error: Arc::default(),
        }
    }

    fn device(&mut self) -> Result<(), Fault> {
        let error = Arc::clone(&self.error);
        let d = self.device.get_or_insert_with(|| {
            Device::open().inspect(|d| {
                // An error wgpu does not hand back is the guest's to read, never a host panic.
                d.device.on_uncaptured_error(Box::new(move |e| {
                    error.lock().unwrap().get_or_insert_with(|| e.to_string());
                }));
            })
        });
        match d {
            Ok(_) => Ok(()),
            Err(why) => Err(Fault::NoDevice(why.clone())),
        }
    }

    fn set(&mut self, id: Id, obj: Obj) {
        let i = id as usize;
        if i >= self.objs.len() {
            self.objs.resize_with(i + 1, || Obj::Free);
        }
        self.objs[i] = obj;
    }
}

// What the session's checks guarantee: each id names a live object of the kind its command
// expects. A broken guarantee panics, which `run` turns into a lost device.
macro_rules! get {
    ($self:ident, $kind:ident, $id:expr) => {
        match &$self.objs[$id as usize] {
            Obj::$kind(o) => o,
            _ => panic!("id {} is not a live {}", $id, stringify!($kind)),
        }
    };
}

impl Backend for Wgpu {
    fn run(&mut self, cmds: &[Command<'_>]) -> Result<(), Fault> {
        self.device()?;
        let ran = catch_unwind(AssertUnwindSafe(|| self.execute(cmds)));
        if let Err(panic) = ran {
            let why = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            let why = format!("the WebGPU device failed, and is lost: {why}");
            self.device = Some(Err(why.clone()));
            return Err(Fault::NoDevice(why));
        }
        Ok(())
    }

    fn readback(&mut self, buffer: Id) -> Readback {
        let Some(i) = self.pending.iter().position(|p| p.buffer == buffer) else {
            return Readback::None;
        };
        if let Some(Ok(d)) = &self.device {
            let _ = d.device.poll(wgpu::Maintain::Poll);
        }
        match self.pending[i].state.load(Ordering::Acquire) {
            0 => Readback::Pending,
            1 => {
                let p = self.pending.remove(i);
                let bytes = p.staging.slice(..).get_mapped_range().to_vec();
                p.staging.unmap();
                Readback::Ready(bytes)
            }
            _ => {
                self.pending.remove(i);
                self.error
                    .lock()
                    .unwrap()
                    .get_or_insert_with(|| format!("the readback of buffer {buffer} failed"));
                Readback::None
            }
        }
    }

    fn surface(&mut self) -> Option<SurfaceInfo> {
        match &self.surface {
            Surface::None => None,
            Surface::Offscreen {
                width,
                height,
                format,
                ..
            } => Some(SurfaceInfo {
                width: *width,
                height: *height,
                format: *format,
            }),
        }
    }

    fn take_error(&mut self) -> Option<String> {
        self.error.lock().unwrap().take()
    }
}

impl Wgpu {
    /// Runs a checked stream in order. Commands that record go into one encoder, which is
    /// submitted before anything that acts on the queue, so each command sees all before it.
    fn execute(&mut self, cmds: &[Command<'_>]) {
        let mut enc: Option<wgpu::CommandEncoder> = None;
        let mut i = 0;
        while i < cmds.len() {
            match &cmds[i] {
                Command::BeginRenderPass { colors } => {
                    let attachments: Vec<_> = colors
                        .iter()
                        .map(|a| {
                            Some(wgpu::RenderPassColorAttachment {
                                view: get!(self, View, a.view),
                                resolve_target: a.resolve.map(|r| get!(self, View, r)),
                                ops: wgpu::Operations {
                                    load: match a.load {
                                        LoadOp::Clear => wgpu::LoadOp::Clear(wgpu::Color {
                                            r: a.clear[0],
                                            g: a.clear[1],
                                            b: a.clear[2],
                                            a: a.clear[3],
                                        }),
                                        LoadOp::Load => wgpu::LoadOp::Load,
                                    },
                                    store: match a.store {
                                        StoreOp::Store => wgpu::StoreOp::Store,
                                        StoreOp::Discard => wgpu::StoreOp::Discard,
                                    },
                                },
                            })
                        })
                        .collect();
                    let d = self.dev();
                    let e = enc.get_or_insert_with(|| {
                        d.device.create_command_encoder(&Default::default())
                    });
                    let mut pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    i += 1;
                    while !matches!(cmds[i], Command::EndPass) {
                        self.render(&mut pass, &cmds[i]);
                        i += 1;
                    }
                }
                Command::BeginComputePass => {
                    let d = self.dev();
                    let e = enc.get_or_insert_with(|| {
                        d.device.create_command_encoder(&Default::default())
                    });
                    let mut pass = e.begin_compute_pass(&Default::default());
                    i += 1;
                    while !matches!(cmds[i], Command::EndPass) {
                        match &cmds[i] {
                            Command::SetPipeline { pipeline } => {
                                pass.set_pipeline(get!(self, Compute, *pipeline))
                            }
                            Command::SetBindGroup { index, group } => {
                                pass.set_bind_group(*index, get!(self, Group, *group), &[])
                            }
                            Command::Dispatch { x, y, z } => pass.dispatch_workgroups(*x, *y, *z),
                            c => panic!("{} in a compute pass", c.name()),
                        }
                        i += 1;
                    }
                }
                Command::CopyBufferToBuffer {
                    src,
                    dst,
                    src_offset,
                    dst_offset,
                    size,
                } => {
                    let d = self.dev();
                    enc.get_or_insert_with(|| d.device.create_command_encoder(&Default::default()))
                        .copy_buffer_to_buffer(
                            get!(self, Buffer, *src),
                            *src_offset,
                            get!(self, Buffer, *dst),
                            *dst_offset,
                            *size,
                        );
                }
                Command::CopyBufferToTexture {
                    buffer,
                    texture,
                    size,
                } => {
                    let d = self.dev();
                    enc.get_or_insert_with(|| d.device.create_command_encoder(&Default::default()))
                        .copy_buffer_to_texture(
                            self.buffer_copy(buffer),
                            self.texture_copy(texture),
                            extent(*size),
                        );
                }
                Command::CopyTextureToBuffer {
                    buffer,
                    texture,
                    size,
                } => {
                    let d = self.dev();
                    enc.get_or_insert_with(|| d.device.create_command_encoder(&Default::default()))
                        .copy_texture_to_buffer(
                            self.texture_copy(texture),
                            self.buffer_copy(buffer),
                            extent(*size),
                        );
                }
                c @ (Command::WriteBuffer { .. }
                | Command::WriteTexture { .. }
                | Command::Readback { .. }
                | Command::Present { .. }
                | Command::Destroy { .. }) => {
                    self.flush(&mut enc);
                    self.queue(c);
                }
                c => self.create(c),
            }
            i += 1;
        }
        self.flush(&mut enc);
    }

    fn dev(&self) -> &Device {
        match &self.device {
            Some(Ok(d)) => d,
            _ => panic!("no device"),
        }
    }

    fn flush(&self, enc: &mut Option<wgpu::CommandEncoder>) {
        if let Some(e) = enc.take() {
            self.dev().queue.submit(Some(e.finish()));
        }
    }

    fn texture_copy(&self, t: &TextureCopy) -> wgpu::TexelCopyTextureInfo<'_> {
        wgpu::TexelCopyTextureInfo {
            texture: get!(self, Texture, t.texture),
            mip_level: t.mip,
            origin: wgpu::Origin3d {
                x: t.origin[0],
                y: t.origin[1],
                z: t.origin[2],
            },
            aspect: wgpu::TextureAspect::All,
        }
    }

    fn buffer_copy(&self, b: &BufferCopy) -> wgpu::TexelCopyBufferInfo<'_> {
        wgpu::TexelCopyBufferInfo {
            buffer: get!(self, Buffer, b.buffer),
            layout: layout(b.offset, b.bytes_per_row, b.rows_per_image),
        }
    }

    /// One command inside a render pass.
    fn render(&self, pass: &mut wgpu::RenderPass<'_>, c: &Command<'_>) {
        match c {
            Command::SetPipeline { pipeline } => pass.set_pipeline(get!(self, Render, *pipeline)),
            Command::SetBindGroup { index, group } => {
                pass.set_bind_group(*index, get!(self, Group, *group), &[])
            }
            Command::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => pass.set_vertex_buffer(*slot, slice(get!(self, Buffer, *buffer), *offset, *size)),
            Command::SetIndexBuffer {
                buffer,
                format,
                offset,
                size,
            } => pass.set_index_buffer(
                slice(get!(self, Buffer, *buffer), *offset, *size),
                index_format(*format),
            ),
            Command::SetScissor {
                x,
                y,
                width,
                height,
            } => pass.set_scissor_rect(*x, *y, *width, *height),
            Command::Draw {
                vertices,
                instances,
                first_vertex,
                first_instance,
            } => pass.draw(
                *first_vertex..first_vertex + vertices,
                *first_instance..first_instance + instances,
            ),
            Command::DrawIndexed {
                indices,
                instances,
                first_index,
                base_vertex,
                first_instance,
            } => pass.draw_indexed(
                *first_index..first_index + indices,
                *base_vertex,
                *first_instance..first_instance + instances,
            ),
            c => panic!("{} in a render pass", c.name()),
        }
    }

    /// A command that acts on the queue, after the encoder before it is submitted.
    fn queue(&mut self, c: &Command<'_>) {
        match c {
            Command::WriteBuffer {
                buffer,
                offset,
                data,
            } => self
                .dev()
                .queue
                .write_buffer(get!(self, Buffer, *buffer), *offset, data),
            Command::WriteTexture {
                texture,
                size,
                bytes_per_row,
                rows_per_image,
                data,
            } => self.dev().queue.write_texture(
                self.texture_copy(texture),
                data,
                layout(0, *bytes_per_row, *rows_per_image),
                extent(*size),
            ),
            Command::Readback {
                buffer,
                offset,
                size,
            } => {
                let d = self.dev();
                let staging = d.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: *size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut e = d.device.create_command_encoder(&Default::default());
                e.copy_buffer_to_buffer(get!(self, Buffer, *buffer), *offset, &staging, 0, *size);
                d.queue.submit(Some(e.finish()));
                let state = Arc::new(AtomicU8::new(0));
                let s = Arc::clone(&state);
                staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    s.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release);
                });
                // A newer readback of the same buffer replaces an older one not yet read.
                self.pending.retain(|p| p.buffer != *buffer);
                self.pending.push(Pending {
                    buffer: *buffer,
                    staging,
                    state,
                });
            }
            Command::Present { texture, view } => {
                self.present();
                self.set(*texture, Obj::Free);
                self.set(*view, Obj::Free);
            }
            Command::Destroy { id } => {
                match std::mem::replace(&mut self.objs[*id as usize], Obj::Free) {
                    Obj::Buffer(b) => b.destroy(),
                    Obj::Texture(t) => t.destroy(),
                    _ => {}
                }
                self.pending.retain(|p| p.buffer != *id);
            }
            c => panic!("{} on the queue", c.name()),
        }
    }

    /// Reads the offscreen surface back into the frames, as the guest presented it.
    fn present(&mut self) {
        let Surface::Offscreen {
            width,
            height,
            format,
            frames,
        } = &self.surface
        else {
            panic!("present with no surface");
        };
        let (w, h) = (*width, *height);
        let texture = self.offscreen.as_ref().expect("acquired before present");
        let row = w * 4;
        let padded = row.div_ceil(256) * 256;
        let d = self.dev();
        let staging = d.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut e = d.device.create_command_encoder(&Default::default());
        e.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: layout(0, padded, h),
            },
            extent([w, h, 1]),
        );
        d.queue.submit(Some(e.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = d.device.poll(wgpu::Maintain::Wait);
        let mapped = staging.slice(..).get_mapped_range();
        let mut texels = Vec::with_capacity((row * h) as usize);
        for y in 0..h as usize {
            let at = y * padded as usize;
            texels.extend_from_slice(&mapped[at..at + row as usize]);
        }
        drop(mapped);
        staging.unmap();
        let mut f = frames.0.lock().unwrap();
        f.0 += 1;
        f.1 = Some(Frame {
            width: w,
            height: h,
            format: *format,
            texels,
        });
    }

    /// A command that makes an object.
    fn create(&mut self, c: &Command<'_>) {
        let Some(Ok(d)) = &self.device else {
            panic!("no device");
        };
        let (id, obj) = match c {
            Command::CreateBuffer { id, size, usage } => (
                *id,
                Obj::Buffer(d.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: *size,
                    usage: wgpu::BufferUsages::from_bits_truncate(*usage),
                    mapped_at_creation: false,
                })),
            ),
            Command::CreateTexture(t) => (
                t.id,
                Obj::Texture(d.device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: extent(t.size),
                    mip_level_count: t.mips,
                    sample_count: t.samples,
                    dimension: match t.dimension {
                        TextureDimension::D1 => wgpu::TextureDimension::D1,
                        TextureDimension::D2 => wgpu::TextureDimension::D2,
                        TextureDimension::D3 => wgpu::TextureDimension::D3,
                    },
                    format: texture_format(t.format),
                    usage: wgpu::TextureUsages::from_bits_truncate(t.usage),
                    view_formats: &[],
                })),
            ),
            Command::CreateView { id, texture } => (
                *id,
                Obj::View(get!(self, Texture, *texture).create_view(&Default::default())),
            ),
            Command::CreateSampler(s) => (
                s.id,
                Obj::Sampler(d.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: None,
                    address_mode_u: address(s.address[0]),
                    address_mode_v: address(s.address[1]),
                    address_mode_w: address(s.address[2]),
                    mag_filter: filter(s.mag),
                    min_filter: filter(s.min),
                    mipmap_filter: filter(s.mipmap),
                    lod_min_clamp: s.lod_min,
                    lod_max_clamp: s.lod_max,
                    compare: s.compare.map(compare),
                    anisotropy_clamp: s.anisotropy as u16,
                    border_color: None,
                })),
            ),
            Command::CreateShaderModule { id, wgsl } => (
                *id,
                Obj::Module(d.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl((*wgsl).to_owned().into()),
                })),
            ),
            Command::CreateRenderPipeline(p) => (p.id, Obj::Render(self.render_pipeline(p))),
            Command::CreateComputePipeline { id, module, entry } => (
                *id,
                Obj::Compute(
                    d.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: None,
                            layout: None,
                            module: get!(self, Module, *module),
                            entry_point: Some(*entry),
                            compilation_options: Default::default(),
                            cache: None,
                        }),
                ),
            ),
            Command::BindGroupLayoutOf {
                id,
                pipeline,
                group,
            } => (
                *id,
                Obj::Layout(match &self.objs[*pipeline as usize] {
                    Obj::Render(p) => p.get_bind_group_layout(*group),
                    Obj::Compute(p) => p.get_bind_group_layout(*group),
                    _ => panic!("id {pipeline} is not a pipeline"),
                }),
            ),
            Command::CreateBindGroup {
                id,
                layout,
                entries,
            } => {
                let entries: Vec<_> = entries
                    .iter()
                    .map(|e| wgpu::BindGroupEntry {
                        binding: e.binding,
                        resource: match e.resource {
                            Resource::Buffer { id, offset, size } => {
                                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: get!(self, Buffer, id),
                                    offset,
                                    size: NonZeroU64::new(size),
                                })
                            }
                            Resource::Sampler(id) => {
                                wgpu::BindingResource::Sampler(get!(self, Sampler, id))
                            }
                            Resource::View(id) => {
                                wgpu::BindingResource::TextureView(get!(self, View, id))
                            }
                        },
                    })
                    .collect();
                (
                    *id,
                    Obj::Group(d.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: get!(self, Layout, *layout),
                        entries: &entries,
                    })),
                )
            }
            Command::Acquire { texture, view } => {
                let Surface::Offscreen {
                    width,
                    height,
                    format,
                    ..
                } = &self.surface
                else {
                    panic!("acquire with no surface");
                };
                let t = self
                    .offscreen
                    .get_or_insert_with(|| {
                        d.device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("offscreen surface"),
                            size: extent([*width, *height, 1]),
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: texture_format(*format),
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                | wgpu::TextureUsages::COPY_SRC,
                            view_formats: &[],
                        })
                    })
                    .clone();
                let v = t.create_view(&Default::default());
                self.set(*texture, Obj::Texture(t));
                self.set(*view, Obj::View(v));
                return;
            }
            c => panic!("{} is not a creation", c.name()),
        };
        self.set(id, obj);
    }

    fn render_pipeline(&self, p: &RenderPipeline<'_>) -> wgpu::RenderPipeline {
        let attributes: Vec<Vec<wgpu::VertexAttribute>> = p
            .buffers
            .iter()
            .map(|b| {
                b.attributes
                    .iter()
                    .map(|a| wgpu::VertexAttribute {
                        format: vertex_format(a.format),
                        offset: a.offset,
                        shader_location: a.location,
                    })
                    .collect()
            })
            .collect();
        let buffers: Vec<wgpu::VertexBufferLayout> = p
            .buffers
            .iter()
            .zip(&attributes)
            .map(|(b, a)| wgpu::VertexBufferLayout {
                array_stride: b.stride,
                step_mode: match b.step {
                    StepMode::Vertex => wgpu::VertexStepMode::Vertex,
                    StepMode::Instance => wgpu::VertexStepMode::Instance,
                },
                attributes: a,
            })
            .collect();
        let targets: Vec<Option<wgpu::ColorTargetState>> = p
            .targets
            .iter()
            .map(|t| {
                Some(wgpu::ColorTargetState {
                    format: texture_format(t.format),
                    blend: t.blend.map(|b| wgpu::BlendState {
                        color: component(b.color),
                        alpha: component(b.alpha),
                    }),
                    write_mask: wgpu::ColorWrites::from_bits_truncate(t.write_mask),
                })
            })
            .collect();
        let module = get!(self, Module, p.module);
        self.dev()
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(p.vertex),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: match p.topology {
                        Topology::PointList => wgpu::PrimitiveTopology::PointList,
                        Topology::LineList => wgpu::PrimitiveTopology::LineList,
                        Topology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
                        Topology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
                        Topology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
                    },
                    strip_index_format: p.strip_index_format.map(index_format),
                    front_face: match p.front_face {
                        FrontFace::Ccw => wgpu::FrontFace::Ccw,
                        FrontFace::Cw => wgpu::FrontFace::Cw,
                    },
                    cull_mode: p.cull.map(|f| match f {
                        Face::Front => wgpu::Face::Front,
                        Face::Back => wgpu::Face::Back,
                    }),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: p.samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(p.fragment),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview: None,
                cache: None,
            })
    }
}

/// A slice the session checked: inside the buffer, and not empty.
fn slice(b: &wgpu::Buffer, offset: u64, size: u64) -> wgpu::BufferSlice<'_> {
    if size == 0 {
        b.slice(offset..)
    } else {
        b.slice(offset..offset + size)
    }
}

fn layout(offset: u64, bytes_per_row: u32, rows_per_image: u32) -> wgpu::TexelCopyBufferLayout {
    wgpu::TexelCopyBufferLayout {
        offset,
        bytes_per_row: (bytes_per_row != 0).then_some(bytes_per_row),
        rows_per_image: (rows_per_image != 0).then_some(rows_per_image),
    }
}

fn extent(s: [u32; 3]) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: s[0],
        height: s[1],
        depth_or_array_layers: s[2],
    }
}

fn texture_format(f: TextureFormat) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as W;
    match f {
        TextureFormat::R8Unorm => W::R8Unorm,
        TextureFormat::Rg8Unorm => W::Rg8Unorm,
        TextureFormat::Rgba8Unorm => W::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => W::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => W::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb => W::Bgra8UnormSrgb,
        TextureFormat::R16Float => W::R16Float,
        TextureFormat::Rg16Float => W::Rg16Float,
        TextureFormat::Rgba16Float => W::Rgba16Float,
        TextureFormat::R32Float => W::R32Float,
        TextureFormat::Rg32Float => W::Rg32Float,
        TextureFormat::Rgba32Float => W::Rgba32Float,
        TextureFormat::R32Uint => W::R32Uint,
        TextureFormat::Rgb10a2Unorm => W::Rgb10a2Unorm,
    }
}

fn vertex_format(f: VertexFormat) -> wgpu::VertexFormat {
    use wgpu::VertexFormat as W;
    match f {
        VertexFormat::Float32 => W::Float32,
        VertexFormat::Float32x2 => W::Float32x2,
        VertexFormat::Float32x3 => W::Float32x3,
        VertexFormat::Float32x4 => W::Float32x4,
        VertexFormat::Uint32 => W::Uint32,
        VertexFormat::Uint32x2 => W::Uint32x2,
        VertexFormat::Uint32x3 => W::Uint32x3,
        VertexFormat::Uint32x4 => W::Uint32x4,
        VertexFormat::Sint32 => W::Sint32,
        VertexFormat::Sint32x2 => W::Sint32x2,
        VertexFormat::Sint32x3 => W::Sint32x3,
        VertexFormat::Sint32x4 => W::Sint32x4,
        VertexFormat::Unorm8x4 => W::Unorm8x4,
        VertexFormat::Uint8x4 => W::Uint8x4,
        VertexFormat::Uint16x2 => W::Uint16x2,
        VertexFormat::Sint16x2 => W::Sint16x2,
        VertexFormat::Unorm16x2 => W::Unorm16x2,
        VertexFormat::Float16x2 => W::Float16x2,
        VertexFormat::Float16x4 => W::Float16x4,
    }
}

fn index_format(f: IndexFormat) -> wgpu::IndexFormat {
    match f {
        IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
        IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
    }
}

fn component(c: BlendComponent) -> wgpu::BlendComponent {
    use wgpu::BlendFactor as F;
    let factor = |f: BlendFactor| match f {
        BlendFactor::Zero => F::Zero,
        BlendFactor::One => F::One,
        BlendFactor::Src => F::Src,
        BlendFactor::OneMinusSrc => F::OneMinusSrc,
        BlendFactor::SrcAlpha => F::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => F::OneMinusSrcAlpha,
        BlendFactor::Dst => F::Dst,
        BlendFactor::OneMinusDst => F::OneMinusDst,
        BlendFactor::DstAlpha => F::DstAlpha,
        BlendFactor::OneMinusDstAlpha => F::OneMinusDstAlpha,
        BlendFactor::SrcAlphaSaturated => F::SrcAlphaSaturated,
    };
    wgpu::BlendComponent {
        src_factor: factor(c.src),
        dst_factor: factor(c.dst),
        operation: match c.op {
            BlendOperation::Add => wgpu::BlendOperation::Add,
            BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
            BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
            BlendOperation::Min => wgpu::BlendOperation::Min,
            BlendOperation::Max => wgpu::BlendOperation::Max,
        },
    }
}

fn address(a: AddressMode) -> wgpu::AddressMode {
    match a {
        AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        AddressMode::Repeat => wgpu::AddressMode::Repeat,
        AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
    }
}

fn filter(f: FilterMode) -> wgpu::FilterMode {
    match f {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

fn compare(c: CompareFunction) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as W;
    match c {
        CompareFunction::Never => W::Never,
        CompareFunction::Less => W::Less,
        CompareFunction::Equal => W::Equal,
        CompareFunction::LessEqual => W::LessEqual,
        CompareFunction::Greater => W::Greater,
        CompareFunction::NotEqual => W::NotEqual,
        CompareFunction::GreaterEqual => W::GreaterEqual,
        CompareFunction::Always => W::Always,
    }
}
