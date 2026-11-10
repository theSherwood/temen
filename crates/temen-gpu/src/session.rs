//! What a stream may name: each live id's kind, and what a command checks against it before any
//! backend sees the stream. A stream that fails a check runs no command at all, and the table is as
//! it was. So a backend executes only streams in which every id names a live object of the right
//! kind, every buffer range lies inside its buffer, and every pass command lies inside its pass. It
//! never meets a case its GPU API would panic on; what WebGPU itself validates (formats, usages,
//! layouts), it leaves to WebGPU.

use alloc::vec;
use alloc::vec::Vec;

use crate::command::*;
use crate::enums::{buffer_usage, texture_usage};
use crate::Id;

/// One more than the largest id a guest may use. Id 0 names nothing, so a zeroed field fails.
pub const MAX_IDS: u32 = 1 << 16;

/// The kind of object an id names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Buffer,
    Texture,
    View,
    Sampler,
    Module,
    RenderPipeline,
    ComputePipeline,
    Layout,
    BindGroup,
}

#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    kind: Option<Kind>,
    /// A buffer's size and usage.
    size: u64,
    usage: u32,
}

/// Why a stream was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The index of the command refused.
    pub index: usize,
    pub command: &'static str,
    pub what: &'static str,
}

/// A draw's range of vertices, indices or instances, `[first, first + count)`, fits in a `u32`.
fn within(first: u32, count: u32) -> Result<(), &'static str> {
    first
        .checked_add(count)
        .map(|_| ())
        .ok_or("a draw's range runs past u32")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    None,
    Render,
    Compute,
}

/// The live ids of one device.
pub struct Session {
    slots: Vec<Slot>,
    /// The surface texture and its view, between `Acquire` and `Present`.
    acquired: Option<(Id, Id)>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Session {
            slots: vec![Slot::default(); 1],
            acquired: None,
        }
    }

    /// The kind of object `id` names, if it is live.
    pub fn kind(&self, id: Id) -> Option<Kind> {
        self.slots.get(id as usize).and_then(|s| s.kind)
    }

    /// The surface texture and its view, if one is acquired.
    pub fn acquired(&self) -> Option<(Id, Id)> {
        self.acquired
    }

    /// Checks `cmds` against the live ids and applies what they create and free. On a refusal,
    /// the table is left as it was.
    pub fn check(&mut self, cmds: &[Command<'_>]) -> Result<(), Refusal> {
        let mut undo: Vec<(Id, Slot)> = Vec::new();
        let acquired = self.acquired;
        let mut pass = Pass::None;
        let mut result = Ok(());
        for (index, c) in cmds.iter().enumerate() {
            if let Err(what) = self.one(c, &mut pass, &mut undo) {
                result = Err(Refusal {
                    index,
                    command: c.name(),
                    what,
                });
                break;
            }
        }
        if result.is_ok() && pass != Pass::None {
            result = Err(Refusal {
                index: cmds.len(),
                command: "end of stream",
                what: "a pass is still open",
            });
        }
        if result.is_err() {
            for (id, slot) in undo.into_iter().rev() {
                self.slots[id as usize] = slot;
            }
            self.acquired = acquired;
        }
        result
    }

    fn set(&mut self, id: Id, slot: Slot, undo: &mut Vec<(Id, Slot)>) {
        let i = id as usize;
        if i >= self.slots.len() {
            self.slots.resize(i + 1, Slot::default());
        }
        undo.push((id, self.slots[i]));
        self.slots[i] = slot;
    }

    fn create(
        &mut self,
        id: Id,
        kind: Kind,
        undo: &mut Vec<(Id, Slot)>,
    ) -> Result<(), &'static str> {
        self.create_buffer(id, kind, 0, 0, undo)
    }

    fn create_buffer(
        &mut self,
        id: Id,
        kind: Kind,
        size: u64,
        usage: u32,
        undo: &mut Vec<(Id, Slot)>,
    ) -> Result<(), &'static str> {
        if id == 0 || id >= MAX_IDS {
            return Err("an id is 0 or over the limit");
        }
        if self.kind(id).is_some() {
            return Err("an id is already in use");
        }
        self.set(
            id,
            Slot {
                kind: Some(kind),
                size,
                usage,
            },
            undo,
        );
        Ok(())
    }

    fn free(&mut self, id: Id, undo: &mut Vec<(Id, Slot)>) {
        self.set(id, Slot::default(), undo);
    }

    fn is(&self, id: Id, kind: Kind) -> Result<(), &'static str> {
        match self.kind(id) {
            Some(k) if k == kind => Ok(()),
            Some(_) => Err("an id names an object of another kind"),
            None => Err("an id names no live object"),
        }
    }

    /// A slice a pass reads vertices or indices from: inside the buffer, and not empty.
    fn slice(&self, id: Id, offset: u64, size: u64) -> Result<(), &'static str> {
        self.range(id, offset, size)?;
        if offset >= self.slots[id as usize].size {
            return Err("a vertex or index slice is empty");
        }
        Ok(())
    }

    /// `[offset, offset + size)` lies inside buffer `id`; a size of 0 is the rest of it.
    fn range(&self, id: Id, offset: u64, size: u64) -> Result<(), &'static str> {
        self.is(id, Kind::Buffer)?;
        let len = self.slots[id as usize].size;
        let end = if size == 0 {
            Some(len)
        } else {
            offset.checked_add(size)
        };
        match end {
            Some(end) if offset <= len && end <= len => Ok(()),
            _ => Err("a range runs past its buffer"),
        }
    }

    fn one(
        &mut self,
        c: &Command<'_>,
        pass: &mut Pass,
        undo: &mut Vec<(Id, Slot)>,
    ) -> Result<(), &'static str> {
        let inside = |want: Pass| {
            if *pass == want {
                Ok(())
            } else if want == Pass::None {
                Err("only outside a pass")
            } else if want == Pass::Render {
                Err("only inside a render pass")
            } else {
                Err("only inside a compute pass")
            }
        };
        match c {
            Command::CreateBuffer { id, size, usage } => {
                inside(Pass::None)?;
                if *size == 0 || *usage == 0 || usage & !buffer_usage::ALLOWED != 0 {
                    return Err("a buffer needs a size, and usages from the allowed set");
                }
                self.create_buffer(*id, Kind::Buffer, *size, *usage, undo)
            }
            Command::CreateTexture(t) => {
                inside(Pass::None)?;
                if t.usage == 0 || t.usage & !texture_usage::ALLOWED != 0 {
                    return Err("a texture needs usages from the allowed set");
                }
                if t.size.contains(&0) || t.mips == 0 || !matches!(t.samples, 1 | 4) {
                    return Err("a texture needs a size, mips, and 1 or 4 samples");
                }
                self.create(t.id, Kind::Texture, undo)
            }
            Command::CreateView { id, texture } => {
                inside(Pass::None)?;
                self.is(*texture, Kind::Texture)?;
                self.create(*id, Kind::View, undo)
            }
            Command::CreateSampler(s) => {
                inside(Pass::None)?;
                if s.anisotropy == 0 || s.anisotropy > 16 {
                    return Err("a sampler's anisotropy is 1 to 16");
                }
                self.create(s.id, Kind::Sampler, undo)
            }
            Command::CreateShaderModule { id, .. } => {
                inside(Pass::None)?;
                self.create(*id, Kind::Module, undo)
            }
            Command::CreateRenderPipeline(p) => {
                inside(Pass::None)?;
                self.is(p.module, Kind::Module)?;
                if p.vertex.is_empty() || p.fragment.is_empty() {
                    return Err("a render pipeline names both entry points");
                }
                if p.targets.is_empty() || !matches!(p.samples, 1 | 4) {
                    return Err("a render pipeline has a target, and 1 or 4 samples");
                }
                self.create(p.id, Kind::RenderPipeline, undo)
            }
            Command::CreateComputePipeline { id, module, entry } => {
                inside(Pass::None)?;
                self.is(*module, Kind::Module)?;
                if entry.is_empty() {
                    return Err("a compute pipeline names its entry point");
                }
                self.create(*id, Kind::ComputePipeline, undo)
            }
            Command::BindGroupLayoutOf {
                id,
                pipeline,
                group,
            } => {
                inside(Pass::None)?;
                if !matches!(
                    self.kind(*pipeline),
                    Some(Kind::RenderPipeline | Kind::ComputePipeline)
                ) {
                    return Err("a layout comes from a pipeline");
                }
                if *group >= MAX_BIND_GROUPS {
                    return Err("a bind group index is over the limit");
                }
                self.create(*id, Kind::Layout, undo)
            }
            Command::CreateBindGroup {
                id,
                layout,
                entries,
            } => {
                inside(Pass::None)?;
                self.is(*layout, Kind::Layout)?;
                for e in entries {
                    match e.resource {
                        Resource::Buffer { id, offset, size } => self.range(id, offset, size)?,
                        Resource::Sampler(id) => self.is(id, Kind::Sampler)?,
                        Resource::View(id) => self.is(id, Kind::View)?,
                    }
                }
                self.create(*id, Kind::BindGroup, undo)
            }
            Command::WriteBuffer {
                buffer,
                offset,
                data,
            } => {
                inside(Pass::None)?;
                if data.is_empty() {
                    return Err("a write needs bytes");
                }
                self.range(*buffer, *offset, data.len() as u64)
            }
            Command::WriteTexture { texture, data, .. } => {
                inside(Pass::None)?;
                if data.is_empty() {
                    return Err("a write needs bytes");
                }
                self.is(texture.texture, Kind::Texture)
            }
            Command::BeginRenderPass { colors } => {
                inside(Pass::None)?;
                if colors.is_empty() {
                    return Err("a render pass has a colour attachment");
                }
                for a in colors {
                    self.is(a.view, Kind::View)?;
                    if let Some(r) = a.resolve {
                        self.is(r, Kind::View)?;
                    }
                }
                *pass = Pass::Render;
                Ok(())
            }
            Command::BeginComputePass => {
                inside(Pass::None)?;
                *pass = Pass::Compute;
                Ok(())
            }
            Command::SetPipeline { pipeline } => match *pass {
                Pass::Render => self.is(*pipeline, Kind::RenderPipeline),
                Pass::Compute => self.is(*pipeline, Kind::ComputePipeline),
                Pass::None => Err("only inside a pass"),
            },
            Command::SetBindGroup { index, group } => {
                if *pass == Pass::None {
                    return Err("only inside a pass");
                }
                if *index >= MAX_BIND_GROUPS {
                    return Err("a bind group index is over the limit");
                }
                self.is(*group, Kind::BindGroup)
            }
            Command::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => {
                inside(Pass::Render)?;
                if *slot as usize >= MAX_VERTEX_BUFFERS {
                    return Err("a vertex buffer slot is over the limit");
                }
                self.slice(*buffer, *offset, *size)
            }
            Command::SetIndexBuffer {
                buffer,
                offset,
                size,
                ..
            } => {
                inside(Pass::Render)?;
                self.slice(*buffer, *offset, *size)
            }
            Command::SetScissor { .. } => inside(Pass::Render),
            Command::Draw {
                vertices,
                instances,
                first_vertex,
                first_instance,
            } => {
                inside(Pass::Render)?;
                within(*first_vertex, *vertices)?;
                within(*first_instance, *instances)
            }
            Command::DrawIndexed {
                indices,
                instances,
                first_index,
                first_instance,
                ..
            } => {
                inside(Pass::Render)?;
                within(*first_index, *indices)?;
                within(*first_instance, *instances)
            }
            Command::Dispatch { .. } => inside(Pass::Compute),
            Command::EndPass => {
                if *pass == Pass::None {
                    return Err("no pass is open");
                }
                *pass = Pass::None;
                Ok(())
            }
            Command::CopyBufferToBuffer {
                src,
                dst,
                src_offset,
                dst_offset,
                size,
            } => {
                inside(Pass::None)?;
                if *size == 0 {
                    return Err("a copy needs a size");
                }
                self.range(*src, *src_offset, *size)?;
                self.range(*dst, *dst_offset, *size)
            }
            Command::CopyBufferToTexture {
                buffer, texture, ..
            }
            | Command::CopyTextureToBuffer {
                buffer, texture, ..
            } => {
                inside(Pass::None)?;
                self.range(buffer.buffer, buffer.offset, 0)?;
                self.is(texture.texture, Kind::Texture)
            }
            Command::Readback {
                buffer,
                offset,
                size,
            } => {
                inside(Pass::None)?;
                if *size == 0 || offset % 4 != 0 || size % 4 != 0 {
                    return Err(
                        "a readback's offset and size are multiples of 4, and it has a size",
                    );
                }
                self.range(*buffer, *offset, *size)?;
                if self.slots[*buffer as usize].usage & buffer_usage::COPY_SRC == 0 {
                    return Err("a readback needs a buffer with COPY_SRC");
                }
                Ok(())
            }
            Command::Acquire { texture, view } => {
                inside(Pass::None)?;
                if self.acquired.is_some() {
                    return Err("a surface texture is already acquired");
                }
                self.create(*texture, Kind::Texture, undo)?;
                self.create(*view, Kind::View, undo)?;
                self.acquired = Some((*texture, *view));
                Ok(())
            }
            Command::Present { texture, view } => {
                inside(Pass::None)?;
                if self.acquired != Some((*texture, *view)) {
                    return Err("present names the acquired texture and its view");
                }
                self.acquired = None;
                self.free(*texture, undo);
                self.free(*view, undo);
                Ok(())
            }
            Command::Destroy { id } => {
                inside(Pass::None)?;
                if self.kind(*id).is_none() {
                    return Err("an id names no live object");
                }
                if self.acquired.is_some_and(|(t, v)| *id == t || *id == v) {
                    return Err("the surface texture is freed by present");
                }
                self.free(*id, undo);
                Ok(())
            }
        }
    }
}
