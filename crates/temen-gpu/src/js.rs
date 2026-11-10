//! The browser's form of a checked stream: the WebGPU JS calls it stands for, as JSON, for a
//! generic executor (`browser/web/webgpu.js`) to make against `navigator.gpu`. Every piece of
//! knowledge of the WebGPU API lives here, beside the native backend's, so the executor holds none:
//! it resolves references and calls methods by name.
//!
//! The form is a JSON array of steps, each `[target, method, args]` or `[target, method, args,
//! into]`:
//! - `target` is `"device"`, `"queue"`, `"ctx"` (the canvas context), a register (`"enc"`,
//!   `"pass"`, `"cb"`), an object id (a number), or `"js"` for the executor's own helpers:
//!   `readback(buffer, offset, size)`, `destroy(id)` and `present(texture, view)`;
//! - `args` are JSON values in which `{"$": n}` is object `n` or register `n`, and `{"@": [offset,
//!   length]}` is a `Uint8Array` of the bytes beside the steps;
//! - `into`, when present, is the id or register the call's result goes to.
//!
//! Commands take effect in stream order. A command encoder opens on the first command that records
//! into one, and is finished and submitted before anything that acts on the queue (a write, a
//! readback, a present, a destroy) and at the end of the stream.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use crate::command::*;
use crate::Id;

/// Appends the JSON form of `cmds` to `out`, and their data to `blob`.
pub fn to_js(cmds: &[Command<'_>], out: &mut String, blob: &mut Vec<u8>) {
    let mut j = Js {
        out,
        blob,
        first: true,
        encoding: false,
    };
    j.out.push('[');
    for c in cmds {
        j.command(c);
    }
    j.flush();
    j.out.push(']');
}

struct Js<'o> {
    out: &'o mut String,
    blob: &'o mut Vec<u8>,
    first: bool,
    /// Whether a command encoder is open.
    encoding: bool,
}

/// A reference to an object by id.
struct Obj(Id);

impl Js<'_> {
    /// Opens a step: `[target, "method", `; the caller writes the args array, then `close`.
    fn step(&mut self, target: &str, method: &str) {
        if !self.first {
            self.out.push(',');
        }
        self.first = false;
        let _ = write!(self.out, "[{target},\"{method}\",");
    }
    fn close(&mut self) {
        self.out.push(']');
    }
    fn yields(&mut self, id: Id) {
        let _ = write!(self.out, ",{id}]");
    }
    fn yields_reg(&mut self, reg: &str) {
        let _ = write!(self.out, ",\"{reg}\"]");
    }
    fn obj(&mut self, o: Obj) {
        let _ = write!(self.out, "{{\"$\":{}}}", o.0);
    }
    fn bytes(&mut self, data: &[u8]) {
        let at = self.blob.len();
        self.blob.extend_from_slice(data);
        let _ = write!(self.out, "{{\"@\":[{at},{}]}}", data.len());
    }
    fn str(&mut self, s: &str) {
        self.out.push('"');
        for ch in s.chars() {
            match ch {
                '"' => self.out.push_str("\\\""),
                '\\' => self.out.push_str("\\\\"),
                '\n' => self.out.push_str("\\n"),
                '\r' => self.out.push_str("\\r"),
                '\t' => self.out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(self.out, "\\u{:04x}", c as u32);
                }
                c => self.out.push(c),
            }
        }
        self.out.push('"');
    }
    fn num(&mut self, v: impl core::fmt::Display) {
        let _ = write!(self.out, "{v}");
    }

    /// Opens a command encoder if none is.
    fn encoder(&mut self) {
        if !self.encoding {
            self.step("\"device\"", "createCommandEncoder");
            self.out.push_str("[]");
            self.yields_reg("enc");
            self.encoding = true;
        }
    }

    /// Finishes and submits the open encoder, if one is.
    fn flush(&mut self) {
        if self.encoding {
            self.step("\"enc\"", "finish");
            self.out.push_str("[]");
            self.yields_reg("cb");
            self.step("\"queue\"", "submit");
            self.out.push_str("[[{\"$\":\"cb\"}]]");
            self.close();
            self.encoding = false;
        }
    }

    fn texture_copy(&mut self, t: &TextureCopy) {
        self.out.push_str("{\"texture\":");
        self.obj(Obj(t.texture));
        let _ = write!(
            self.out,
            ",\"mipLevel\":{},\"origin\":[{},{},{}]}}",
            t.mip, t.origin[0], t.origin[1], t.origin[2]
        );
    }
    fn buffer_copy(&mut self, b: &BufferCopy) {
        self.out.push_str("{\"buffer\":");
        self.obj(Obj(b.buffer));
        let _ = write!(
            self.out,
            ",\"offset\":{},\"bytesPerRow\":{},\"rowsPerImage\":{}}}",
            b.offset, b.bytes_per_row, b.rows_per_image
        );
    }
    fn extent(&mut self, s: [u32; 3]) {
        let _ = write!(self.out, "[{},{},{}]", s[0], s[1], s[2]);
    }

    fn command(&mut self, c: &Command<'_>) {
        match c {
            Command::CreateBuffer { id, size, usage } => {
                self.step("\"device\"", "createBuffer");
                let _ = write!(self.out, "[{{\"size\":{size},\"usage\":{usage}}}]");
                self.yields(*id);
            }
            Command::CreateTexture(t) => {
                self.step("\"device\"", "createTexture");
                let _ = write!(
                    self.out,
                    "[{{\"size\":[{},{},{}],\"format\":\"{}\",\"usage\":{},\"dimension\":\"{}\",\
                     \"mipLevelCount\":{},\"sampleCount\":{}}}]",
                    t.size[0],
                    t.size[1],
                    t.size[2],
                    t.format.name(),
                    t.usage,
                    t.dimension.name(),
                    t.mips,
                    t.samples
                );
                self.yields(t.id);
            }
            Command::CreateView { id, texture } => {
                self.step(&alloc::format!("{texture}"), "createView");
                self.out.push_str("[]");
                self.yields(*id);
            }
            Command::CreateSampler(s) => {
                self.step("\"device\"", "createSampler");
                let _ = write!(
                    self.out,
                    "[{{\"addressModeU\":\"{}\",\"addressModeV\":\"{}\",\"addressModeW\":\"{}\",\
                     \"magFilter\":\"{}\",\"minFilter\":\"{}\",\"mipmapFilter\":\"{}\",\
                     \"lodMinClamp\":{},\"lodMaxClamp\":{},\"maxAnisotropy\":{}",
                    s.address[0].name(),
                    s.address[1].name(),
                    s.address[2].name(),
                    s.mag.name(),
                    s.min.name(),
                    s.mipmap.name(),
                    s.lod_min,
                    s.lod_max,
                    s.anisotropy
                );
                if let Some(c) = s.compare {
                    let _ = write!(self.out, ",\"compare\":\"{}\"", c.name());
                }
                self.out.push_str("}]");
                self.yields(s.id);
            }
            Command::CreateShaderModule { id, wgsl } => {
                self.step("\"device\"", "createShaderModule");
                self.out.push_str("[{\"code\":");
                self.str(wgsl);
                self.out.push_str("}]");
                self.yields(*id);
            }
            Command::CreateRenderPipeline(p) => {
                self.step("\"device\"", "createRenderPipeline");
                self.out
                    .push_str("[{\"layout\":\"auto\",\"vertex\":{\"module\":");
                self.obj(Obj(p.module));
                self.out.push_str(",\"entryPoint\":");
                self.str(p.vertex);
                self.out.push_str(",\"buffers\":[");
                for (i, b) in p.buffers.iter().enumerate() {
                    if i > 0 {
                        self.out.push(',');
                    }
                    let _ = write!(
                        self.out,
                        "{{\"arrayStride\":{},\"stepMode\":\"{}\",\"attributes\":[",
                        b.stride,
                        b.step.name()
                    );
                    for (k, a) in b.attributes.iter().enumerate() {
                        if k > 0 {
                            self.out.push(',');
                        }
                        let _ = write!(
                            self.out,
                            "{{\"format\":\"{}\",\"offset\":{},\"shaderLocation\":{}}}",
                            a.format.name(),
                            a.offset,
                            a.location
                        );
                    }
                    self.out.push_str("]}");
                }
                self.out.push_str("]},\"fragment\":{\"module\":");
                self.obj(Obj(p.module));
                self.out.push_str(",\"entryPoint\":");
                self.str(p.fragment);
                self.out.push_str(",\"targets\":[");
                for (i, t) in p.targets.iter().enumerate() {
                    if i > 0 {
                        self.out.push(',');
                    }
                    let _ = write!(
                        self.out,
                        "{{\"format\":\"{}\",\"writeMask\":{}",
                        t.format.name(),
                        t.write_mask
                    );
                    if let Some(b) = t.blend {
                        let comp = |c: BlendComponent| {
                            alloc::format!(
                                "{{\"srcFactor\":\"{}\",\"dstFactor\":\"{}\",\"operation\":\"{}\"}}",
                                c.src.name(),
                                c.dst.name(),
                                c.op.name()
                            )
                        };
                        let _ = write!(
                            self.out,
                            ",\"blend\":{{\"color\":{},\"alpha\":{}}}",
                            comp(b.color),
                            comp(b.alpha)
                        );
                    }
                    self.out.push('}');
                }
                let _ = write!(
                    self.out,
                    "]}},\"primitive\":{{\"topology\":\"{}\",\"frontFace\":\"{}\",\"cullMode\":\"{}\"",
                    p.topology.name(),
                    p.front_face.name(),
                    p.cull.map_or("none", |f| f.name())
                );
                if let Some(f) = p.strip_index_format {
                    let _ = write!(self.out, ",\"stripIndexFormat\":\"{}\"", f.name());
                }
                let _ = write!(
                    self.out,
                    "}},\"multisample\":{{\"count\":{}}}}}]",
                    p.samples
                );
                self.yields(p.id);
            }
            Command::CreateComputePipeline { id, module, entry } => {
                self.step("\"device\"", "createComputePipeline");
                self.out
                    .push_str("[{\"layout\":\"auto\",\"compute\":{\"module\":");
                self.obj(Obj(*module));
                self.out.push_str(",\"entryPoint\":");
                self.str(entry);
                self.out.push_str("}}]");
                self.yields(*id);
            }
            Command::BindGroupLayoutOf {
                id,
                pipeline,
                group,
            } => {
                self.step(&alloc::format!("{pipeline}"), "getBindGroupLayout");
                let _ = write!(self.out, "[{group}]");
                self.yields(*id);
            }
            Command::CreateBindGroup {
                id,
                layout,
                entries,
            } => {
                self.step("\"device\"", "createBindGroup");
                self.out.push_str("[{\"layout\":");
                self.obj(Obj(*layout));
                self.out.push_str(",\"entries\":[");
                for (i, e) in entries.iter().enumerate() {
                    if i > 0 {
                        self.out.push(',');
                    }
                    let _ = write!(self.out, "{{\"binding\":{},\"resource\":", e.binding);
                    match e.resource {
                        Resource::Buffer { id, offset, size } => {
                            self.out.push_str("{\"buffer\":");
                            self.obj(Obj(id));
                            let _ = write!(self.out, ",\"offset\":{offset}");
                            if size != 0 {
                                let _ = write!(self.out, ",\"size\":{size}");
                            }
                            self.out.push('}');
                        }
                        Resource::Sampler(id) | Resource::View(id) => self.obj(Obj(id)),
                    }
                    self.out.push('}');
                }
                self.out.push_str("]}]");
                self.yields(*id);
            }
            Command::WriteBuffer {
                buffer,
                offset,
                data,
            } => {
                self.flush();
                self.step("\"queue\"", "writeBuffer");
                self.out.push('[');
                self.obj(Obj(*buffer));
                let _ = write!(self.out, ",{offset},");
                self.bytes(data);
                self.out.push(']');
                self.close();
            }
            Command::WriteTexture {
                texture,
                size,
                bytes_per_row,
                rows_per_image,
                data,
            } => {
                self.flush();
                self.step("\"queue\"", "writeTexture");
                self.out.push('[');
                self.texture_copy(texture);
                self.out.push(',');
                self.bytes(data);
                let _ = write!(
                    self.out,
                    ",{{\"offset\":0,\"bytesPerRow\":{bytes_per_row},\"rowsPerImage\":{rows_per_image}}},"
                );
                self.extent(*size);
                self.out.push(']');
                self.close();
            }
            Command::BeginRenderPass { colors } => {
                self.encoder();
                self.step("\"enc\"", "beginRenderPass");
                self.out.push_str("[{\"colorAttachments\":[");
                for (i, a) in colors.iter().enumerate() {
                    if i > 0 {
                        self.out.push(',');
                    }
                    self.out.push_str("{\"view\":");
                    self.obj(Obj(a.view));
                    if let Some(r) = a.resolve {
                        self.out.push_str(",\"resolveTarget\":");
                        self.obj(Obj(r));
                    }
                    let _ = write!(
                        self.out,
                        ",\"loadOp\":\"{}\",\"storeOp\":\"{}\",\"clearValue\":[{},{},{},{}]}}",
                        a.load.name(),
                        a.store.name(),
                        a.clear[0],
                        a.clear[1],
                        a.clear[2],
                        a.clear[3]
                    );
                }
                self.out.push_str("]}]");
                self.yields_reg("pass");
            }
            Command::BeginComputePass => {
                self.encoder();
                self.step("\"enc\"", "beginComputePass");
                self.out.push_str("[]");
                self.yields_reg("pass");
            }
            Command::SetPipeline { pipeline } => {
                self.step("\"pass\"", "setPipeline");
                self.out.push('[');
                self.obj(Obj(*pipeline));
                self.out.push(']');
                self.close();
            }
            Command::SetBindGroup { index, group } => {
                self.step("\"pass\"", "setBindGroup");
                let _ = write!(self.out, "[{index},");
                self.obj(Obj(*group));
                self.out.push(']');
                self.close();
            }
            Command::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => {
                self.step("\"pass\"", "setVertexBuffer");
                let _ = write!(self.out, "[{slot},");
                self.obj(Obj(*buffer));
                let _ = write!(self.out, ",{offset}");
                if *size != 0 {
                    let _ = write!(self.out, ",{size}");
                }
                self.out.push(']');
                self.close();
            }
            Command::SetIndexBuffer {
                buffer,
                format,
                offset,
                size,
            } => {
                self.step("\"pass\"", "setIndexBuffer");
                self.out.push('[');
                self.obj(Obj(*buffer));
                let _ = write!(self.out, ",\"{}\",{offset}", format.name());
                if *size != 0 {
                    let _ = write!(self.out, ",{size}");
                }
                self.out.push(']');
                self.close();
            }
            Command::SetScissor {
                x,
                y,
                width,
                height,
            } => {
                self.step("\"pass\"", "setScissorRect");
                let _ = write!(self.out, "[{x},{y},{width},{height}]");
                self.close();
            }
            Command::Draw {
                vertices,
                instances,
                first_vertex,
                first_instance,
            } => {
                self.step("\"pass\"", "draw");
                let _ = write!(
                    self.out,
                    "[{vertices},{instances},{first_vertex},{first_instance}]"
                );
                self.close();
            }
            Command::DrawIndexed {
                indices,
                instances,
                first_index,
                base_vertex,
                first_instance,
            } => {
                self.step("\"pass\"", "drawIndexed");
                let _ = write!(
                    self.out,
                    "[{indices},{instances},{first_index},{base_vertex},{first_instance}]"
                );
                self.close();
            }
            Command::Dispatch { x, y, z } => {
                self.step("\"pass\"", "dispatchWorkgroups");
                let _ = write!(self.out, "[{x},{y},{z}]");
                self.close();
            }
            Command::EndPass => {
                self.step("\"pass\"", "end");
                self.out.push_str("[]");
                self.close();
            }
            Command::CopyBufferToBuffer {
                src,
                dst,
                src_offset,
                dst_offset,
                size,
            } => {
                self.encoder();
                self.step("\"enc\"", "copyBufferToBuffer");
                self.out.push('[');
                self.obj(Obj(*src));
                let _ = write!(self.out, ",{src_offset},");
                self.obj(Obj(*dst));
                let _ = write!(self.out, ",{dst_offset},{size}]");
                self.close();
            }
            Command::CopyBufferToTexture {
                buffer,
                texture,
                size,
            } => {
                self.encoder();
                self.step("\"enc\"", "copyBufferToTexture");
                self.out.push('[');
                self.buffer_copy(buffer);
                self.out.push(',');
                self.texture_copy(texture);
                self.out.push(',');
                self.extent(*size);
                self.out.push(']');
                self.close();
            }
            Command::CopyTextureToBuffer {
                buffer,
                texture,
                size,
            } => {
                self.encoder();
                self.step("\"enc\"", "copyTextureToBuffer");
                self.out.push('[');
                self.texture_copy(texture);
                self.out.push(',');
                self.buffer_copy(buffer);
                self.out.push(',');
                self.extent(*size);
                self.out.push(']');
                self.close();
            }
            Command::Readback {
                buffer,
                offset,
                size,
            } => {
                self.flush();
                self.step("\"js\"", "readback");
                let _ = write!(self.out, "[{buffer},{offset},{size}]");
                self.close();
            }
            Command::Acquire { texture, view } => {
                self.step("\"ctx\"", "getCurrentTexture");
                self.out.push_str("[]");
                self.yields(*texture);
                self.step(&alloc::format!("{texture}"), "createView");
                self.out.push_str("[]");
                self.yields(*view);
            }
            Command::Present { texture, view } => {
                self.flush();
                // The browser presents a canvas when the task that drew it ends; the executor frees
                // the frame's two ids, as the session did.
                self.step("\"js\"", "present");
                let _ = write!(self.out, "[{texture},{view}]");
                self.close();
            }
            Command::Destroy { id } => {
                self.flush();
                self.step("\"js\"", "destroy");
                self.out.push('[');
                self.num(id);
                self.out.push(']');
                self.close();
            }
        }
    }
}
