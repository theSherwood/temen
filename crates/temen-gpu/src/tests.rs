use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::command::*;
use crate::enums::*;
use crate::*;

fn stream(cmds: &[Command<'_>]) -> Vec<u8> {
    let mut e = Encoder::new();
    for c in cmds {
        e.push(c);
    }
    e.bytes
}

const WGSL: &str = "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }";

fn pipeline() -> RenderPipeline<'static> {
    RenderPipeline {
        id: 5,
        module: 4,
        vertex: "vs",
        fragment: "fs",
        topology: Topology::TriangleList,
        strip_index_format: None,
        front_face: FrontFace::Ccw,
        cull: Some(Face::Back),
        samples: 1,
        buffers: vec![VertexBuffer {
            stride: 16,
            step: StepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x2,
                    location: 0,
                    offset: 0,
                },
                VertexAttribute {
                    format: VertexFormat::Unorm8x4,
                    location: 1,
                    offset: 8,
                },
            ],
        }],
        targets: vec![
            ColorTarget {
                format: TextureFormat::Rgba8Unorm,
                write_mask: color_write::ALL,
                blend: Some(Blend {
                    color: BlendComponent {
                        src: BlendFactor::SrcAlpha,
                        dst: BlendFactor::OneMinusSrcAlpha,
                        op: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src: BlendFactor::One,
                        dst: BlendFactor::Zero,
                        op: BlendOperation::Add,
                    },
                }),
            },
            ColorTarget {
                format: TextureFormat::R8Unorm,
                write_mask: color_write::RED,
                blend: None,
            },
        ],
    }
}

/// One of every command, in an order a session accepts.
fn every() -> Vec<Command<'static>> {
    use buffer_usage as bu;
    use texture_usage as tu;
    vec![
        Command::CreateBuffer {
            id: 1,
            size: 256,
            usage: bu::VERTEX | bu::COPY_DST | bu::COPY_SRC | bu::INDEX | bu::UNIFORM,
        },
        Command::CreateTexture(Texture {
            id: 2,
            format: TextureFormat::Rgba8Unorm,
            usage: tu::RENDER_ATTACHMENT | tu::COPY_SRC | tu::COPY_DST | tu::TEXTURE_BINDING,
            dimension: TextureDimension::D2,
            size: [64, 32, 1],
            mips: 1,
            samples: 1,
        }),
        Command::CreateView { id: 3, texture: 2 },
        Command::CreateShaderModule { id: 4, wgsl: WGSL },
        Command::CreateRenderPipeline(pipeline()),
        Command::CreateSampler(Sampler {
            id: 6,
            address: [
                AddressMode::ClampToEdge,
                AddressMode::Repeat,
                AddressMode::MirrorRepeat,
            ],
            mag: FilterMode::Linear,
            min: FilterMode::Nearest,
            mipmap: FilterMode::Nearest,
            compare: None,
            anisotropy: 1,
            lod_min: 0.0,
            lod_max: 32.0,
        }),
        Command::BindGroupLayoutOf {
            id: 7,
            pipeline: 5,
            group: 0,
        },
        Command::CreateBindGroup {
            id: 8,
            layout: 7,
            entries: vec![
                BindEntry {
                    binding: 0,
                    resource: Resource::Buffer {
                        id: 1,
                        offset: 0,
                        size: 64,
                    },
                },
                BindEntry {
                    binding: 1,
                    resource: Resource::Sampler(6),
                },
                BindEntry {
                    binding: 2,
                    resource: Resource::View(3),
                },
            ],
        },
        Command::WriteBuffer {
            buffer: 1,
            offset: 16,
            data: b"abcdefghijk",
        },
        Command::WriteTexture {
            texture: TextureCopy {
                texture: 2,
                mip: 0,
                origin: [1, 2, 0],
            },
            size: [2, 1, 1],
            bytes_per_row: 8,
            rows_per_image: 1,
            data: &[9; 8],
        },
        Command::BeginRenderPass {
            colors: vec![ColorAttachment {
                view: 3,
                resolve: None,
                load: LoadOp::Clear,
                store: StoreOp::Store,
                clear: [0.0, 0.25, 0.5, 1.0],
            }],
        },
        Command::SetPipeline { pipeline: 5 },
        Command::SetBindGroup { index: 0, group: 8 },
        Command::SetVertexBuffer {
            slot: 0,
            buffer: 1,
            offset: 0,
            size: 0,
        },
        Command::SetIndexBuffer {
            buffer: 1,
            format: IndexFormat::Uint16,
            offset: 128,
            size: 12,
        },
        Command::SetScissor {
            x: 1,
            y: 2,
            width: 3,
            height: 4,
        },
        Command::Draw {
            vertices: 6,
            instances: 2,
            first_vertex: 0,
            first_instance: 1,
        },
        Command::DrawIndexed {
            indices: 6,
            instances: 1,
            first_index: 0,
            base_vertex: -3,
            first_instance: 0,
        },
        Command::EndPass,
        Command::CreateShaderModule {
            id: 9,
            wgsl: "@compute @workgroup_size(1) fn main() {}",
        },
        Command::CreateComputePipeline {
            id: 10,
            module: 9,
            entry: "main",
        },
        Command::BeginComputePass,
        Command::SetPipeline { pipeline: 10 },
        Command::Dispatch { x: 4, y: 1, z: 1 },
        Command::EndPass,
        Command::CopyBufferToBuffer {
            src: 1,
            dst: 1,
            src_offset: 0,
            dst_offset: 128,
            size: 64,
        },
        Command::CopyBufferToTexture {
            buffer: BufferCopy {
                buffer: 1,
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
            texture: TextureCopy {
                texture: 2,
                mip: 0,
                origin: [0, 0, 0],
            },
            size: [1, 1, 1],
        },
        Command::CopyTextureToBuffer {
            buffer: BufferCopy {
                buffer: 1,
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
            texture: TextureCopy {
                texture: 2,
                mip: 0,
                origin: [0, 0, 0],
            },
            size: [1, 1, 1],
        },
        Command::Readback {
            buffer: 1,
            offset: 0,
            size: 4,
        },
        Command::Acquire {
            texture: 11,
            view: 12,
        },
        Command::Present {
            texture: 11,
            view: 12,
        },
        Command::Destroy { id: 8 },
    ]
}

#[test]
fn every_command_round_trips() {
    let cmds = every();
    let bytes = stream(&cmds);
    assert_eq!(bytes.len() % 8, 0);
    assert_eq!(decode(&bytes).unwrap(), cmds);
    // Every command code is used once, so the stream covers the whole protocol.
    let mut codes: Vec<u32> = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        codes.push(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()));
        at += 8 + u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
    }
    codes.sort();
    codes.dedup();
    assert_eq!(codes, (1..=29).collect::<Vec<u32>>());
    assert_eq!(Session::new().check(&cmds), Ok(()));
}

/// Decoding `bytes` refuses them, saying `what`.
fn refused(bytes: &[u8], what: &str) {
    match decode(bytes) {
        Ok(c) => panic!("decoded {c:?}"),
        Err(e) => assert_eq!(e.what, what),
    }
}

#[test]
fn a_malformed_stream_is_refused_whole() {
    let one = stream(&[Command::Draw {
        vertices: 3,
        instances: 1,
        first_vertex: 0,
        first_instance: 0,
    }]);
    refused(&one[..5], "a record header is cut short");
    refused(&one[..one.len() - 8], "a record runs past the stream");
    let mut b = one.clone();
    b[4] = 12;
    refused(&b, "a record's size is not a multiple of 8");
    let mut b = one.clone();
    b[0] = 99;
    refused(&b, "no such command");
    // A record longer than its command's fields.
    let mut b = one.clone();
    b[4] = 24;
    b.extend_from_slice(&[0; 8]);
    refused(&b, "a record is longer than its fields");

    // A nonzero padding word.
    let mut b = stream(&[Command::SetPipeline { pipeline: 1 }]);
    b[12] = 1;
    refused(&b, "a padding field is not zero");
    // Nonzero padding after data.
    let mut b = stream(&[Command::CreateShaderModule { id: 1, wgsl: "x" }]);
    let last = b.len() - 1;
    b[last] = 1;
    refused(&b, "padding after the data is not zero");
    // A string that is not UTF-8.
    let mut b = stream(&[Command::CreateShaderModule { id: 1, wgsl: "x" }]);
    b[16] = 0xff;
    refused(&b, "a string is not UTF-8");
    // An enum code no table names: a texture format of 0, then of 99.
    let tex = |format: u32| {
        let mut b = stream(&[Command::CreateTexture(Texture {
            id: 1,
            format: TextureFormat::R8Unorm,
            usage: 1,
            dimension: TextureDimension::D2,
            size: [1, 1, 1],
            mips: 1,
            samples: 1,
        })]);
        b[12..16].copy_from_slice(&format.to_le_bytes());
        b
    };
    refused(&tex(0), "a code no table names");
    refused(&tex(99), "a code no table names");
    // A count over its limit.
    let mut b = stream(&[Command::BeginRenderPass { colors: vec![] }]);
    b[8] = 9;
    refused(&b, "a count is over its limit");
    // A clear value that is not finite.
    let mut b = stream(&[Command::BeginRenderPass {
        colors: vec![ColorAttachment {
            view: 1,
            resolve: None,
            load: LoadOp::Clear,
            store: StoreOp::Store,
            clear: [0.0; 4],
        }],
    }]);
    b[32..40].copy_from_slice(&f64::NAN.to_le_bytes());
    refused(&b, "a clear value is not finite");
    // A target without blending that names blend factors.
    let mut p = pipeline();
    p.targets.truncate(1);
    p.targets[0].blend = None;
    let mut b = stream(&[Command::CreateRenderPipeline(p)]);
    let at = b.len() - 40 + 12;
    b[at] = 2;
    refused(&b, "a target without blending names blend factors");
}

/// `cmds` on a fresh session are refused at `index`, saying `what`, and leave nothing behind.
fn check_refused(prefix: &[Command<'_>], cmds: &[Command<'_>], index: usize, what: &str) {
    let mut s = Session::new();
    s.check(prefix).expect("the prefix is accepted");
    let r = s.check(cmds).expect_err("refused");
    assert_eq!((r.index, r.what), (index, what), "{r:?}");
    // The refusal changed nothing: the ids the stream would have created are still free.
    for c in cmds {
        if let Command::CreateBuffer { id, .. } = c {
            if !prefix
                .iter()
                .any(|p| matches!(p, Command::CreateBuffer { id: q, .. } if q == id))
            {
                assert_eq!(s.kind(*id), None, "a refused stream left id {id}");
            }
        }
    }
}

fn buffer(id: Id, size: u64, usage: u32) -> Command<'static> {
    Command::CreateBuffer { id, size, usage }
}

#[test]
fn a_session_refuses_what_names_no_object_and_changes_nothing() {
    use buffer_usage as bu;
    let draw = Command::Draw {
        vertices: 3,
        instances: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    check_refused(
        &[],
        &[buffer(0, 4, bu::VERTEX)],
        0,
        "an id is 0 or over the limit",
    );
    check_refused(
        &[],
        &[buffer(session::MAX_IDS, 4, bu::VERTEX)],
        0,
        "an id is 0 or over the limit",
    );
    check_refused(
        &[buffer(1, 4, bu::VERTEX)],
        &[buffer(2, 4, bu::VERTEX), buffer(1, 4, bu::VERTEX)],
        1,
        "an id is already in use",
    );
    check_refused(
        &[],
        &[buffer(1, 4, 0x1)],
        0,
        "a buffer needs a size, and usages from the allowed set",
    );
    check_refused(
        &[buffer(1, 16, bu::COPY_DST)],
        &[Command::WriteBuffer {
            buffer: 1,
            offset: 12,
            data: b"12345",
        }],
        0,
        "a range runs past its buffer",
    );
    check_refused(
        &[buffer(1, 16, bu::COPY_DST)],
        &[Command::CreateView { id: 2, texture: 1 }],
        0,
        "an id names an object of another kind",
    );
    check_refused(
        &[],
        core::slice::from_ref(&draw),
        0,
        "only inside a render pass",
    );
    check_refused(
        &[],
        &[Command::BeginComputePass, draw.clone()],
        1,
        "only inside a render pass",
    );
    check_refused(
        &[],
        &[Command::BeginComputePass, Command::BeginComputePass],
        1,
        "only outside a pass",
    );
    check_refused(&[], &[Command::BeginComputePass], 1, "a pass is still open");
    check_refused(&[], &[Command::EndPass], 0, "no pass is open");
    check_refused(
        &[buffer(1, 16, bu::COPY_DST)],
        &[Command::Readback {
            buffer: 1,
            offset: 0,
            size: 4,
        }],
        0,
        "a readback needs a buffer with COPY_SRC",
    );
    check_refused(
        &[],
        &[
            Command::BeginRenderPass { colors: vec![] },
            Command::EndPass,
        ],
        0,
        "a render pass has a colour attachment",
    );
    check_refused(
        &[buffer(1, 16, bu::COPY_SRC)],
        &[Command::Readback {
            buffer: 1,
            offset: 2,
            size: 4,
        }],
        0,
        "a readback's offset and size are multiples of 4, and it has a size",
    );
    let acquire = Command::Acquire {
        texture: 3,
        view: 4,
    };
    check_refused(
        core::slice::from_ref(&acquire),
        &[Command::Destroy { id: 3 }],
        0,
        "the surface texture is freed by present",
    );
    check_refused(
        core::slice::from_ref(&acquire),
        &[Command::Present {
            texture: 3,
            view: 5,
        }],
        0,
        "present names the acquired texture and its view",
    );
    check_refused(
        &[],
        &[Command::Present {
            texture: 3,
            view: 4,
        }],
        0,
        "present names the acquired texture and its view",
    );
}

#[test]
fn a_pass_refuses_what_its_gpu_api_would_panic_on() {
    use buffer_usage as bu;
    let prefix = [
        buffer(1, 16, bu::VERTEX),
        Command::CreateTexture(Texture {
            id: 2,
            format: TextureFormat::Rgba8Unorm,
            usage: texture_usage::RENDER_ATTACHMENT,
            dimension: TextureDimension::D2,
            size: [4, 4, 1],
            mips: 1,
            samples: 1,
        }),
        Command::CreateView { id: 3, texture: 2 },
    ];
    let pass = |c: Command<'static>| {
        vec![
            Command::BeginRenderPass {
                colors: vec![ColorAttachment {
                    view: 3,
                    resolve: None,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    clear: [0.0; 4],
                }],
            },
            c,
            Command::EndPass,
        ]
    };
    check_refused(
        &prefix,
        &pass(Command::Draw {
            vertices: 2,
            instances: 1,
            first_vertex: u32::MAX,
            first_instance: 0,
        }),
        1,
        "a draw's range runs past u32",
    );
    check_refused(
        &prefix,
        &pass(Command::SetVertexBuffer {
            slot: 0,
            buffer: 1,
            offset: 16,
            size: 0,
        }),
        1,
        "a vertex or index slice is empty",
    );
    check_refused(
        &prefix,
        &pass(Command::SetVertexBuffer {
            slot: 0,
            buffer: 1,
            offset: 8,
            size: 16,
        }),
        1,
        "a range runs past its buffer",
    );
    let mut s = Session::new();
    s.check(&prefix).unwrap();
    s.check(&pass(Command::SetVertexBuffer {
        slot: 0,
        buffer: 1,
        offset: 8,
        size: 0,
    }))
    .unwrap();
}

#[test]
fn a_refused_stream_leaves_the_acquired_frame_as_it_was() {
    let mut s = Session::new();
    s.check(&[Command::Acquire {
        texture: 3,
        view: 4,
    }])
    .unwrap();
    // Present succeeds, then a later command fails: the present is undone too.
    let r = s.check(&[
        Command::Present {
            texture: 3,
            view: 4,
        },
        Command::EndPass,
    ]);
    assert!(r.is_err());
    assert_eq!(s.acquired(), Some((3, 4)));
    assert_eq!(s.kind(3), Some(Kind::Texture));
    assert_eq!(s.kind(4), Some(Kind::View));
    // Then it presents, and the ids are free again for the next frame.
    s.check(&[Command::Present {
        texture: 3,
        view: 4,
    }])
    .unwrap();
    assert_eq!((s.acquired(), s.kind(3), s.kind(4)), (None, None, None));
    s.check(&[Command::Acquire {
        texture: 3,
        view: 4,
    }])
    .unwrap();
}

/// A backend that records what it ran, and answers readbacks from a table.
#[derive(Default)]
struct Fake {
    ran: usize,
    ready: Vec<(Id, Vec<u8>)>,
    error: Option<String>,
}

impl Backend for Fake {
    fn run(&mut self, cmds: &[Command<'_>]) -> Result<(), Fault> {
        self.ran += cmds.len();
        Ok(())
    }
    fn readback(&mut self, buffer: Id) -> Readback {
        match self.ready.iter().position(|(b, _)| *b == buffer) {
            Some(i) => Readback::Ready(self.ready.remove(i).1),
            None if buffer == 7 => Readback::Pending,
            None => Readback::None,
        }
    }
    fn surface(&mut self) -> Option<SurfaceInfo> {
        Some(SurfaceInfo {
            width: 640,
            height: 480,
            format: TextureFormat::Bgra8Unorm,
        })
    }
    fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }
}

/// A window of 4 KiB.
struct Mem(Vec<u8>);

impl Window for Mem {
    fn read(&mut self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        let end = ptr.checked_add(len)? as usize;
        self.0.get(ptr as usize..end).map(|s| s.to_vec())
    }
    fn write(&mut self, ptr: u64, bytes: &[u8]) -> Option<()> {
        let end = (ptr as usize).checked_add(bytes.len())?;
        self.0.get_mut(ptr as usize..end)?.copy_from_slice(bytes);
        Some(())
    }
}

#[test]
fn the_ops_answer_as_the_table_says() {
    let mut gpu = Gpu::new(Fake::default());
    let mut mem = Mem(vec![0; 4096]);
    let ok = stream(&[buffer(1, 16, buffer_usage::COPY_SRC)]);
    mem.0[..ok.len()].copy_from_slice(&ok);
    let call = |gpu: &mut Gpu<Fake>, mem: &mut Mem, op, args: &[i64]| {
        gpu.call(op, args, Some(mem as &mut dyn Window))
    };
    assert_eq!(
        call(&mut gpu, &mut mem, op::SUBMIT, &[0, ok.len() as i64]),
        0
    );
    assert_eq!(gpu.backend.ran, 1);
    // No error yet.
    assert_eq!(call(&mut gpu, &mut mem, op::ERROR, &[1024, 256]), 0);

    // The same stream again: id 1 is in use, so it is refused, runs nothing, and says why.
    assert_eq!(
        call(&mut gpu, &mut mem, op::SUBMIT, &[0, ok.len() as i64]),
        errno::EINVAL
    );
    assert_eq!(gpu.backend.ran, 1);
    let n = call(&mut gpu, &mut mem, op::ERROR, &[1024, 256]) as usize;
    assert_eq!(
        core::str::from_utf8(&mem.0[1024..1024 + n]).unwrap(),
        "command 0 (create_buffer): an id is already in use"
    );
    assert_eq!(
        call(&mut gpu, &mut mem, op::ERROR, &[1024, 256]),
        0,
        "read once"
    );
    // A truncated copy still says the whole length.
    gpu.backend.error = Some("a long WebGPU error".to_string());
    assert_eq!(call(&mut gpu, &mut mem, op::ERROR, &[1024, 4]), 19);
    assert_eq!(&mem.0[1024..1028], b"a lo");

    // A stream outside the window, or too long, is refused before it is read.
    assert_eq!(
        call(&mut gpu, &mut mem, op::SUBMIT, &[4090, 16]),
        errno::EFAULT
    );
    assert_eq!(
        call(&mut gpu, &mut mem, op::SUBMIT, &[0, i64::MAX]),
        errno::EINVAL
    );

    // Readbacks: pending, then ready, then gone.
    assert_eq!(
        call(&mut gpu, &mut mem, op::READ, &[7, 2048, 8]),
        errno::EAGAIN
    );
    gpu.backend.ready.push((7, vec![1, 2, 3, 4, 5]));
    assert_eq!(call(&mut gpu, &mut mem, op::READ, &[7, 2048, 3]), 5);
    assert_eq!(&mem.0[2048..2052], &[1, 2, 3, 0]);
    assert_eq!(
        call(&mut gpu, &mut mem, op::READ, &[8, 2048, 8]),
        errno::ENOENT
    );

    // The surface.
    assert_eq!(call(&mut gpu, &mut mem, op::SURFACE, &[3072]), 0);
    assert_eq!(&mem.0[3072..3076], &640u32.to_le_bytes());
    assert_eq!(&mem.0[3076..3080], &480u32.to_le_bytes());
    assert_eq!(
        &mem.0[3080..3084],
        &TextureFormat::Bgra8Unorm.code().to_le_bytes()
    );
    assert_eq!(call(&mut gpu, &mut mem, 99, &[]), errno::EINVAL);
}

/// The browser's form of a small frame: a write, a pass, a readback and a present, each in its
/// place in stream order, with the encoder flushed before each queue operation.
#[test]
fn the_js_form_of_a_frame() {
    use buffer_usage as bu;
    let cmds = vec![
        buffer(1, 16, bu::VERTEX | bu::COPY_DST | bu::COPY_SRC),
        Command::Acquire {
            texture: 2,
            view: 3,
        },
        Command::BeginRenderPass {
            colors: vec![ColorAttachment {
                view: 3,
                resolve: None,
                load: LoadOp::Clear,
                store: StoreOp::Store,
                clear: [0.0, 0.5, 1.0, 1.0],
            }],
        },
        Command::SetVertexBuffer {
            slot: 0,
            buffer: 1,
            offset: 0,
            size: 0,
        },
        Command::Draw {
            vertices: 3,
            instances: 1,
            first_vertex: 0,
            first_instance: 0,
        },
        Command::EndPass,
        Command::WriteBuffer {
            buffer: 1,
            offset: 4,
            data: &[1, 2, 3, 4],
        },
        Command::Readback {
            buffer: 1,
            offset: 0,
            size: 8,
        },
        Command::Present {
            texture: 2,
            view: 3,
        },
    ];
    Session::new().check(&cmds).unwrap();
    let (mut out, mut blob) = (String::new(), Vec::new());
    js::to_js(&cmds, &mut out, &mut blob);
    assert_eq!(blob, [1, 2, 3, 4]);
    let want = concat!(
        r#"[["device","createBuffer",[{"size":16,"usage":44}],1],"#,
        r#"["ctx","getCurrentTexture",[],2],[2,"createView",[],3],"#,
        r#"["device","createCommandEncoder",[],"enc"],"#,
        r#"["enc","beginRenderPass",[{"colorAttachments":[{"view":{"$":3},"loadOp":"clear","storeOp":"store","clearValue":[0,0.5,1,1]}]}],"pass"],"#,
        r#"["pass","setVertexBuffer",[0,{"$":1},0]],"#,
        r#"["pass","draw",[3,1,0,0]],"#,
        r#"["pass","end",[]],"#,
        r#"["enc","finish",[],"cb"],["queue","submit",[[{"$":"cb"}]]],"#,
        r#"["queue","writeBuffer",[{"$":1},4,{"@":[0,4]}]],"#,
        r#"["js","readback",[1,0,8]],"#,
        r#"["js","present",[2,3]]]"#,
    );
    assert_eq!(out, want);
}

/// The JS form of every command is well-formed JSON with balanced brackets, and strings are
/// escaped.
#[test]
fn the_js_form_of_every_command_is_balanced() {
    let mut cmds = every();
    cmds.push(Command::CreateShaderModule {
        id: 20,
        wgsl: "// \"quoted\" \\ and\n\ta control \u{1}",
    });
    let (mut out, mut blob) = (String::new(), Vec::new());
    js::to_js(&cmds, &mut out, &mut blob);
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for ch in out.chars() {
        if in_str {
            match (esc, ch) {
                (true, _) => esc = false,
                (false, '\\') => esc = true,
                (false, '"') => in_str = false,
                (false, c) => assert!(c as u32 >= 0x20, "a raw control character"),
            }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            _ => {}
        }
        assert!(depth >= 0);
    }
    assert_eq!((depth, in_str), (0, false));
    assert!(out.contains(r#""// \"quoted\" \\ and\n\ta control \u0001""#));
}
