//! The protocol's enumerations: each a small integer code on the wire, and the WebGPU name the
//! browser's form uses. A code outside the table is refused (fail closed).

/// One table per enum: its codes, and the WebGPU string each stands for.
macro_rules! codes {
    ($(#[$m:meta])* $name:ident { $($v:ident = $c:literal => $s:literal,)* }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name { $($v = $c,)* }

        impl $name {
            /// The enum a wire code names, or `None` for any other code.
            pub fn from_code(c: u32) -> Option<Self> {
                match c { $($c => Some(Self::$v),)* _ => None }
            }
            /// Its wire code.
            pub fn code(self) -> u32 { self as u32 }
            /// Its WebGPU name.
            pub fn name(self) -> &'static str {
                match self { $(Self::$v => $s,)* }
            }
        }
    };
}

codes! {
    /// The texture formats a guest may name: the colour formats a compositor and a compute kernel
    /// use. Depth and compressed formats come with a consumer.
    TextureFormat {
        R8Unorm = 1 => "r8unorm",
        Rg8Unorm = 2 => "rg8unorm",
        Rgba8Unorm = 3 => "rgba8unorm",
        Rgba8UnormSrgb = 4 => "rgba8unorm-srgb",
        Bgra8Unorm = 5 => "bgra8unorm",
        Bgra8UnormSrgb = 6 => "bgra8unorm-srgb",
        R16Float = 7 => "r16float",
        Rg16Float = 8 => "rg16float",
        Rgba16Float = 9 => "rgba16float",
        R32Float = 10 => "r32float",
        Rg32Float = 11 => "rg32float",
        Rgba32Float = 12 => "rgba32float",
        R32Uint = 13 => "r32uint",
        Rgb10a2Unorm = 14 => "rgb10a2unorm",
    }
}

codes! {
    TextureDimension {
        D1 = 1 => "1d",
        D2 = 2 => "2d",
        D3 = 3 => "3d",
    }
}

codes! {
    VertexFormat {
        Float32 = 1 => "float32",
        Float32x2 = 2 => "float32x2",
        Float32x3 = 3 => "float32x3",
        Float32x4 = 4 => "float32x4",
        Uint32 = 5 => "uint32",
        Uint32x2 = 6 => "uint32x2",
        Uint32x3 = 7 => "uint32x3",
        Uint32x4 = 8 => "uint32x4",
        Sint32 = 9 => "sint32",
        Sint32x2 = 10 => "sint32x2",
        Sint32x3 = 11 => "sint32x3",
        Sint32x4 = 12 => "sint32x4",
        Unorm8x4 = 13 => "unorm8x4",
        Uint8x4 = 14 => "uint8x4",
        Uint16x2 = 15 => "uint16x2",
        Sint16x2 = 16 => "sint16x2",
        Unorm16x2 = 17 => "unorm16x2",
        Float16x2 = 18 => "float16x2",
        Float16x4 = 19 => "float16x4",
    }
}

codes! {
    StepMode {
        Vertex = 1 => "vertex",
        Instance = 2 => "instance",
    }
}

codes! {
    Topology {
        PointList = 1 => "point-list",
        LineList = 2 => "line-list",
        LineStrip = 3 => "line-strip",
        TriangleList = 4 => "triangle-list",
        TriangleStrip = 5 => "triangle-strip",
    }
}

codes! {
    /// An index format; 0 on the wire, where it is optional, is none.
    IndexFormat {
        Uint16 = 1 => "uint16",
        Uint32 = 2 => "uint32",
    }
}

codes! {
    FrontFace {
        Ccw = 1 => "ccw",
        Cw = 2 => "cw",
    }
}

codes! {
    /// The face culled; 0 on the wire is none, WebGPU's default.
    Face {
        Front = 1 => "front",
        Back = 2 => "back",
    }
}

codes! {
    /// The blend factors, less the blend constant's, which come with `setBlendConstant`.
    BlendFactor {
        Zero = 1 => "zero",
        One = 2 => "one",
        Src = 3 => "src",
        OneMinusSrc = 4 => "one-minus-src",
        SrcAlpha = 5 => "src-alpha",
        OneMinusSrcAlpha = 6 => "one-minus-src-alpha",
        Dst = 7 => "dst",
        OneMinusDst = 8 => "one-minus-dst",
        DstAlpha = 9 => "dst-alpha",
        OneMinusDstAlpha = 10 => "one-minus-dst-alpha",
        SrcAlphaSaturated = 11 => "src-alpha-saturated",
    }
}

codes! {
    BlendOperation {
        Add = 1 => "add",
        Subtract = 2 => "subtract",
        ReverseSubtract = 3 => "reverse-subtract",
        Min = 4 => "min",
        Max = 5 => "max",
    }
}

codes! {
    LoadOp {
        Clear = 1 => "clear",
        Load = 2 => "load",
    }
}

codes! {
    StoreOp {
        Store = 1 => "store",
        Discard = 2 => "discard",
    }
}

codes! {
    AddressMode {
        ClampToEdge = 1 => "clamp-to-edge",
        Repeat = 2 => "repeat",
        MirrorRepeat = 3 => "mirror-repeat",
    }
}

codes! {
    FilterMode {
        Nearest = 1 => "nearest",
        Linear = 2 => "linear",
    }
}

codes! {
    /// A sampler's comparison; 0 on the wire is none.
    CompareFunction {
        Never = 1 => "never",
        Less = 2 => "less",
        Equal = 3 => "equal",
        LessEqual = 4 => "less-equal",
        Greater = 5 => "greater",
        NotEqual = 6 => "not-equal",
        GreaterEqual = 7 => "greater-equal",
        Always = 8 => "always",
    }
}

/// `GPUBufferUsage`'s bits, as WebGPU numbers them. A guest buffer takes no map bits: readback
/// goes through the host's own staging (`Command::Readback`), and the protocol has no indirect
/// draws or queries yet.
pub mod buffer_usage {
    pub const COPY_SRC: u32 = 0x0004;
    pub const COPY_DST: u32 = 0x0008;
    pub const INDEX: u32 = 0x0010;
    pub const VERTEX: u32 = 0x0020;
    pub const UNIFORM: u32 = 0x0040;
    pub const STORAGE: u32 = 0x0080;
    /// Every bit a guest may set.
    pub const ALLOWED: u32 = COPY_SRC | COPY_DST | INDEX | VERTEX | UNIFORM | STORAGE;
}

/// `GPUTextureUsage`'s bits, as WebGPU numbers them.
pub mod texture_usage {
    pub const COPY_SRC: u32 = 0x01;
    pub const COPY_DST: u32 = 0x02;
    pub const TEXTURE_BINDING: u32 = 0x04;
    pub const STORAGE_BINDING: u32 = 0x08;
    pub const RENDER_ATTACHMENT: u32 = 0x10;
    pub const ALLOWED: u32 =
        COPY_SRC | COPY_DST | TEXTURE_BINDING | STORAGE_BINDING | RENDER_ATTACHMENT;
}

/// `GPUColorWrite`'s bits.
pub mod color_write {
    pub const RED: u32 = 0x1;
    pub const GREEN: u32 = 0x2;
    pub const BLUE: u32 = 0x4;
    pub const ALPHA: u32 = 0x8;
    pub const ALL: u32 = 0xF;
}
