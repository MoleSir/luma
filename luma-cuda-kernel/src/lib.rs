mod ptx;

pub const BINARY: Module = Module { name: "binary", ptx: ptx::BINARY };
pub const UNARY: Module = Module { name: "unary", ptx: ptx::UNARY };
pub const REDUCE: Module = Module { name: "reduce", ptx: ptx::REDUCE };
pub const CAST: Module = Module { name: "cast", ptx: ptx::CAST };
pub const BINARY_SCALAR: Module = Module { name: "binary_scalar", ptx: ptx::BINARY_SCALAR };
pub const COPY: Module = Module { name: "copy", ptx: ptx::COPY };
pub const INDEXING: Module = Module { name: "indexing", ptx: ptx::INDEXING };
pub const PICK: Module = Module { name: "pick", ptx: ptx::PICK };
pub const ALLCLOSE: Module = Module { name: "allclose", ptx: ptx::ALLCLOSE };
pub const NN: Module = Module { name: "nn", ptx: ptx::NN };

pub struct Module {
    name: &'static str,
    ptx: &'static str,
}

impl Module {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}


