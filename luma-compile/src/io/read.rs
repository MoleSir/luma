use crate::{CompileResult, Graph};
use std::path::Path;

pub fn read<P: AsRef<Path>>(_path: P) -> CompileResult<Graph> {
    unimplemented!("read luma.mlir");
}
