use crate::{CompileResult, Graph};
use std::path::Path;

pub fn write<P: AsRef<Path>>(_graph: &Graph, _path: P) -> CompileResult<()> {
    unimplemented!("write graph to luma.mlir");
}
