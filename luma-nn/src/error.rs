use luma_tensor::Shape;

#[thiserrorctx::context_error]
pub enum NnError {
    #[error(transparent)]
    Core(#[from] luma_tensor::Error),

    #[error(transparent)]
    SafeTensors(#[from] luma_io::safetensors::SafeTensorsError),

    #[error(transparent)]
    LumaPack(#[from] luma_io::lpk::LumaPackError),

    #[error("can't found param {0} in {1}")]
    ParamNotFound(String, &'static str),

    #[error("shape unmatch when load param: expect {0}, but got {1}")]
    ShapeUnmatchWhenLoadParam(Shape, Shape),

    #[error("head_size {0} can't divde by num_head {1}")]
    HeadSizeCannotDivideByNumhead(usize, usize),

    #[error("head_size {0} can't divde by kv_num_head {1}")]
    HeadSizeCannotDivideByKvNumhead(usize, usize),

    #[error("unsupport shape {0} of input in batch norm 1d")]
    BatchNorm1dUnsupportShape(Shape),

    #[error("drop_p {0} invalid(not in [0, 1)])")]
    DropoutInvalid(f64),

    #[error("unsuppor activate {0}")]
    UnsupportActivate(String),
}
