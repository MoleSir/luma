use std::fmt::Display;

use crate::{Device, Float, FloatMeta, NoGradGuard, Shape, Tensor};

#[derive(Debug)]
pub struct CustomOpError(pub Box<dyn std::error::Error + 'static + Sync + Send>);

/// A custom op computes its output `Storage` + `Shape` directly (typically from
/// a hand-written kernel); the framework assembles the output `Tensor` and
/// attaches the op node for autograd. `forward` runs as a black box under
/// `no_grad`; gradients are the op's `backward`'s responsibility.
pub trait CustomOp1<Dev: Device> {
    fn name(&self) -> String;

    /// ## Forward
    /// ret = op(arg)
    fn forward(&self, arg: &Tensor<Dev>) -> Result<(Dev::FloatStorage, Shape), CustomOpError>;

    /// ## Backward
    /// arg_grad = back_op(arg, ret, ret_grad)
    fn backward(&self, arg: &Tensor<Dev>, ret: &Tensor<Dev>, ret_grad: &Tensor<Dev>) -> Result<Tensor<Dev>, CustomOpError>;
}

pub trait CustomOp2<Dev: Device> {
    fn name(&self) -> String;

    /// ## Forward
    /// ret = op(arg1, arg2)
    fn forward(&self, arg1: &Tensor<Dev>, arg2: &Tensor<Dev>) -> Result<(Dev::FloatStorage, Shape), CustomOpError>;

    /// ## Backward
    /// (arg1_grad, arg2_grad) = back_op(arg1, arg2, ret, ret_grad)
    fn backward(
        &self, 
        arg1: &Tensor<Dev>, arg2: &Tensor<Dev>,
        ret: &Tensor<Dev>, ret_grad: &Tensor<Dev>
    ) -> Result<(Tensor<Dev>, Tensor<Dev>), CustomOpError>;
}

pub trait CustomOp3<Dev: Device> {
    fn name(&self) -> String;

    /// ## Forward
    /// ret = op(arg1, arg2, arg3)
    fn forward(&self, arg1: &Tensor<Dev>, arg2: &Tensor<Dev>, arg3: &Tensor<Dev>) -> Result<(Dev::FloatStorage, Shape), CustomOpError>;

    /// ## Backward
    /// (arg1_grad, arg2_grad, arg3_grad) = back_op(arg1, arg2, arg3, ret, ret_grad)
    fn backward(
        &self, 
        arg1: &Tensor<Dev>, arg2: &Tensor<Dev>, arg3: &Tensor<Dev>,
        ret: &Tensor<Dev>, ret_grad: &Tensor<Dev>
    ) -> Result<(Tensor<Dev>, Tensor<Dev>, Tensor<Dev>), CustomOpError>;
}

pub trait CustomOp<Dev: Device> {
    fn name(&self) -> String;

    /// ## Forward
    /// ret = op(args)
    fn forward(&self, args: &[Tensor<Dev>]) -> Result<(Dev::FloatStorage, Shape), CustomOpError>;

    /// ## Backward
    /// arg_gards = back_op(args, ret, ret_grad)
    fn backward(&self, args: &[Tensor<Dev>], ret: &Tensor<Dev>, ret_grad: &Tensor<Dev>) -> Result<Vec<Tensor<Dev>>, CustomOpError>;
}

impl<D: Device> Tensor<D, Float> {
    /// `ret = op(self)`; records `Op::CustomOp1` when grad is required.
    pub fn custom_op1(&self, op: Box<dyn CustomOp1<D> + Send + Sync>) -> crate::Result<Self> {
        let (storage, shape) = {
            let _guard = NoGradGuard::new();
            op.forward(self)?
        };
        let meta = FloatMeta::on_custom_op1(self, op);
        Ok(Self::from_storage(storage, shape, meta))
    }

    /// `ret = op(self, arg2)`; records `Op::CustomOp2` when grad is required.
    pub fn custom_op2(&self, arg2: &Self, op: Box<dyn CustomOp2<D> + Send + Sync>) -> crate::Result<Self> {
        let (storage, shape) = {
            let _guard = NoGradGuard::new();
            op.forward(self, arg2)?
        };
        let meta = FloatMeta::on_custom_op2(self, arg2, op);
        Ok(Self::from_storage(storage, shape, meta))
    }

    /// `ret = op(self, arg2, arg3)`; records `Op::CustomOp3` when grad is required.
    pub fn custom_op3(&self, arg2: &Self, arg3: &Self, op: Box<dyn CustomOp3<D> + Send + Sync>) -> crate::Result<Self> {
        let (storage, shape) = {
            let _guard = NoGradGuard::new();
            op.forward(self, arg2, arg3)?
        };
        let meta = FloatMeta::on_custom_op3(self, arg2, arg3, op);
        Ok(Self::from_storage(storage, shape, meta))
    }

    /// `ret = op(args)`; records `Op::CustomOp` when grad is required.
    pub fn custom_op(args: &[Self], op: Box<dyn CustomOp<D> + Send + Sync>) -> crate::Result<Self> {
        let (storage, shape) = {
            let _guard = NoGradGuard::new();
            op.forward(args)?
        };
        let meta = FloatMeta::on_custom_op(args, op);
        Ok(Self::from_storage(storage, shape, meta))
    }
}

impl CustomOpError {
    pub fn msg<S: Into<String>>(s: S) -> Self {
        let s = s.into();
        Self( Box::new(MsgError(s)) )
    }
}

/// Let op authors use `?` on tensor ops inside `forward`/`backward`.
impl From<crate::Error> for CustomOpError {
    fn from(e: crate::Error) -> Self {
        Self(Box::new(e))
    }
}

#[derive(Debug)]
struct MsgError(String);

impl Display for MsgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for MsgError {}

impl Display for CustomOpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CustomOpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0) 
    }
}