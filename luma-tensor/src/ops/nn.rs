use crate::{Device, Dim, Float, FloatMeta, Tensor, Storage};

impl<D: Device> Tensor<D, Float> {
    /// Softmax over `dim`. Records `Op::Softmax`.
    pub fn softmax<Dm: Dim>(&self, dim: Dm) -> crate::Result<Self> {
        let dim = dim.to_index(self.shape(), "softmax")?;
        let storage = D::f_softmax(&*self.storage_read()?, self.layout(), dim)?;
        let meta = FloatMeta::on_softmax(self, dim);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }

    /// RMSNorm over the last dim: `x / sqrt(mean(x^2)+eps) * weight`.
    /// Records `Op::RmsNorm`.
    pub fn rms_norm(&self, weight: &Self, eps: f64) -> crate::Result<Self> {
        let storage = D::f_rms_norm(&*self.storage_read()?, self.layout(), &*weight.storage_read()?, weight.layout(), eps)?;
        let meta = FloatMeta::on_rms_norm(self, weight, eps);
        assert_eq!(self.dtype(), storage.dtype());
        Ok(Self::from_storage(storage, self.shape().clone(), meta))
    }
}
