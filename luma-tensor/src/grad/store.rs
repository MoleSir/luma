use crate::{Device, Float, Tensor, TensorId, no_grad};
use std::collections::HashMap;
use std::ops::Index;

/// Maps tensor ids to their accumulated gradients during a backward pass.
/// Gradients are always `Float`-kind tensors on the same device.
pub struct GradStore<D: Device>(HashMap<TensorId, Tensor<D, Float>>);

impl<D: Device> GradStore<D> {
    pub fn new() -> Self {
        GradStore(HashMap::new())
    }

    pub fn get(&self, tensor: &Tensor<D, Float>) -> Option<&Tensor<D, Float>> {
        self.0.get(&tensor.id())
    }

    pub fn get_by_id(&self, id: TensorId) -> Option<&Tensor<D, Float>> {
        self.0.get(&id)
    }

    pub fn remove(&mut self, tensor: &Tensor<D, Float>) -> Option<Tensor<D, Float>> {
        self.0.remove(&tensor.id())
    }

    pub fn insert(&mut self, tensor: &Tensor<D, Float>, grad: Tensor<D, Float>) -> Option<Tensor<D, Float>> {
        self.0.insert(tensor.id(), grad)
    }

    /// Get the gradient accumulator for `tensor`, inserting a zeros tensor of the
    /// same shape/dtype if absent.
    pub fn or_insert(&mut self, tensor: &Tensor<D, Float>) -> crate::Result<&mut Tensor<D, Float>> {
        use std::collections::hash_map::Entry;
        let grad = match self.0.entry(tensor.id()) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(tensor.zeros_like()?),
        };
        Ok(grad)
    }

    pub fn get_ids(&self) -> impl Iterator<Item = &TensorId> {
        self.0.keys()
    }

    pub fn tensors(&self) -> impl Iterator<Item = &Tensor<D, Float>> {
        self.0.values()
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, TensorId, Tensor<D, Float>> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Global L2 norm of all gradients: `sqrt(sum ||g||^2)`.
    pub fn global_norm(&self) -> crate::Result<f64> {
        let mut total = 0.0f64;
        for g in self.0.values() {
            total += g.sqr()?.sum_all()?.to_scalar()?;
        }
        Ok(total.sqrt())
    }

    /// Clip all gradients by their global norm: if `norm > max_norm`, scale
    /// every gradient by `max_norm / norm`. Returns the (pre-clip) norm.
    pub fn clip_grad_norm(&mut self, max_norm: f64) -> crate::Result<f64> {
        no_grad!();
        let norm = self.global_norm()?;
        if norm > max_norm {
            let scale = max_norm / norm;
            for g in self.0.values_mut() {
                g.mul_scalar_(scale)?;
            }
        }
        Ok(norm)
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<D: Device> Default for GradStore<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Device> Index<&Tensor<D, Float>> for GradStore<D> {
    type Output = Tensor<D, Float>;
    fn index(&self, index: &Tensor<D, Float>) -> &Self::Output {
        self.get(index).unwrap()
    }
}
