//! Device-generic view (shape) executor tests.

use super::*;
use luma_tensor::Device;

pub fn test_reshape<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), |a| a.reshape((3, 2)), |a| a.reshape((3, 2)));
}
pub fn test_transpose<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), |a| a.transpose(0usize, 1usize), |a| a.transpose(0usize, 1usize));
}
pub fn test_permute<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 2, 2), |a| a.permute([1usize, 0, 2]), |a| a.permute([1usize, 0, 2]));
}
pub fn test_narrow<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), |a| a.narrow(0usize, 1, 1), |a| a.narrow(0usize, 1, 1));
}
pub fn test_slice<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 4), |a| a.slice(1usize, 0, 4, 2), |a| a.slice(1usize, 0, 4, 2));
}
pub fn test_squeeze<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0], (1, 3), |a| a.squeeze(0usize), |a| a.squeeze(0usize));
}
pub fn test_unsqueeze<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0], (3,), |a| a.unsqueeze(0usize), |a| a.unsqueeze(0usize));
}
/// Squeezing/unsqueezing when another size-1 dim remains. The IR records the
/// exact dim, so execution must not derive it from the shapes — (1, 1, 2)
/// squeezing dim 0 or dim 1 both yield (1, 2), which is ambiguous.
pub fn test_squeeze_ambiguous_dim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0], (1, 1, 2), |a| a.squeeze(0usize), |a| a.squeeze(0usize));
}
pub fn test_unsqueeze_ambiguous_dim<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0], (1, 2), |a| a.unsqueeze(0usize), |a| a.unsqueeze(0usize));
}
pub fn test_broadcast_as<D: Device>(dev: &D) {
    unary_check(dev, &[1.0, 2.0, 3.0], (3,), |a| a.broadcast_as((2, 3)), |a| a.broadcast_as((2, 3)));
}
