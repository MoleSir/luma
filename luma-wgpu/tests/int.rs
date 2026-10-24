use luma_tensor::{Int, IntDType, Tensor};
use luma_wgpu::Wgpu;

use std::sync::LazyLock;

static DEVICE: LazyLock<Option<Wgpu>> = LazyLock::new(|| match Wgpu::new(0) {
    Ok(device) => Some(device),
    Err(e) => {
        eprintln!("skip wgpu test: {e}");
        None
    }
});

fn device() -> Option<Wgpu> {
    DEVICE.clone()
}

fn i32_tensor(dev: &Wgpu, data: &[i64], shape: impl Into<luma_tensor::Shape>) -> Tensor<Wgpu, Int> {
    Tensor::<Wgpu, Int>::from_slice(data, shape, (dev, IntDType::I32)).unwrap()
}

fn u32_tensor(dev: &Wgpu, data: &[i64], shape: impl Into<luma_tensor::Shape>) -> Tensor<Wgpu, Int> {
    Tensor::<Wgpu, Int>::from_slice(data, shape, (dev, IntDType::U32)).unwrap()
}

#[test]
fn binary_i32() {
    let Some(dev) = device() else { return };

    let a = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3));
    let b = i32_tensor(&dev, &[10, 20, 30, 40, 50, 60], (2, 3));

    assert_eq!(a.add(&b).unwrap().to_vec().unwrap(), vec![11, 22, 33, 44, 55, 66]);
    assert_eq!(a.sub(&b).unwrap().to_vec().unwrap(), vec![-9, -18, -27, -36, -45, -54]);
    assert_eq!(a.mul(&b).unwrap().to_vec().unwrap(), vec![10, 40, 90, 160, 250, 360]);
    assert_eq!(a.maximum(&b).unwrap().to_vec().unwrap(), vec![10, 20, 30, 40, 50, 60]);
    assert_eq!(a.minimum(&b).unwrap().to_vec().unwrap(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn binary_u32() {
    let Some(dev) = device() else { return };

    let a = u32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3));
    let b = u32_tensor(&dev, &[10, 20, 30, 40, 50, 60], (2, 3));

    assert_eq!(a.add(&b).unwrap().to_vec().unwrap(), vec![11, 22, 33, 44, 55, 66]);
    assert_eq!(a.mul(&b).unwrap().to_vec().unwrap(), vec![10, 40, 90, 160, 250, 360]);
    assert_eq!(a.maximum(&b).unwrap().to_vec().unwrap(), vec![10, 20, 30, 40, 50, 60]);
    assert_eq!(a.minimum(&b).unwrap().to_vec().unwrap(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn binary_strided_i32() {
    let Some(dev) = device() else { return };

    let a = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3)).transpose(0, 1).unwrap();
    let b = i32_tensor(&dev, &[10, 20, 30, 40, 50, 60], (3, 2));

    assert_eq!(a.add(&b).unwrap().to_vec().unwrap(), vec![11, 24, 32, 45, 53, 66]);
}

#[test]
fn binary_scalar_i32() {
    let Some(dev) = device() else { return };

    let a = i32_tensor(&dev, &[1, 2, 3, 4], (2, 2));
    assert_eq!(a.add_scalar(10).unwrap().to_vec().unwrap(), vec![11, 12, 13, 14]);
    assert_eq!(a.sub_scalar(-1).unwrap().to_vec().unwrap(), vec![2, 3, 4, 5]);
    assert_eq!(a.mul_scalar(3).unwrap().to_vec().unwrap(), vec![3, 6, 9, 12]);
    assert_eq!(a.div_scalar(2).unwrap().to_vec().unwrap(), vec![0, 1, 1, 2]);
    assert_eq!(a.sub_scalar_lhs(10).unwrap().to_vec().unwrap(), vec![9, 8, 7, 6]);
}

#[test]
fn binary_inplace_i32() {
    let Some(dev) = device() else { return };

    let dst = i32_tensor(&dev, &[1, 2, 3, 4], (2, 2));
    let src = i32_tensor(&dev, &[10, 20, 30, 40], (2, 2));
    dst.add_(&src).unwrap();
    assert_eq!(dst.to_vec().unwrap(), vec![11, 22, 33, 44]);

    dst.mul_scalar_(2).unwrap();
    assert_eq!(dst.to_vec().unwrap(), vec![22, 44, 66, 88]);
}

#[test]
fn binary_inplace_strided_i32() {
    let Some(dev) = device() else { return };

    let base = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3));
    let view = base.transpose(0, 1).unwrap();
    let src = i32_tensor(&dev, &[10, 20, 30, 40, 50, 60], (3, 2));

    view.add_(&src).unwrap();
    assert_eq!(base.to_vec().unwrap(), vec![11, 32, 53, 24, 45, 66]);
}

#[test]
fn unary_i32() {
    let Some(dev) = device() else { return };

    let a = i32_tensor(&dev, &[-2, -1, 0, 1, 2], (5,));
    assert_eq!(a.neg().unwrap().to_vec().unwrap(), vec![2, 1, 0, -1, -2]);
    assert_eq!(a.abs().unwrap().to_vec().unwrap(), vec![2, 1, 0, 1, 2]);
    assert_eq!(a.sign().unwrap().to_vec().unwrap(), vec![-1, -1, 0, 1, 1]);
    assert_eq!(a.affine(2, 3).unwrap().to_vec().unwrap(), vec![-1, 1, 3, 5, 7]);
    assert_eq!(a.pow(2).unwrap().to_vec().unwrap(), vec![4, 1, 0, 1, 4]);
    assert_eq!(a.clamp(Some(-1), Some(1)).unwrap().to_vec().unwrap(), vec![-1, -1, 0, 1, 1]);
}

#[test]
fn unary_u32() {
    let Some(dev) = device() else { return };

    let a = u32_tensor(&dev, &[0, 1, 2, 3], (4,));
    assert_eq!(a.sign().unwrap().to_vec().unwrap(), vec![0, 1, 1, 1]);
    assert_eq!(a.abs().unwrap().to_vec().unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(a.affine(2, 1).unwrap().to_vec().unwrap(), vec![1, 3, 5, 7]);
}

#[test]
fn cmp_i32() {
    let Some(dev) = device() else { return };

    let a = i32_tensor(&dev, &[1, 5, 3, 2], (4,));
    let b = i32_tensor(&dev, &[2, 3, 3, 1], (4,));
    assert_eq!(a.eq(&b).unwrap().to_vec().unwrap(), vec![false, false, true, false]);
    assert_eq!(a.lt(&b).unwrap().to_vec().unwrap(), vec![true, false, false, false]);
    assert_eq!(a.gt(&b).unwrap().to_vec().unwrap(), vec![false, true, false, true]);
    assert_eq!(a.le(&b).unwrap().to_vec().unwrap(), vec![true, false, true, false]);
    assert_eq!(a.ge(&b).unwrap().to_vec().unwrap(), vec![false, true, true, true]);
    assert_eq!(a.ne(&b).unwrap().to_vec().unwrap(), vec![true, true, false, true]);
    assert_eq!(a.lt_scalar(3).unwrap().to_vec().unwrap(), vec![true, false, false, true]);
}

#[test]
fn cmp_u32() {
    let Some(dev) = device() else { return };

    let a = u32_tensor(&dev, &[1, 5, 3, 2], (4,));
    let b = u32_tensor(&dev, &[2, 3, 3, 1], (4,));
    assert_eq!(a.lt(&b).unwrap().to_vec().unwrap(), vec![true, false, false, false]);
    assert_eq!(a.ge_scalar(3).unwrap().to_vec().unwrap(), vec![false, true, true, false]);
}

#[test]
fn unary_strided_i32() {
    let Some(dev) = device() else { return };

    let x = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3)).transpose(0, 1).unwrap();
    assert_eq!(x.neg().unwrap().to_vec().unwrap(), vec![-1, -4, -2, -5, -3, -6]);
}

#[test]
fn cmp_strided_i32() {
    let Some(dev) = device() else { return };

    let a = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3)).transpose(0, 1).unwrap();
    let b = i32_tensor(&dev, &[2, 3, 4, 5, 6, 7], (3, 2));
    assert_eq!(a.lt(&b).unwrap().to_vec().unwrap(), vec![true, false, true, false, true, true]);
}

#[test]
fn matmul_i32() {
    let Some(dev) = device() else { return };

    let a = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3));
    let b = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (3, 2));
    assert_eq!(a.matmul(&b).unwrap().to_vec().unwrap(), vec![22, 28, 49, 64]);
}

#[test]
fn matmul_u32() {
    let Some(dev) = device() else { return };

    let a = u32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3));
    let b = u32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (3, 2));
    assert_eq!(a.matmul(&b).unwrap().to_vec().unwrap(), vec![22, 28, 49, 64]);
}

#[test]
fn matmul_transposed_weight_i32() {
    let Some(dev) = device() else { return };

    let a = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3));
    let w = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (4, 3));
    let wt = w.transpose(0, 1).unwrap();
    assert_eq!(a.matmul(&wt).unwrap().to_vec().unwrap(), vec![14, 32, 50, 68, 32, 77, 122, 167]);
}

#[test]
fn reduce_i32() {
    let Some(dev) = device() else { return };

    let x = i32_tensor(&dev, &[1, 5, 3, 2, 4, 6], (2, 3));
    assert_eq!(x.sum(1usize).unwrap().to_vec().unwrap(), vec![9, 12]);
    assert_eq!(x.sum_all().unwrap().to_vec().unwrap(), vec![21]);
    assert_eq!(x.max(1usize).unwrap().to_vec().unwrap(), vec![5, 6]);
    assert_eq!(x.min(1usize).unwrap().to_vec().unwrap(), vec![1, 2]);
    assert_eq!(x.prod(1usize).unwrap().to_vec().unwrap(), vec![15, 48]);

    let t = x.transpose(0, 1).unwrap();
    assert_eq!(t.sum(0usize).unwrap().to_vec().unwrap(), vec![9, 12]);

    let keep = x.sum_keepdim(0usize).unwrap();
    assert_eq!(keep.dims(), &[1, 3]);
    assert_eq!(keep.to_vec().unwrap(), vec![3, 9, 9]);
}

#[test]
fn reduce_u32() {
    let Some(dev) = device() else { return };

    let x = u32_tensor(&dev, &[100, 200, 300, 400], (2, 2));
    assert_eq!(x.sum(0usize).unwrap().to_vec().unwrap(), vec![400, 600]);
    assert_eq!(x.sum_all().unwrap().to_vec().unwrap(), vec![1000]);
    assert_eq!(x.max_all().unwrap().to_vec().unwrap(), vec![400]);
    assert_eq!(x.min_all().unwrap().to_vec().unwrap(), vec![100]);
    assert_eq!(x.prod_all().unwrap().to_vec().unwrap(), vec![2_400_000_000]);

    let y = u32_tensor(&dev, &[100, 200, 300, 400], (2, 2));
    let z = u32_tensor(&dev, &[100, 200, 300, 401], (2, 2));
    assert!(x.allclose(&y).unwrap());
    assert!(!x.allclose(&z).unwrap());
}

#[test]
fn arg_reduce_i32() {
    let Some(dev) = device() else { return };

    let x = i32_tensor(&dev, &[1, 5, 3, 2, 4, 6], (2, 3));
    assert_eq!(x.argmax(1usize).unwrap().to_vec().unwrap(), vec![1, 2]);
    assert_eq!(x.argmin(1usize).unwrap().to_vec().unwrap(), vec![0, 0]);
    assert_eq!(x.argmax(0usize).unwrap().to_vec().unwrap(), vec![1, 0, 1]);
    assert_eq!(x.argmin(0usize).unwrap().to_vec().unwrap(), vec![0, 1, 0]);

    let keep = x.argmax_keepdim(1usize).unwrap();
    assert_eq!(keep.dims(), &[2, 1]);
    assert_eq!(keep.to_vec().unwrap(), vec![1, 2]);

    let ties = i32_tensor(&dev, &[3, 1, 3, 2, 3, 0], (2, 3));
    assert_eq!(ties.argmax(1usize).unwrap().to_vec().unwrap(), vec![0, 1]);
    assert_eq!(ties.argmin(1usize).unwrap().to_vec().unwrap(), vec![1, 2]);

    let t = x.transpose(0, 1).unwrap();
    assert_eq!(t.argmax(0usize).unwrap().to_vec().unwrap(), vec![1, 2]);

    let data: Vec<i64> = (0..1000).collect();
    let big = i32_tensor(&dev, &data, (1000,));
    assert_eq!(big.argmax(0usize).unwrap().to_vec().unwrap(), vec![999]);
    assert_eq!(big.argmin(0usize).unwrap().to_vec().unwrap(), vec![0]);
}

#[test]
fn arg_reduce_u32() {
    let Some(dev) = device() else { return };

    let x = u32_tensor(&dev, &[1, 5, 3, 2, 4, 6], (2, 3));
    assert_eq!(x.argmax(1usize).unwrap().to_vec().unwrap(), vec![1, 2]);
    assert_eq!(x.argmin(1usize).unwrap().to_vec().unwrap(), vec![0, 0]);

    let ties = u32_tensor(&dev, &[3, 1, 3, 2, 3, 0], (2, 3));
    assert_eq!(ties.argmax(1usize).unwrap().to_vec().unwrap(), vec![0, 1]);
    assert_eq!(ties.argmin(1usize).unwrap().to_vec().unwrap(), vec![1, 2]);
}

#[test]
fn unary_inplace_i32() {
    let Some(dev) = device() else { return };

    let a = i32_tensor(&dev, &[-2, -1, 0, 1, 2], (5,));
    a.neg_().unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![2, 1, 0, -1, -2]);

    a.abs_().unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![2, 1, 0, 1, 2]);

    a.affine_(2, 3).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![7, 5, 3, 5, 7]);

    a.pow_(2).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![49, 25, 9, 25, 49]);

    a.clamp_(Some(10), Some(30)).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![30, 25, 10, 25, 30]);

    let base = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3));
    let view = base.transpose(0, 1).unwrap();
    view.neg_().unwrap();
    assert_eq!(base.to_vec().unwrap(), vec![-1, -2, -3, -4, -5, -6]);
}

#[test]
fn unary_inplace_u32() {
    let Some(dev) = device() else { return };

    let a = u32_tensor(&dev, &[0, 1, 2, 3], (4,));
    a.sign_().unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![0, 1, 1, 1]);

    a.affine_(2, 1).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![1, 3, 3, 3]);
}

#[test]
fn arange_i32() {
    let Some(dev) = device() else { return };

    let a = Tensor::<Wgpu, Int>::arange(0, 5, 1, (&dev, IntDType::I32)).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![0, 1, 2, 3, 4]);

    let b = Tensor::<Wgpu, Int>::arange(5, 0, -2, (&dev, IntDType::I32)).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![5, 3, 1]);

    let empty = Tensor::<Wgpu, Int>::arange(0, 0, 1, (&dev, IntDType::I32)).unwrap();
    assert_eq!(empty.element_count(), 0);
}

#[test]
fn arange_u32() {
    let Some(dev) = device() else { return };

    let a = Tensor::<Wgpu, Int>::arange(0, 10, 3, (&dev, IntDType::U32)).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![0, 3, 6, 9]);
}

#[test]
fn arange_invalid() {
    let Some(dev) = device() else { return };

    assert!(Tensor::<Wgpu, Int>::arange(0, 5, 0, (&dev, IntDType::I32)).is_err());
    assert!(Tensor::<Wgpu, Int>::arange(0, 5, 1, (&dev, IntDType::U8)).is_err());
}

#[test]
fn contiguous_i32() {
    let Some(dev) = device() else { return };

    let x = i32_tensor(&dev, &[1, 2, 3, 4, 5, 6], (2, 3)).transpose(0, 1).unwrap();
    let c = x.contiguous().unwrap();
    assert!(c.is_contiguous());
    assert_eq!(c.to_vec().unwrap(), vec![1, 4, 2, 5, 3, 6]);
}
