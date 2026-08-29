use std::collections::HashSet;
use luma_tensor::{ops::{IndexingDTypeKind, ShapeDTypeKind}, Bool, DTypeKind, Device, IndexOp, Tensor};

/// X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
pub fn train_test_split<Dev, K1, K2>(
    x: &Tensor<Dev, K1>, y: &Tensor<Dev, K2>, test_ratio: f64,
) -> luma_tensor::Result<(Tensor<Dev, K1>, Tensor<Dev, K1>, Tensor<Dev, K2>, Tensor<Dev, K2>)> 
where
    Dev: Device,
    K1: DTypeKind<Dev> + IndexingDTypeKind<Dev> + ShapeDTypeKind<Dev>,
    K2: DTypeKind<Dev> + IndexingDTypeKind<Dev> + ShapeDTypeKind<Dev>,
{
    if test_ratio < 0.0 || test_ratio >= 1.0 {
        luma_tensor::bail!("invalid test_ratio {test_ratio}");
    }

    let n_samples = x.dims()[0];
    let n_samples_y = y.dims()[0];
    if n_samples != n_samples_y {
        luma_tensor::bail!("x n_samples != y n_samples");
    }

    let test_count = (n_samples as f64 * test_ratio) as usize;
    let mut test_indexs = HashSet::new();
    let mut test_mask = vec![false; n_samples];
    while test_indexs.len() < test_count {
        let index = rand::random_range(0..n_samples);
        if test_indexs.insert(index) {
            test_mask[index] = true;
        }
    }
    let test_mask = Tensor::<Dev, Bool>::new(test_mask, x.device())?;
    let train_mask = test_mask.not()?;

    let x_train = x.i(&train_mask)?;
    let y_train = y.i(&train_mask)?;
    
    let x_test = x.i(&test_mask)?;
    let y_test = y.i(&test_mask)?;
    
    Ok((x_train, x_test, y_train, y_test))
}