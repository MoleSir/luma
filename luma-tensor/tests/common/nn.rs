#![allow(dead_code)]

use super::*;
use luma_tensor::Device;

#[allow(dead_code)]
pub fn test_softmax_dim0(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], (2, 3), device);
    let out = t.softmax(1usize).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    let v = out.to_vec().unwrap();
    for row in v.chunks(3) {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row sum {} != 1", sum);
    }
}

#[allow(dead_code)]
pub fn test_softmax_dim1(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], (2, 3), device);
    let out = t.softmax(0usize).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    let v = out.to_vec().unwrap();
    assert!((v[0] + v[3] - 1.0).abs() < 1e-5, "col 0 sum {}", v[0] + v[3]);
    assert!((v[1] + v[4] - 1.0).abs() < 1e-5, "col 1 sum {}", v[1] + v[4]);
    assert!((v[2] + v[5] - 1.0).abs() < 1e-5, "col 2 sum {}", v[2] + v[5]);
}

#[allow(dead_code)]
pub fn test_softmax_numerical_stability(device: &impl Device) {
    let t = tensor_f32_dev(&[1000.0, 1000.0, 1000.0], (3,), device);
    let out = t.softmax(0usize).unwrap();
    let v = out.to_vec().unwrap();
    let expected = 1.0 / 3.0;
    for &x in &v {
        assert!((x - expected).abs() < 1e-4, "large values: {} vs {}", x, expected);
    }
}

#[allow(dead_code)]
pub fn test_softmax_grad(device: &impl Device) {
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0], (4,), device);
    t.set_requires_grad(true);
    let out = t.softmax(0usize).unwrap();
    let loss = out.sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let gv = grads.get(&t).unwrap().to_vec().unwrap();
    assert_close(&gv, &[0.0, 0.0, 0.0, 0.0], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_rms_norm_f32(device: &impl Device) {
    let x = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let w = tensor_f32_dev(&[1.0, 1.0, 1.0], (3,), device);
    let out = x.rms_norm(&w, 1e-5).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    let v = out.to_vec().unwrap();
    for row in v.chunks(3) {
        let mean_sq = row.iter().map(|&x| x * x).sum::<f64>() / 3.0;
        assert!((mean_sq - 1.0).abs() < 0.1, "row mean_sq {}", mean_sq);
    }
}

#[allow(dead_code)]
pub fn test_rms_norm_weighted(device: &impl Device) {
    let x = tensor_f32_dev(&[1.0, 2.0, 3.0], (3,), device);
    let w = tensor_f32_dev(&[2.0, 0.0, 1.0], (3,), device);
    let out = x.rms_norm(&w, 0.0).unwrap();
    let v = out.to_vec().unwrap();
    let inv_rms = 1.0_f64 / (14.0_f64 / 3.0_f64).sqrt();
    let e0 = 1.0 * inv_rms * 2.0;
    let e2 = 3.0 * inv_rms * 1.0;
    assert_close(&[v[0]], &[e0], 1e-4, 1e-4);
    assert!((v[1] - 0.0).abs() < 1e-5, "expected 0, got {}", v[1]);
    assert_close(&[v[2]], &[e2], 1e-4, 1e-4);
}

#[allow(dead_code)]
pub fn test_large_softmax(device: &impl Device) {
    let n = 5000usize;
    let data: Vec<f64> = (0..n).map(|i| (i as f64 % 100.0) - 50.0).collect();
    let t = tensor_f32_dev(&data, (n,), device);
    let out = t.softmax(0usize).unwrap();
    let v = out.to_vec().unwrap();
    let sum: f64 = v.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "softmax sum for {} elems: {}", n, sum);
    assert!(v.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

// ---- cross entropy (manual chain: softmax → ln → gather → neg → mean_all) ----

#[allow(dead_code)]
pub fn test_cross_entropy_chain_f32(device: &impl Device) {
    // (batch=4, classes=3) logits
    let logits = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (4, 3), device);
    // target class indices: shape (4, 1)
    let targets = tensor_i32_dev(&[0, 1, 2, 0], (4, 1), device);

    // step 1: softmax on last dim
    let softmax_out = logits.softmax(1usize).unwrap();
    assert_eq!(softmax_out.dims(), &[4, 3]);
    let sm = softmax_out.to_vec().unwrap();
    for row in sm.chunks(3) {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row sum {}", sum);
    }

    // step 2: ln (log)
    let log_probs = softmax_out.ln().unwrap();

    // step 3: gather (simulates NLL: select log-prob at target class)
    let selected = log_probs.gather(&targets, 1usize).unwrap();
    assert_eq!(selected.dims(), &[4, 1]);

    // step 4: neg + mean_all
    let loss = (-&selected).mean_all().unwrap();
    let v = loss.to_vec().unwrap();
    assert_eq!(v.len(), 1);

    // verify positivity: cross entropy should be > 0 for these non-trivial logits
    assert!(v[0] > 0.0, "cross entropy should be positive, got {}", v[0]);
}

#[allow(dead_code)]
pub fn test_cross_entropy_basic_f32(device: &impl Device) {
    // (batch=2, classes=3) — class 0 should be more confident than class 2
    let logits = tensor_f32_dev(&[2.0, 1.0, 0.1, 0.1, 1.0, 2.0], (2, 3), device);
    // target: class 0 for first sample, class 2 for second
    let targets = tensor_i32_dev(&[0, 2], (2, 1), device);

    let log_probs = logits.softmax(1usize).unwrap().ln().unwrap();
    let selected = log_probs.gather(&targets, 1usize).unwrap();
    let loss = (-&selected).mean_all().unwrap();

    let v = loss.to_vec().unwrap();
    assert!(v[0] > 0.0, "loss positive, got {}", v[0]);
    // rough sanity: loss should be relatively small since logits favor correct class
    assert!(v[0] < 1.5, "loss {} too high for easy case", v[0]);
}

#[allow(dead_code)]
pub fn test_cross_entropy_mnist_shape_f32(device: &impl Device) {
    // exact MNIST shape: batch=64, classes=10
    let n = 64 * 10;
    let logits_data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let logits = tensor_f32_dev(&logits_data, (64, 10), device);

    let target_data: Vec<i64> = (0..64).map(|i| (i % 10) as i64).collect();
    let targets = tensor_i32_dev(&target_data, (64, 1), device);

    let log_probs = logits.softmax(1usize).unwrap().ln().unwrap();
    let selected = log_probs.gather(&targets, 1usize).unwrap();
    assert_eq!(selected.dims(), &[64, 1]);
    let loss = (-&selected).mean_all().unwrap();
    assert!(loss.to_scalar().unwrap() > 0.0);
}

// ---- matmul with transposed weight (simulates Linear::forward) ----

#[allow(dead_code)]
pub fn test_matmul_transposed_weight_small_f32(device: &impl Device) {
    // small: (batch=2, 3) @ (4, 3)^T = (2, 3) @ (3, 4)
    let input = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device);
    let weight = tensor_f32_dev(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], (4, 3), device);
    let weight_t = weight.transpose_last().unwrap();
    assert_eq!(weight_t.dims(), &[3, 4]);
    let output = input.matmul(&weight_t).unwrap();
    assert_eq!(output.dims(), &[2, 4]);
    let v = output.to_vec().unwrap();
    assert!(v.iter().all(|&x| x.is_finite()));
}

#[allow(dead_code)]
pub fn test_matmul_transposed_weight_f32(device: &impl Device) {
    // simulate: F::linear does input.matmul(&weight.transpose_last()?)
    // for fc1: (batch=64, 784) @ (512, 784)^T = (64, 784) @ (784, 512)
    let in_features = 784usize;
    let out_features = 512usize;
    let batch = 64usize;

    let input_data: Vec<f64> = (0..(batch * in_features)).map(|i| (i % 128) as f64).collect();
    let input = tensor_f32_dev(&input_data, (batch, in_features), device);

    // weight: (out_features=512, in_features=784), stored contiguously
    let weight_data: Vec<f64> = (0..(out_features * in_features)).map(|i| (i % 256) as f64 * 0.01).collect();
    let weight = tensor_f32_dev(&weight_data, (out_features, in_features), device);

    // transpose_last: swap last two dims → (in_features, out_features) with strides (1, in_features)
    let weight_t = weight.transpose_last().unwrap();
    assert_eq!(weight_t.dims(), &[in_features, out_features]);

    let output = input.matmul(&weight_t).unwrap();
    assert_eq!(output.dims(), &[batch, out_features]);

    let v = output.to_vec().unwrap();
    assert_eq!(v.len(), batch * out_features);
    // sanity: all values finite
    assert!(v.iter().all(|&x| x.is_finite()), "output contains NaN or Inf");
}

#[allow(dead_code)]
pub fn test_matmul_add_bias_f32(device: &impl Device) {
    // Simulates Linear::forward: matmul then add bias with broadcasting
    let batch = 4usize;
    let in_features = 3usize;
    let out_features = 5usize;

    let input_data: Vec<f64> = (1..=(batch * in_features)).map(|i| i as f64).collect();
    let input = tensor_f32_dev(&input_data, (batch, in_features), device);

    let weight_data: Vec<f64> = (0..(out_features * in_features)).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let weight = tensor_f32_dev(&weight_data, (out_features, in_features), device);
    let weight_t = weight.transpose_last().unwrap();

    let output = input.matmul(&weight_t).unwrap();
    assert_eq!(output.dims(), &[batch, out_features]);

    let bias = tensor_f32_dev(&[0.1, 0.2, 0.3, 0.4, 0.5], (out_features,), device);
    // Use broadcast_add (same as fixed Linear::forward)
    let output = output.broadcast_add(&bias).unwrap();
    assert_eq!(output.dims(), &[batch, out_features]);
    let v = output.to_vec().unwrap();
    assert!(v.iter().all(|&x| x.is_finite()), "output has NaN/Inf");

    // Verify: each row should have bias added
    let matmul_only = input.matmul(&weight_t).unwrap();
    let mv = matmul_only.to_vec().unwrap();
    let bv = bias.to_vec().unwrap();
    for r in 0..batch {
        for c in 0..out_features {
            let idx = r * out_features + c;
            assert!(
                (v[idx] - (mv[idx] + bv[c])).abs() < 1e-4,
                "row {} col {}: expected {} + {} = {}, got {}",
                r,
                c,
                mv[idx],
                bv[c],
                mv[idx] + bv[c],
                v[idx]
            );
        }
    }
}

// ---- broadcast_add correctness (bias broadcasting pattern in Linear) ----

#[allow(dead_code)]
pub fn test_broadcast_add_precision(device: &impl Device) {
    // Recreate the exact Linear::forward pattern: (batch, out) + (out,)
    let batch = 4usize;
    let out = 5usize;

    let a_data: Vec<f64> = vec![0.5, 1.2, 3.4, 5.6, 7.8, 1.1, 2.3, 4.5, 6.7, 8.9, 2.2, 3.4, 5.6, 7.8, 9.0, 3.3, 4.5, 6.7, 8.9, 0.1];
    let a = tensor_f32_dev(&a_data, (batch, out), device);

    let b_data: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let b = tensor_f32_dev(&b_data, (out,), device);

    let result = a.broadcast_add(&b).unwrap();
    let v = result.to_vec().unwrap();

    // Expected: a[i][j] + b[j]
    for i in 0..batch {
        for j in 0..out {
            let expected = a_data[i * out + j] + b_data[j];
            let actual = v[i * out + j];
            assert!((actual - expected).abs() < 1e-5, "broadcast_add mismatch at ({},{}): expected {}, got {}", i, j, expected, actual);
        }
    }
}

// ---- cross-entropy chain precision check ----

#[allow(dead_code)]
pub fn test_cross_entropy_precision(device: &impl Device) {
    // Fixed logits and targets — compare against manually computed f64 reference
    let logits_data = vec![2.0_f64, 1.0, 0.1, 0.1, 2.0, 1.0, 1.0, 0.1, 2.0];
    let logits = tensor_f32_dev(&logits_data, (3, 3), device);
    let targets = tensor_i32_dev(&[0_i64, 1, 2], (3, 1), device);

    // Compute softmax in f64 for reference
    let f64_logits: Vec<Vec<f64>> = logits_data.chunks(3).map(|r| r.to_vec()).collect();
    let target_classes = [0usize, 1usize, 2usize];
    let mut ref_values = vec![0.0_f64; 3];
    for (row_idx, row) in f64_logits.iter().enumerate() {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
        let log_prob = ((row[target_classes[row_idx]] - max_val).exp() / exp_sum).ln();
        ref_values[row_idx] = -log_prob;
    }
    let ref_loss = ref_values.iter().sum::<f64>() / 3.0;

    let log_probs = logits.softmax(1usize).unwrap().ln().unwrap();
    let selected = log_probs.gather(&targets, 1usize).unwrap();
    let loss = (-&selected).mean_all().unwrap();
    let actual = loss.to_scalar().unwrap();

    assert!(
        (actual - ref_loss).abs() < 1e-4,
        "cross entropy mismatch: expected {}, got {} (diff: {:.2e})",
        ref_loss,
        actual,
        (actual - ref_loss).abs()
    );
}

// ---- gradient through broadcast_add (Linear bias pattern) ----

#[allow(dead_code)]
pub fn test_broadcast_add_grad_f32(device: &impl Device) {
    let batch = 4usize;
    let in_features = 3usize;
    let out_features = 2usize;

    let input = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (batch, in_features), device);

    let weight = tensor_f32_dev(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (out_features, in_features), device);
    weight.set_requires_grad(true);

    let bias = tensor_f32_dev(&[0.01, 0.02], (out_features,), device);
    bias.set_requires_grad(true);

    let weight_t = weight.transpose_last().unwrap();
    let matmul_out = input.matmul(&weight_t).unwrap();
    let output = matmul_out.broadcast_add(&bias).unwrap();

    assert_eq!(output.dims(), &[batch, out_features]);

    let loss = output.sum_all().unwrap();
    let grads = loss.backward().unwrap();

    let wg = grads.get(&weight).unwrap();
    assert_eq!(wg.dims(), &[out_features, in_features]);
    let wgv = wg.to_vec().unwrap();
    assert!(wgv.iter().all(|&x| x.is_finite()), "weight grad has NaN/Inf");

    let bg = grads.get(&bias).unwrap();
    assert_eq!(bg.dims(), &[out_features]);
    let bgv = bg.to_vec().unwrap();
    assert!(bgv.iter().all(|&x| x.is_finite()), "bias grad has NaN/Inf");
    // For loss=sum(output), bias gradient = batch for each output class
    for &bg_val in &bgv {
        assert!((bg_val - batch as f64).abs() < 1e-3, "bias grad expected {}, got {}", batch, bg_val);
    }

    // Simulate one SGD step: var -= lr * grad
    let lr = 0.01;
    let bias_before = bias.to_vec().unwrap();
    let scaled = bg.mul_scalar(lr).unwrap();
    bias.sub_(&scaled).unwrap();
    let bias_after = bias.to_vec().unwrap();
    for (i, (&before, &after)) in bias_before.iter().zip(bias_after.iter()).enumerate() {
        let expected = before - lr * bgv[i];
        assert!(
            (after - expected).abs() < 1e-5,
            "SGD step mismatch at {}: {} - {}*{} = {}, got {}",
            i,
            before,
            lr,
            bgv[i],
            expected,
            after
        );
    }
}

// ---- sum_keepdim + squeeze (used in broadcast backward) ----

#[allow(dead_code)]
pub fn test_broadcast_reduce_backward_f32(device: &impl Device) {
    let batch = 4usize;
    let out = 3usize;

    let grad = tensor_f32_dev(&[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3], (batch, out), device);

    let summed = grad.sum_keepdim(0usize).unwrap();
    assert_eq!(summed.dims(), &[1, out]);

    let squeezed = summed.squeeze(0usize).unwrap();
    assert_eq!(squeezed.dims(), &[out]);

    let v = squeezed.to_vec().unwrap();
    for i in 0..out {
        assert!(
            (v[i] - batch as f64 * 0.1 * (i + 1) as f64).abs() < 1e-5,
            "reduce mismatch at {}: expected {}, got {}",
            i,
            batch as f64 * 0.1 * (i + 1) as f64,
            v[i]
        );
    }
}

// ---- argmax + eq + true_count (evaluation pipeline) ----

#[allow(dead_code)]
pub fn test_argmax_eval_pipeline_f32(device: &impl Device) {
    // Simulates the test evaluation: argmax_keepdim(1) → eq → true_count
    let batch = 4usize;
    let classes = 3usize;

    // Model output: row 0→class 0, row 1→class 1, row 2→class 2, row 3→class 0
    let logits = tensor_f32_dev(
        &[
            3.0, 1.0, 2.0, // max at col 0
            1.0, 5.0, 2.0, // max at col 1
            2.0, 1.0, 3.0, // max at col 2
            4.0, 2.0, 1.0, // max at col 0
        ],
        (batch, classes),
        device,
    );

    let predictions = logits.argmax_keepdim(1usize).unwrap();
    assert_eq!(predictions.dims(), &[batch, 1]);
    let pred_vals = predictions.to_vec().unwrap();
    // argmax should give u32 indices
    assert_eq!(pred_vals, vec![0i64, 1, 2, 0]);

    // Target labels (u32 to match argmax output type)
    let target = tensor_u32_dev(&[0i64, 1, 2, 0], (batch, 1), device);
    let correct = predictions.eq(&target).unwrap();
    let count = correct.true_count().unwrap();
    assert_eq!(count, 4, "all 4 predictions should be correct, got {}", count);

    // Test with partially wrong predictions
    let target2 = tensor_u32_dev(&[0i64, 0, 2, 1], (batch, 1), device);
    let correct2 = predictions.eq(&target2).unwrap();
    let count2 = correct2.true_count().unwrap();
    assert_eq!(count2, 2, "2 predictions should be correct, got {}", count2);
}

// ---- mini training loop (forward + backward + SGD step + re-forward) ----

#[allow(dead_code)]
pub fn test_mini_training_step_f32(device: &impl Device) {
    // Simulate one SGD step: loss should decrease after parameter update
    let batch = 8usize;
    let in_features = 3usize;
    let out_features = 2usize;

    let input = tensor_f32_dev(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        (batch, in_features),
        device,
    );

    let weight = tensor_f32_dev(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (out_features, in_features), device);
    weight.set_requires_grad(true);

    let bias = tensor_f32_dev(&[0.01, 0.02], (out_features,), device);
    bias.set_requires_grad(true);

    // Forward 1
    let w_t = weight.transpose_last().unwrap();
    let m = input.matmul(&w_t).unwrap();
    let out1 = m.broadcast_add(&bias).unwrap();
    let loss1 = out1.sum_all().unwrap().to_scalar().unwrap();

    // Backward
    let grads = out1.sum_all().unwrap().backward().unwrap();
    let lr = 0.1;

    // SGD step
    if let Some(wg) = grads.get(&weight) {
        let scaled = wg.mul_scalar(lr).unwrap();
        weight.sub_(&scaled).unwrap();
    }
    if let Some(bg) = grads.get(&bias) {
        let scaled = bg.mul_scalar(lr).unwrap();
        bias.sub_(&scaled).unwrap();
    }

    // Forward 2 with same input
    let w_t2 = weight.transpose_last().unwrap();
    let m2 = input.matmul(&w_t2).unwrap();
    let out2 = m2.broadcast_add(&bias).unwrap();
    let loss2 = out2.sum_all().unwrap().to_scalar().unwrap();

    // Loss should decrease: gradient descent moves params in -grad direction
    // which should reduce sum(output) since grad of sum is all ones
    assert!(loss2 < loss1, "SGD step should reduce loss: before={}, after={}", loss1, loss2);
}

#[allow(dead_code)]
pub fn test_sum_keepdim_nonuniform_f32(device: &impl Device) {
    // Non-uniform data — different values per row
    // sum over dim=0 should produce row-wise sum
    let t = tensor_f32_dev(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (3, 3), device);

    let summed = t.sum_keepdim(0usize).unwrap();
    assert_eq!(summed.dims(), &[1, 3]);
    let v = summed.to_vec().unwrap();
    assert_close(&v, &[12.0, 15.0, 18.0], 1e-5, 1e-5);

    let summed1 = t.sum_keepdim(1usize).unwrap();
    assert_eq!(summed1.dims(), &[3, 1]);
    let v1 = summed1.to_vec().unwrap();
    assert_close(&v1, &[6.0, 15.0, 24.0], 1e-5, 1e-5);
}

// ---- argmax + eq + true_count at MNIST test scale (batch=1000, classes=10) ----

#[allow(dead_code)]
pub fn test_argmax_eval_large_f32(device: &impl Device) {
    let batch = 1000usize;
    let classes = 10usize;

    // Generate logits: row i gets max at position i%10
    let mut logits_data = vec![0.0f64; batch * classes];
    let mut target_data = vec![0i64; batch];
    for i in 0..batch {
        let correct_class = i % classes;
        logits_data[i * classes + correct_class] = 100.0;
        // Make every 100th row wrong (rows 0, 100, 200, ...)
        target_data[i] = if i % 100 == 0 { ((correct_class + 1) % classes) as i64 } else { correct_class as i64 };
    }

    let logits = tensor_f32_dev(&logits_data, (batch, classes), device);
    let target = tensor_u32_dev(&target_data, (batch, 1), device);

    // Step 1: argmax_keepdim
    let preds = logits.argmax_keepdim(1usize).unwrap();
    assert_eq!(preds.dims(), &[batch, 1]);
    let pred_vals = preds.to_vec().unwrap();
    for i in 0..batch {
        assert_eq!(pred_vals[i], (i % classes) as i64, "argmax wrong at row {}: expected {}, got {}", i, i % classes, pred_vals[i]);
    }

    // Step 2: eq (compare predictions against target)
    let correct_mask = preds.eq(&target).unwrap();
    assert_eq!(correct_mask.dims(), &[batch, 1]);

    // Step 3: true_count — expect 990 correct (10 wrong rows at 0, 100, ..., 900)
    let count = correct_mask.true_count().unwrap();
    let expected = batch - batch / 100; // 1000 - 10 = 990
    assert_eq!(count, expected, "true_count mismatch: expected {}, got {}", expected, count);
}
