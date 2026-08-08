# Candle CUDA Backend - MatMul 实现参考

> 基于 `src/cuda_backend/` 目录源码分析，导出日期：2026-08-04

---

## 目录

1. [架构概览](#1-架构概览)
2. [核心函数：`gemm_config`](#2-核心函数gemm_config)
3. [MatMul 主入口：`matmul`](#3-matmul-主入口matmul)
4. [分数据类型的 GEMM 实现](#4-分数据类型的-gemm-实现)
5. [精度控制](#5-精度控制)
6. [MatMul 在卷积中的应用](#6-matmul-在卷积中的应用)
7. [辅助工具 Traits](#7-辅助工具-traits)
8. [数据流总结](#8-数据流总结)

---

## 1. 架构概览

Candle 的 CUDA matmul 完全基于 **cuBLAS** 库实现，使用 `cudarc` crate 作为 FFI 绑定层。不包含手写的 CUDA kernel 用于矩阵乘法。

### 关键依赖

```rust
// mod.rs
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};
use half::{bf16, f16};
```

### 支持的 dtype

| DType | cuBLAS 数据类型 | 说明 |
|-------|----------------|------|
| F16   | `CUDA_R_16F`  | 半精度浮点 |
| BF16  | `CUDA_R_16BF` | Brain floating point |
| F32   | `CUDA_R_32F`  | 单精度浮点，支持 TF32 加速 |
| F64   | `CUDA_R_64F`  | 双精度浮点 |

不支持 U8/U32/I64/F8E4M3 类型的 matmul。

### 文件结构

| 文件 | 内容 |
|------|------|
| `mod.rs` | matmul 核心逻辑、gemm_config、各 dtype 的 gemm 函数、BackendStorage trait 实现 |
| `device.rs` | `CudaDevice` 结构体，持有 `CudaBlas` handle |
| `utils.rs` | `Map1`/`Map2`/`Map2InPlace` 等辅助 trait |
| `error.rs` | `CudaError` 错误类型定义 |
| `cudnn.rs` | cuDNN 卷积实现（卷积中间接使用 matmul） |

---

## 2. 核心函数：`gemm_config`

**位置**: `mod.rs:1200-1290`

```rust
fn gemm_config<T>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<StridedBatchedConfig<T>>
```

### 功能

将 candle 的 `Layout`（shape + stride）转换为 cuBLAS 所需的 `StridedBatchedConfig`，处理了转置和广播语义。

### 参数说明

| 参数 | 含义 |
|------|------|
| `b` | batch size |
| `m` | 输出行数（对应 LHS 的行） |
| `n` | 输出列数（对应 RHS 的列） |
| `k` | 内积维度 |
| `alpha`, `beta` | GEMM 的缩放系数: `C = alpha * A * B + beta * C` |
| `lhs_l` | 左矩阵的 Layout |
| `rhs_l` | 右矩阵的 Layout |

### 核心逻辑

**矩阵维度约定**（遵循 cuBLAS 文档）：

- **A 矩阵 (RHS)**: dims = `(batching, k, n)`
- **B 矩阵 (LHS)**: dims = `(batching, m, k)`
- **C 矩阵**: dims = `(batching, m, n)`

> 注意：candle 中 matmul 调用时传入 `(b, m, n, k)`，其中 RHS (A) 形状为 `(..., k, n)`，LHS (B) 形状为 `(..., m, k)`。输入时 rhs 在前、lhs 在后，这是因为 `C = rhs * lhs` 的约定。

### 转置检测逻辑

#### RHS (A 矩阵) 的转置判断

```rust
// mod.rs:1219-1229
let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
    (n as i32, cublasOperation_t::CUBLAS_OP_N)  // Normal: 最后两个stride是 (1, n) → 列主序
} else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
    (k as i32, cublasOperation_t::CUBLAS_OP_T)  // Transpose: 最后两个stride是 (k, 1)
} else {
    Err(CudaError::MatMulNonContiguous { ... })?
};
```

#### LHS (B 矩阵) 的转置判断

```rust
// mod.rs:1233-1243
let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
    (k as i32, cublasOperation_t::CUBLAS_OP_N)  // Normal: 最后两个stride是 (1, k)
} else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
    (m as i32, cublasOperation_t::CUBLAS_OP_T)  // Transpose: 最后两个stride是 (m, 1)
} else {
    Err(CudaError::MatMulNonContiguous { ... })?
};
```

**规则总结**：检查两个矩阵最后两个维度的 stride：
- LHS (B) 期望形状 `(..., m, k)`，对应 stride `(*, stride_m2, stride_m1)`
- RHS (A) 期望形状 `(..., k, n)`，对应 stride `(*, stride_m2, stride_m1)`

如果 stride 不符合这四种合法的连续/转置排列之一，则返回 `MatMulNonContiguous` 错误。

### Batch Stride 计算

```rust
// mod.rs:1259-1289
// LHS batch stride
let stride_b = match lhs_stride[..lhs_stride.len() - 2] {
    [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
    [_, stride] if lhs_l.dims()[0] == 1 => stride,
    [stride, _] if lhs_l.dims()[1] == 1 => stride,
    [stride] => stride,
    [] => m * k,   // 无 batch 维度，用紧凑步长
    _ => Err(...)?,
};

// RHS batch stride
let stride_a = match rhs_stride[..rhs_stride.len() - 2] {
    [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
    [_, stride] if rhs_l.dims()[0] == 1 => stride,
    [stride, _] if rhs_l.dims()[1] == 1 => stride,
    [stride] => stride,
    [] => n * k,
    _ => Err(...)?,
};
```

### 返回的 GemmConfig

```rust
let gemm = GemmConfig {
    alpha, beta,
    m: n as i32,    // 注意：cuBLAS 中 m 是 A 的行数 (= n)
    n: m as i32,    // cuBLAS 中 n 是 B 的列数 (= m)
    k: k as i32,
    lda, ldb, ldc: n as i32,
    transa, transb,
};
```

> 关键点：candle 的 `(m, n)` 与 cuBLAS 的 `(m, n)` 含义不同。candle 做了交换：`gemm.m = n, gemm.n = m`。这是因为 candle 的 LHS 是 `(m, k)`，RHS 是 `(k, n)`，而 cuBLAS 的 A 通常是 `(m, k)`，B 通常是 `(k, n)`。

---

## 3. MatMul 主入口：`matmul`

**位置**: `mod.rs:1965-2019`

`matmul` 是 `BackendStorage for CudaStorage` trait 的方法：

```rust
fn matmul(
    &self,                          // LHS
    rhs: &Self,                     // RHS
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<Self>
```

### 执行流程

```
matmul(lhs, rhs, (b, m, n, k), lhs_l, rhs_l)
  │
  ├─ 1. 计算输出元素数: elem_count = b * m * n
  │
  ├─ 2. 按 dtype 分发:
  │   ├─ BF16 → gemm_config(bf16::ONE, bf16::ZERO, ...) → gemm_strided_batched_bf16()
  │   ├─ F16  → gemm_config(f16::ONE, f16::ZERO, ...)   → gemm_strided_batched_f16()
  │   ├─ F32  → gemm_config(1.0, 0.0, ...)             → gemm_strided_batched_f32()
  │   ├─ F64  → gemm_config(1.0, 0.0, ...)             → blas.gemm_strided_batched()
  │   └─ other → Error("dtype mismatch in matmul op")
  │
  └─ 3. 返回 CudaStorage { slice, device }
```

### alpha/beta

- `alpha = 1` (或类型的 ONE)
- `beta = 0` (或类型的 ZERO)
- 即纯乘法：`C = A * B`，不累加到已有结果

### 调用约定

从调用方传入的参数可以看到：
```
matmul(rhs, (batch, m, n, k), lhs_l, rhs_l)
```
- `self` = LHS 矩阵，形状 `(..., m, k)`
- `rhs` = RHS 矩阵，形状 `(..., k, n)`
- 结果形状 = `(..., m, n)`

---

## 4. 分数据类型的 GEMM 实现

### 4.1 F32 GEMM (`gemm_strided_batched_f32`)

**位置**: `mod.rs:2260-2308`

```rust
unsafe fn gemm_strided_batched_f32(
    cublas: &CudaBlas,
    cfg: StridedBatchedConfig<f32>,
    a: &CudaView<f32>,    // RHS
    b: &CudaView<f32>,    // LHS
    c: &mut CudaSlice<f32>,
) -> Result<(), CublasError>
```

- 调用 `cublas::result::gemm_strided_batched_ex`
- compute_type 根据 `gemm_reduced_precision_f32()` 选择：
  - `true` → `CUBLAS_COMPUTE_32F_FAST_TF32`（使用 Tensor Core TF32 加速）
  - `false` → `CUBLAS_COMPUTE_32F`（标准 FP32）
- 算法：`CUBLAS_GEMM_DEFAULT_TENSOR_OP`
- alpha/beta 通过 `*const f32 as *const _` 传递

### 4.2 F16 GEMM (`gemm_strided_batched_f16`)

**位置**: `mod.rs:2310-2367`

- 调用 `cublas::result::gemm_strided_batched_ex`
- compute_type 根据 `gemm_reduced_precision_f16()` 选择：
  - `true` → `CUBLAS_COMPUTE_16F`（FP16 累加，更快但精度较低）
  - `false` → `CUBLAS_COMPUTE_32F`（FP32 累加，默认，精度更高）
- 数据类型：`CUDA_R_16F`
- alpha/beta 指针类型取决于 compute_type：
  - FP16 累加时传 `*const f16`
  - FP32 累加时传 `*const f32`

### 4.3 BF16 GEMM (`gemm_strided_batched_bf16`)

**位置**: `mod.rs:2369-2426`

- 调用 `cublas::result::gemm_strided_batched_ex`
- compute_type 根据 `gemm_reduced_precision_bf16()` 选择：
  - `true` → `CUBLAS_COMPUTE_32F_FAST_16BF`（使用 BF16 Tensor Core 加速）
  - `false` → `CUBLAS_COMPUTE_32F`（FP32 累加，默认）
- 数据类型：`CUDA_R_16BF`
- alpha/beta 始终作为 `*const f32` 传递（cuBLAS BF16 GEMM 要求）

### 4.4 F64 GEMM

**位置**: `mod.rs:2002-2014`

- 直接调用 `blas.gemm_strided_batched(cfg, rhs, lhs, &mut out)`
- 使用 cudarc 的高级封装而非 `gemm_strided_batched_ex`
- 无精度控制选项

---

## 5. 精度控制

**位置**: `mod.rs:2215-2258`

三组全局 atomic bool 控制 GEMM 的精度模式：

| 函数 | 控制的 dtype | compute_type (false) | compute_type (true) |
|------|-------------|---------------------|--------------------|
| `set_gemm_reduced_precision_f32` | F32 | `CUBLAS_COMPUTE_32F` | `CUBLAS_COMPUTE_32F_FAST_TF32` |
| `set_gemm_reduced_precision_f16` | F16 | `CUBLAS_COMPUTE_32F` (FP32 累加) | `CUBLAS_COMPUTE_16F` (FP16 累加) |
| `set_gemm_reduced_precision_bf16` | BF16 | `CUBLAS_COMPUTE_32F` | `CUBLAS_COMPUTE_32F_FAST_16BF` |

### 默认值

全部默认为 `false`（高精度模式），与 PyTorch 行为一致。

[PyTorch issue #123157](https://github.com/pytorch/pytorch/issues/123157)

### 使用示例

```rust
use candle_core::cuda_backend::set_gemm_reduced_precision_f32;

// 启用 TF32 加速（牺牲一些精度换取约 2x 性能）
set_gemm_reduced_precision_f32(true);
```

---

## 6. MatMul 在卷积中的应用

cuBLAS 的 GEMM 也被复用于卷积运算中（通过 im2col/col2im 转换）。

### 6.1 Conv1D（无 cuDNN）

**位置**: `mod.rs:1570-1618`

```
Conv1D (im2col path)
  │
  ├─ 1. Im2Col1D: 将输入展开为列矩阵
  │     input(b, c_in, l_in) → col(b * l_out, c_in * k_size)
  │
  ├─ 2. matmul: col × kernel^T
  │     col(b*m, k) × kernel(n, k)^T → res(b*m, n)
  │     其中 m=l_out, k=c_in*k_size, n=c_out
  │
  └─ 3. transpose: 将结果 reshape 为 (b, n, l_out)
```

### 6.2 Conv2D（无 cuDNN）

**位置**: `mod.rs:1745-1798`

```
Conv2D (im2col path)
  │
  ├─ 1. Im2Col: 将输入展开为列矩阵
  │     input(b, c_in, h, w) → col(b * h_out * w_out, c_in * h_k * w_k)
  │
  ├─ 2. matmul: col × kernel^T
  │     col(b*m, k) × kernel(n, k)^T → res(b*m, n)
  │     其中 m=h_out*w_out, k=c_in*h_k*w_k, n=c_out
  │
  └─ 3. transpose: 将结果 reshape 为 (b, n, h_out, w_out)
```

### 6.3 ConvTranspose1D (col2im path)

**位置**: `mod.rs:1686-1743`

```
ConvTranspose1D (col2im path, dilation=1, padding=0, output_padding=0 时)
  │
  ├─ 1. 将 kernel reshape 为 (c_in, k_size * c_out)
  │
  ├─ 2. matmul: input^T × kernel_reshaped
  │     input^T(l_in, c_in) × kernel(c_in, c_out*k_size) → col(l_in, c_out*k_size)
  │
  ├─ 3. Col2Im1D: 将列矩阵转回图像
  │     col(b, l_in, c_out, k_size) → output(b, c_out, l_out)
  │
  └─ 4. (如果不满足 col2im 条件) → 直接使用 ConvTranspose1D kernel
```

---

## 7. 辅助工具 Traits

**位置**: `utils.rs`

这些 trait 提供了类型擦除的分发机制，使 matmul（以及其他操作）能在 `CudaStorageSlice` enum 上统一操作：

| Trait | 签名 | 用途 |
|-------|------|------|
| `Map1` | `f(src, dev, layout) -> CudaSlice<T>` | 一元操作（clone, affine, powf, elu, conv, pool 等） |
| `Map2` | `f(src1, l1, src2, l2, dev) -> CudaSlice<T>` | 二元操作（binary ops, conv, where_cond） |
| `Map2InPlace` | `f(dst, dst_l, src, src_l, dev) -> ()` | 原地二元操作（index_add, scatter, scatter_add） |
| `Map1Any` | `f(src, dev, layout, wrap) -> S` | 一元操作，输出可以是任意类型（如 reduce 返回 U32） |
| `Map2Any` | `f(src1, l1, src2, l2, dev) -> S` | 二元操作，输出可以是任意类型（如 cmp 返回 U8） |
| `Map3` | `f(s1, l1, s2, l2, s3, l3, dev) -> CudaSlice<T>` | 三元操作（当前未大量使用） |

每个 trait 的 `map` 方法对 `CudaStorageSlice` enum 的所有 8 种变体做 exhaustive match，调用对应的泛型 `f` 方法。

### CudaStorageSlice 类型别名

```rust
pub type S = super::CudaStorageSlice;
```

---

## 8. 数据流总结

### 完整 GEMM 调用链

```
用户代码
  │
  ▼
Tensor::matmul(&self, &rhs)                    // src/tensor.rs
  │
  ▼
BackendStorage::matmul(self, rhs, (b,m,n,k), lhs_l, rhs_l)
  │                                           // mod.rs:1965
  ├─ gemm_config(alpha, beta, (b,m,n,k), lhs_l, rhs_l)
  │                                           // mod.rs:1200
  │   ├─ 分析 LHS stride → (ldb, transb)
  │   ├─ 分析 RHS stride → (lda, transa)
  │   ├─ 计算 batch stride → stride_a, stride_b
  │   └─ 返回 StridedBatchedConfig { batch_size, gemm, stride_a, stride_b, stride_c }
  │
  ├─ 分配输出内存: dev.alloc::<T>(b * m * n)
  │
  └─ gemm_strided_batched_<dtype>(&blas, cfg, rhs_slice, lhs_slice, &mut out)
      │                                       // mod.rs:2260/2310/2369
      └─ cublas::gemm_strided_batched_ex(
             handle, transa, transb, m, n, k,
             alpha, A, A_type, lda, stride_a,
                    B, B_type, ldb, stride_b,
             beta,  C, C_type, ldc, stride_c,
             batch_size, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP
         )
```

### 关键设计决策

1. **StridedBatched 而非普通 GEMM**：支持 batch matmul，避免在 batch 维度上循环调用 GEMM。

2. **Tensor Core 默认算法**：使用 `CUBLAS_GEMM_DEFAULT_TENSOR_OP`，在支持的 GPU (Volta+) 上自动使用 Tensor Core。

3. **LHS/RHS 的角色**：cuBLAS 中 `C = alpha * op(A) * op(B) + beta * C`，candle 中 rhs 映射到 A，lhs 映射到 B（与函数参数顺序相反）。

4. **transpose 而非 copy**：当矩阵需要转置时，通过 `transa`/`transb` 参数告知 cuBLAS，而不是实际复制数据。

5. **MatMulNonContiguous 硬错误**：只有 4 种合法的 stride 排列。不满足时直接报错，不尝试先拷贝为连续内存（由上层调用者决定是否做拷贝）。

### 原始 GEMM kernel 的替代路径

当前 candle CUDA backend **没有**手写的矩阵乘法 CUDA kernel。所有 matmul 都通过 cuBLAS 完成。手写 kernel（在 `candle-kernels` crate 中定义，通过 `kernels::BINARY`、`kernels::UNARY` 等模块加载）仅用于：

- 逐元素一元/二元操作
- Reduce (sum, min, max, argmin, argmax)
- Convolution (conv1d, conv2d, conv_transpose1d, conv_transpose2d)
- Pooling (max_pool2d, avg_pool2d)
- Indexing (index_select, gather, index_add, scatter, scatter_add)
- 类型转换 (cast)
- 填充/复制 (const_set, copy2d, copy_strided_src)
