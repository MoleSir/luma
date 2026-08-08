# Candle CUDA Backend - contiguous 与 cat 实现参考

> 基于 `src/cuda_backend/` 和 `src/tensor_cat.rs` 源码分析，导出日期：2026-08-04

---

## 目录

1. [contiguous 实现](#1-contiguous-实现)
2. [cat (Concatenate) 实现](#2-cat-concatenate-实现)
3. [底层 CUDA 操作详解](#3-底层-cuda-操作详解)
4. [数据流总结](#4-数据流总结)

---

## 1. contiguous 实现

`contiguous()` 确保张量在内存中是行主序（row-major）连续排列的。核心逻辑在 `Tensor` 层，CUDA backend 提供两个底层操作：`try_clone` 和 `copy_strided_src`。

### 1.1 Tensor 层入口

**位置**: `src/tensor.rs:2295-2306`

```rust
pub fn contiguous(&self) -> Result<Tensor> {
    if self.is_contiguous() {
        Ok(self.clone())       // 已连续：clone（共享同一块 GPU 内存，引用计数+1）
    } else {
        let mut storage = unsafe {
            self.device().alloc_uninit(shape, self.dtype())?
        };
        self.storage()
            .copy_strided_src(&mut storage, 0, self.layout())?;
        Ok(from_storage(storage, shape.clone(), op, false))
    }
}
```

还会调用 `contiguous` 的变体 `force_contiguous`（`tensor.rs:2309-2316`），它始终分配新内存并复制。

### 1.2 已连续路径：`try_clone`

**位置**: `src/cuda_backend/mod.rs:1295-1299`

```rust
fn try_clone(&self, layout: &Layout) -> Result<Self> {
    let slice = Clone.map(&self.slice, self.device(), layout)?;
    let device = self.device.clone();
    Ok(Self { slice, device })
}
```

`Clone` 结构体实现 `Map1` trait，其 `f` 方法调用 `CudaSlice::try_clone()`：

```rust
// mod.rs:77-87
struct Clone;
impl Map1 for Clone {
    fn f<T: DeviceRepr>(&self, s: &CudaSlice<T>, _: &CudaDevice, _: &Layout
    ) -> Result<CudaSlice<T>> {
        s.try_clone().w()   // cudarc API → 底层 cudaMemcpyAsync(D2D)
    }
}
```

- `try_clone` 是 cudarc 提供的 API，基于 CUDA 事件跟踪机制管理引用计数
- 多个 `CudaSlice` 可以指向同一块 GPU 内存，当所有引用释放后自动回收

### 1.3 非连续路径：`copy_strided_src`

**位置**: `src/cuda_backend/mod.rs:2068-2212`

```rust
fn copy_strided_src(
    &self,
    dst: &mut Self,
    dst_offset: usize,
    src_l: &Layout,
) -> Result<()> {
    let el_count = src_shape.elem_count();
    if el_count == 0 { return Ok(()); }

    let cfg = LaunchConfig::for_num_elems(el_count as u32);
    let ds = SlicePtrOrNull::params_from_layout(dev, src_l)?;
    // ds 仅在非连续时包含 [dims, strides]，连续时为 Null

    match (&self.slice, &mut dst.slice) {
        (CudaStorageSlice::F32(src), CudaStorageSlice::F32(dst)) => {
            let (src, mut dst) = slice_src_and_dst(src, src_l, dst, dst_offset);
            if src_l.is_contiguous() {
                dev.memcpy_dtod(&src, &mut dst)?  // 连续：一次 memcpy
            } else {
                // 非连续：启动 ucopy_f32 kernel
                let func = dev.get_or_load_func("ucopy_f32", &kernels::UNARY)?;
                barg!(builder, el_count, dims.len());
                ds.builder_arg(&mut builder);  // dims + strides
                builder.arg(&src);
                builder.arg(&mut dst);
                unsafe { builder.launch(cfg) }.w()?;
            }
        }
        // ... BF16, F16, F64, U8, U32, I64, F8E4M3 同理
    }
}
```

**两种子路径**：

| 条件 | 底层操作 | 开销 |
|------|---------|------|
| `src_l.is_contiguous()` | `memcpy_dtod` (cudaMemcpyAsync) | O(n) 带宽受限，极快 |
| 非连续 | `ucopy_{dtype}` kernel | O(n) 逐元素按 stride 跳读，较慢 |

**`ucopy` kernel 伪代码**：

```c
// ucopy_f32 kernel
__global__ void ucopy(int el_count, int ndims, int* dims_strides,
                      const float* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= el_count) return;
    // 将线性索引映射回多维索引，再按源 stride 读取
    int src_idx = linear_to_strided(idx, ndims, dims_strides);
    dst[idx] = src[src_idx];
}
```

### 1.4 与 CPU backend 的对比

| 操作 | CPU backend | CUDA backend |
|------|------------|-------------|
| 连续 clone | memcpy | `CudaSlice::try_clone` (引用计数，不复制) |
| 非连续 copy | 逐元素 strided loop | `ucopy` kernel (GPU 并行逐元素) |

### 1.5 `contiguous` 调用场景

`contiguous()` 在 candle 中被广泛用于那些要求输入连续的底层操作之前，包括：
- `matmul` — 要求特定 stride 排列（4 种合法排布之一），不匹配时报 `MatMulNonContiguous` 错误
- `conv1d`/`conv2d` — im2col 后确保 kernel 连续
- `index_select` / `gather` / `index_add` — 要求源和目标连续

---

## 2. cat (Concatenate) 实现

`Tensor::cat` 位于独立文件 `src/tensor_cat.rs`，按三条路径分发：

### 2.1 总调度逻辑

**位置**: `src/tensor_cat.rs:21-74`

```rust
pub fn cat<A: AsRef<Tensor>, D: Dim>(args: &[A], dim: D) -> Result<Self> {
    // 1. 验证：所有张量 rank 相同，非 dim 维度大小一致
    // 2. 选择路径：
    let all_contiguous = args.iter().all(|v| v.as_ref().is_contiguous());
    if all_contiguous {
        Self::cat_contiguous(args, dim)   // 路径 A: 全部连续
    } else if dim == 0 {
        Self::cat0(args)                  // 路径 B: dim=0 拼接
    } else {
        // 路径 C: 非连续 + 非 dim0 → 转置到 dim0，cat，再转置回来
        let args: Vec<Tensor> = args
            .iter()
            .map(|a| a.as_ref().transpose(0, dim))
            .collect::<Result<Vec<_>>>()?;
        let cat = Self::cat0(&args)?;
        cat.transpose(0, dim)
    }
}
```

### 2.2 路径 A：`cat_contiguous`（全部连续 + 任意 dim）

**位置**: `src/tensor_cat.rs:151-237`

所有张量连续时，利用 `copy2d` kernel 做高效的块状复制。

```rust
fn cat_contiguous(args: &[Tensor], dim: usize) -> Result<Self> {
    let block_size: usize = cat_dims.iter().skip(1 + dim).product();
    let cat_target_dim_len = cat_dims[dim];

    let mut storage = unsafe { device.alloc_uninit(&shape, dtype)? };
    let mut dst_o = 0;
    for arg in args {
        let d1: usize = arg_dims.iter().take(dim).product();
        let d2 = block_size * arg_dims[dim];
        let dst_s = block_size * cat_target_dim_len;
        arg.storage().copy2d(
            &mut storage,
            d1,          // 行数（展平 dim 之前的维度）
            d2,          // 每行元素数（dim 及其之后的维度乘积）
            /* src_s */ d2,
            dst_s,       // 目标行步长 = 拼接后的总行宽
            src_o,       // 源起始偏移
            dst_o,       // 目标起始偏移（递增）
        )?;
        dst_o += d2;
    }
}
```

**`copy2d` 参数图解**（cat 两个张量在 dim=1）：

```
A: shape=(2, 3), B: shape=(2, 4), cat_dim=1 → result=(2, 7)

block_size = 1  (dim=1 之后没有维度)
d1 = 2          (dim=1 之前的维度乘积 = dim0)
d2_A = 1*3 = 3  (block_size * A's dim_size)
d2_B = 1*4 = 4
dst_s = 1*7 = 7 (block_size * total dim_size)

目标内存 (2行×7列):
┌───────────────────────────────┐
│ A₀  A₁  A₂ │ B₀  B₁  B₂  B₃ │  ← row 0: d2_A=3 cols from A, d2_B=4 cols from B
├───────────────────────────────┤
│ A₃  A₄  A₅ │ B₄  B₅  B₆  B₇ │  ← row 1
└───────────────────────────────┘
     ↑ 7 cols (dst_s) ↑

对 A: copy2d(d1=2, d2=3, src_s=3, dst_s=7, dst_o=0)
     复制 2 行，每行从 A 取 3 个元素，写入 dst 的每行前 3 列
对 B: copy2d(d1=2, d2=4, src_s=4, dst_s=7, dst_o=3)
     复制 2 行，每行从 B 取 4 个元素，写入 dst 每行从第 3 列开始的 4 列
```

### 2.3 路径 B：`cat0`（dim=0 拼接）

**位置**: `src/tensor_cat.rs:76-149`

dim=0 时，拼接退化为简单的前后追加。每个张量按 `elem_count` 计算 offset，使用 `copy_strided_src` 复制。

```rust
fn cat0(args: &[Tensor]) -> Result<Self> {
    let mut offsets = vec![0usize];
    for arg in args {
        let next = offsets.last().unwrap() + arg.elem_count();
        offsets.push(next);
    }
    let mut storage = unsafe { device.alloc_uninit(&shape, dtype)? };
    for (arg, &offset) in args.iter().zip(offsets.iter()) {
        arg.storage().copy_strided_src(&mut storage, offset, arg.layout())?;
    }
    Ok(from_storage(storage, shape, op, false))
}
```

**`cat0` 的 `copy_strided_src` 行为**：

```
A: shape=(2, 3), B: shape=(2, 4), cat_dim=0 → result=(4, max(3,4))? 
   → 实际 dim=0 要求在 dim=0 拼接，所以 A(2,3) + B(2,4) → 不合法(dim=1 不等)
   
正确示例: A(2,3) + B(3,3), dim=0 → result(5,3)

offsets: [0, 6, 15]
   A: 6 个元素 → copy_strided_src(dst, offset=0, A's layout)
   B: 9 个元素 → copy_strided_src(dst, offset=6, B's layout)

内存布局: [A₀ A₁ A₂ A₃ A₄ A₅ | B₀ B₁ B₂ B₃ B₄ B₅ B₆ B₇ B₈]
           ←── 6 elements ──→  ←────── 9 elements ────────→
```

### 2.4 路径 C：转置回绕（非连续 + 非 dim0）

```rust
// 1. 将所有张量沿 dim 和 dim0 转置
//    args[i].transpose(0, dim) → 把目标 dim 换到 dim0
// 2. cat0 → 现在 dim0 就是原始的 dim
// 3. 结果.transpose(0, dim) → 转回去
```

**示例**：`cat([A, B], dim=2)` 且 A, B 非连续

```
A: (2, 3, 4)      transpose(0,2)   A': (4, 3, 2)
B: (2, 3, 5)  ──────────────────>  B': (5, 3, 2)
                                       ↓ cat(dim=0)
C': (9, 3, 2)  ←──────────────────  cat0 → (9, 3, 2)
    transpose(0,2)

C: (2, 3, 9)
```

**注意**：tensor 的 `transpose` 不会移动数据，只改变 `Layout` 中的 stride。所以路径 C 不产生额外数据复制，它只是一个元数据操作。

### 2.5 路径选择总结

```
cat(args, dim)
  │
  ├─ all contiguous? ──Yes──→ cat_contiguous(args, dim)
  │                              └─ 每个 arg → copy2d (kernel)
  │
  └─ No
      ├─ dim == 0? ──Yes──→ cat0(args)
      │                       └─ 每个 arg → copy_strided_src
      │                          ├─ arg 连续 → memcpy_dtod
      │                          └─ arg 非连续 → ucopy kernel
      │
      └─ dim != 0? ──Yes──→ transpose(0, dim) → cat0 → transpose(0, dim)
```

---

## 3. 底层 CUDA 操作详解

`contiguous` 和 `cat` 在 CUDA backend 层涉及三个底层操作：

### 3.1 `try_clone`

| 属性 | 值 |
|------|---|
| 位置 | `mod.rs:77-87` |
| 底层 | `CudaSlice::try_clone()` → cudarc 内部引用计数管理 |
| 数据传输 | 无数据复制（共享同一块 GPU 内存） |
| 适用场景 | 已连续张量的 clone |

### 3.2 `copy_strided_src`

| 属性 | 值 |
|------|---|
| 位置 | `mod.rs:2068-2212` |
| 子路径 | 连续 → `memcpy_dtod` (cudaMemcpyAsync)；非连续 → `ucopy_{dtype}` kernel |
| kernel 来源 | `kernels::UNARY` 模块 |
| 启动参数 | `el_count`(元素数), `dims.len()`(维数), dims+strides 数组 |
| 支持的 dtype | U8, U32, I64, BF16, F16, F32, F64, F8E4M3 |

**ucopy kernel 签名示例**（来自 `candle-kernels` crate）：

```rust
// 编译为 PTX，按需加载
let func = dev.get_or_load_func("ucopy_f32", &kernels::UNARY)?;
// kernel 启动:
//   el_count: 总元素数
//   ndims: 维度数
//   ds: [dims..., strides...]
//   src: 源指针
//   dst: 目标指针
```

### 3.3 `copy2d`

| 属性 | 值 |
|------|---|
| 位置 | `mod.rs:2021-2066` |
| kernel 名称 | `copy2d_{dtype}` |
| kernel 来源 | `kernels::FILL` 模块 |
| 启动参数 | `d1`(行数), `d2`(每行元素数), `src_s`(源行步长), `dst_s`(目标行步长), `src` 指针, `dst` 指针 |
| 支持的 dtype | U8, U32, I64, BF16, F16, F32, F64, F8E4M3 |

**copy2d kernel 伪代码**：

```c
// copy2d_f32 kernel
__global__ void copy2d(const float* src, float* dst,
                       int d1, int d2, int src_s, int dst_s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d1 * d2;
    if (idx >= total) return;

    int row = idx / d2;       // 行索引
    int col = idx % d2;       // 列索引
    dst[row * dst_s + col] = src[row * src_s + col];
}
```

**关键特性**：源行宽和目标行宽（以元素计）相同（都是 `d2`），但行步长可以不同。这正好适合 cat 操作：每个输入张量在自己的行上连续，写入目标时跳过其他张量的列。

### 3.4 函数辅助：`slice_src_and_dst`

**位置**: `mod.rs:1112-1129`

```rust
fn slice_src_and_dst<'a, T>(
    src: &'a CudaSlice<T>,
    src_l: &Layout,
    dst: &'a mut CudaSlice<T>,
    dst_offset: usize,
) -> (CudaView<'a, T>, CudaViewMut<'a, T>) {
    let src_offset = src_l.start_offset();
    let to_copy = dst.len()
        .saturating_sub(dst_offset)
        .min(src.len().saturating_sub(src_offset));
    let src = src.slice(src_offset..src_offset + to_copy);
    let dst = dst.slice_mut(dst_offset..dst_offset + to_copy);
    (src, dst)
}
```

该函数确保源和目标切片长度匹配，避免越界访问。

---

## 4. 数据流总结

### 4.1 `contiguous` 完整调用链

```
Tensor::contiguous()
  │  tensor.rs:2295
  ├─ is_contiguous() == true
  │   └─ self.clone()
  │       └─ Storage::try_clone(&layout)
  │           └─ CudaStorage::try_clone()
  │               └─ CudaSlice::try_clone()   // cudarc: 引用计数+1，无数据复制
  │
  └─ is_contiguous() == false
      └─ device.alloc_uninit(shape, dtype)    // 分配新 GPU 内存
          └─ storage.copy_strided_src(&mut new_storage, 0, layout)
              └─ CudaStorage::copy_strided_src()
                  ├─ layout.is_contiguous()
                  │   └─ memcpy_dtod(src, dst)  // cudaMemcpyAsync, 一次 D2D
                  └─ !layout.is_contiguous()
                      └─ ucopy_{dtype} kernel   // GPU 逐元素 strided copy
```

### 4.2 `cat` 完整调用链

```
Tensor::cat(args, dim)
  │  tensor_cat.rs:21
  │
  ├─ [全部连续] cat_contiguous(args, dim)
  │   └─ for each tensor:
  │       storage.copy2d(&mut dst, d1, d2, d2, dst_s, src_o, dst_o)
  │       └─ CudaStorage::copy2d()
  │           └─ copy2d_{dtype} kernel  // GPU 2D 块状复制
  │
  ├─ [dim=0] cat0(args)
  │   └─ for each tensor:
  │       storage.copy_strided_src(&mut dst, offset, layout)
  │       └─ 同 4.1 的 copy_strided_src 路径
  │
  └─ [非连续, dim≠0] transpose(0, dim) → cat0 → transpose(0, dim)
      └─ 转置仅修改 Layout (stride)，无数据移动
```

### 4.3 性能特征

| 操作 | CUDA 调用 | 并行度 | 带宽利用 |
|------|----------|--------|---------|
| 连续 clone | 无数据复制 (引用计数) | N/A | 最优 |
| 连续 copy_strided_src | `cudaMemcpyAsync` | GPU DMA 引擎 | 接近峰值 |
| 非连续 copy_strided_src | `ucopy` kernel | 每元素一线程 | 受限于 stride 跳读模式 |
| copy2d | `copy2d` kernel | 每元素一线程 | 良好（源和目标都按行连续） |
| transpose | 仅修改 stride 元数据 | N/A | 零开销 |

### 4.4 模块索引

| 文件 | 内容 |
|------|------|
| `src/tensor.rs` | `contiguous()`, `force_contiguous()`, `is_contiguous()` |
| `src/tensor_cat.rs` | `cat()`, `cat0()`, `cat_contiguous()`, `slice_set()` |
| `src/storage.rs` | `Storage::copy_strided_src()`, `Storage::copy2d()` — 按 device 分发 |
| `src/cuda_backend/mod.rs` | `CudaStorage::try_clone()`, `copy_strided_src()`, `copy2d()` |
| `src/cuda_backend/utils.rs` | `Map1`, `Map2` — 类型擦除分发 trait |
| `src/cuda_backend/device.rs` | `CudaDevice::alloc_uninit()`, kernel 加载 |
