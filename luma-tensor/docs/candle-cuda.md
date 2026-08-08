# candle CUDA 后端架构分析：基于 cudarc 的实现

## 1. 整体分层架构

candle 的 CUDA 后端分为 **四层**，自底向上分别为：

```
┌──────────────────────────────────────────────┐
│  Tensor / Op 层 (tensor.rs, op.rs)           │  ← 面向用户的高级 API
├──────────────────────────────────────────────┤
│  BackendStorage trait (cuda_backend/mod.rs)  │  ← 为每个操作实现 CUDA 内核启动
├──────────────────────────────────────────────┤
│  CudaDevice (device.rs)                      │  ← 设备抽象、内存分配、模块管理
├──────────────────────────────────────────────┤
│  candle-kernels (.cu → .ptx)                 │  ← 原始的 CUDA C++ 内核
└──────────────────────────────────────────────┘
```

## 2. `cudarc` 扮演的角色

`cudarc` 是 CUDA 驱动 API（driver API）的**安全的 Rust 封装**。candle **不直接调用** `libcudart.so` 或编写 `unsafe` FFI —— 它完全通过 `cudarc` 完成底层交互。关键在于：

- `cudarc` 提供的是 **CUDA driver API**（`cuMemAlloc`、`cuLaunchKernel`、`cuModuleLoad` 等），而非更常见的 CUDA runtime API（`cudaMalloc`、核函数启动语法 `kernel<<<>>>`）。
- candle 使用 cudarc 的核心类型：
  - **`driver::CudaSlice<T>`** —— GPU 内存分配（相当于设备侧 `malloc`）
  - **`driver::CudaStream`** —— CUDA 流（所有操作默认为异步）
  - **`driver::CudaContext`** —— CUDA 上下文
  - **`driver::CudaModule`** —— 已加载的 PTX 模块
  - **`driver::CudaFunction`** —— PTX 模块中的特定内核函数句柄
  - **`driver::LaunchConfig`** —— 网格/线程块维度（`gridDim`、`blockDim`）
  - **`driver::LaunchArgs`** —— 内核参数构建器
  - **`cublas::CudaBlas`** —— cuBLAS 句柄，用于矩阵乘法
  - **`curand::CudaRng`** —— cuRAND 句柄，用于随机数生成

## 3. `CudaDevice`：设备抽象

源码位置：`candle-core/src/cuda_backend/device.rs:34-42`

```rust
pub struct CudaDevice {
    id: DeviceId,                           // 唯一设备标识符
    context: Arc<CudaContext>,              // cudarc 提供的 CUDA 上下文
    modules: Arc<RwLock<ModuleStore>>,      // 编译好的 PTX 模块缓存
    custom_modules: Arc<RwLock<HashMap<...>>>, // 用户自定义模块
    stream: Arc<CudaStream>,                // 默认 CUDA 流（异步）
    blas: Arc<CudaBlas>,                    // cuBLAS 句柄
    curand: Arc<Mutex<CudaRng>>,            // cuRAND 句柄
}
```

### 初始化

源码位置：`device.rs:262-279`

```rust
fn new(ordinal: usize) -> Result<Self> {
    let context = cudarc::driver::CudaContext::new(ordinal).w()?;  // 根据 GPU 索引创建上下文
    let stream = context.default_stream();                          // 获取默认流
    let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?; // 创建 cuBLAS 句柄
    let curand = cudarc::curand::CudaRng::new(299792458, ...).w()?;// 创建 cuRAND 句柄
    ...
}
```

### 模块延迟加载

源码位置：`device.rs:217-235` —— 这是关键性能设计：

```rust
pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
    // 先检查缓存（读锁）
    let ms = self.modules.read().unwrap();
    if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
        return Ok(...);  // 命中缓存，直接使用已编译的模块
    }
    drop(ms);
    // 缓存未命中，编译并存储（写锁）
    let mut ms = self.modules.write().unwrap();
    let cuda_module = self.context.load_module(mdl.ptx().into()).w()?; // ← cudarc 调用
    ms.mdls[mdl.index()] = Some(cuda_module.clone());
    ...
}
```

PTX 模块仅在**首次使用**时加载到 GPU，并通过 `RwLock` 缓存。`ModuleStore` 只是一个固定大小的数组（每种内核类型一个槽位）。

## 4. PTX 内核如何编译和嵌入

### 构建流程

源码位置：`candle-kernels/build.rs`

```
.cu 文件 → bindgen_cuda → nvcc 编译 → .ptx 文件
                                    ↓
                            include_str!() 嵌入到 ptx.rs
```

### ptx.rs（由构建脚本自动生成）

源码位置：`candle-kernels/src/ptx.rs`

```rust
pub const AFFINE: &str = include_str!(concat!(env!("OUT_DIR"), "/affine.ptx"));
pub const BINARY: &str = include_str!(concat!(env!("OUT_DIR"), "/binary.ptx"));
pub const CAST: &str = include_str!(concat!(env!("OUT_DIR"), "/cast.ptx"));
pub const CONV: &str = include_str!(concat!(env!("OUT_DIR"), "/conv.ptx"));
pub const FILL: &str = include_str!(concat!(env!("OUT_DIR"), "/fill.ptx"));
pub const INDEXING: &str = include_str!(concat!(env!("OUT_DIR"), "/indexing.ptx"));
pub const QUANTIZED: &str = include_str!(concat!(env!("OUT_DIR"), "/quantized.ptx"));
pub const REDUCE: &str = include_str!(concat!(env!("OUT_DIR"), "/reduce.ptx"));
pub const SORT: &str = include_str!(concat!(env!("OUT_DIR"), "/sort.ptx"));
pub const TERNARY: &str = include_str!(concat!(env!("OUT_DIR"), "/ternary.ptx"));
pub const UNARY: &str = include_str!(concat!(env!("OUT_DIR"), "/unary.ptx"));
```

这意味着编译好的 PTX 代码**作为字符串常量直接嵌入到最终的 Rust 二进制文件中** —— 运行时不需要外部 `.ptx` 文件。

### 模块索引

源码位置：`candle-kernels/src/lib.rs`

```rust
pub struct Module { index: usize, ptx: &'static str }
pub const UNARY: Module = Module { index: module_index(Id::Unary), ptx: ptx::UNARY };
pub const BINARY: Module = Module { index: module_index(Id::Binary), ptx: ptx::BINARY };
// ... 共 11 个模块
```

## 5. 数据存储：`CudaStorage` 和 `CudaStorageSlice`

源码位置：`candle-core/src/cuda_backend/mod.rs:66-75, 1132-1135`

```rust
// 枚举变体：每种 dtype 一个 —— 这是类型安全的 GPU 内存
pub enum CudaStorageSlice {
    U8(CudaSlice<u8>),
    U32(CudaSlice<u32>),
    I64(CudaSlice<i64>),
    BF16(CudaSlice<bf16>),
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
    F8E4M3(CudaSlice<F8E4M3>),
}

pub struct CudaStorage {
    pub slice: CudaStorageSlice,  // GPU 上的实际数据
    pub device: CudaDevice,       // 数据"属于"哪个设备
}
```

`CudaSlice<T>` 是 `cudarc` 对 GPU 设备内存的安全抽象 —— 它是引用计数的，并在 drop 时自动释放。

## 6. 内核启动模式：操作的实际执行方式

这是 candle 设计的**核心模式**。每个操作都遵循完全相同的步骤。以一元取反（Neg）为例：

源码位置：`mod.rs:368-394`

```rust
// 步骤 1：获取形状信息
let shape = layout.shape();
let dims = shape.dims();
let el_count = shape.elem_count();

// 步骤 2：创建 LaunchConfig（每个元素一个线程）
let cfg = LaunchConfig::for_num_elems(el_count as u32);  // 自动选择网格/线程块大小

// 步骤 3：准备步幅信息（用于非连续张量）
let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
// 如果张量是连续的 → SlicePtrOrNull::Null（用 nullptr 表示"使用快速路径"）
// 如果不连续 → 包含一个 [dims..., strides...] 的 CudaSlice

// 步骤 4：按名称获取内核函数（延迟加载 + 缓存）
let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), &kernels::UNARY)?;
// 对于 f32 Neg，查找的是 "uneg_f32"

// 步骤 5：分配输出
let mut out = unsafe { dev.alloc::<T>(el_count)? };

// 步骤 6：构建参数并启动
let mut builder = func.builder();   // cudarc 的 LaunchArgs 构建器
barg!(builder, el_count);          // 宏：推送 i32 值参数
barg!(builder, dims.len());        // 推送维度数量
ds.builder_arg(&mut builder);       // 推送 info 指针（或 nullptr/null）
builder.arg(src);                   // 推送输入数据指针
builder.arg(&mut out);             // 推送输出数据指针
unsafe { builder.launch(cfg) }.w()?;  // ← cudarc 调用 cuLaunchKernel
```

## 7. CUDA 内核自身：PTX 代码中的实际情况

### 一元操作内核

以 `uneg_f32` 内核为例，展开自 `UNARY_OP` 宏（`candle-kernels/src/unary.cu`）：

```c
extern "C" __global__ void uneg_f32(
    const size_t numel,          // 元素总数
    const size_t num_dims,       // 维度数量
    const size_t *info,          // [dims..., strides...] 或 nullptr
    const float *inp,            // 输入指针
    float *out                   // 输出指针
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;

    // 快速路径：连续或 null info
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numel;
             i += blockDim.x * gridDim.x) {       // 网格步进循环
            float x = inp ? inp[i] : out[i];
            out[i] = -x;
        }
    }
    // 慢速路径：非连续（带步幅的张量）
    else {
        for (unsigned int i = ...; i < numel; i += ...) {
            unsigned strided_i = logical_index_to_physical_index(i, num_dims, dims, strides);
            float x = inp ? inp[strided_i] : out[i];
            out[i] = -x;
        }
    }
}
```

关键 CUDA 概念：
- `blockIdx.x * blockDim.x + threadIdx.x` → 全局线程索引
- **网格步进循环**（`i += blockDim.x * gridDim.x`）：保证即使线程数少于元素数也能正确处理
- `is_contiguous()` 检查（`cuda_utils.cuh:9-23`）：如果步幅匹配行主序（row-major）顺序，则使用快速直接索引；否则使用 `logical_index_to_physical_index()` 将逻辑索引映射到物理内存偏移

### 二元操作内核

展开自 `BINARY_OP` 宏（`candle-kernels/src/binary_op_macros.cuh`），二元内核处理四种广播情况的组合：

```c
// 情况1：两个操作数都连续 → 直接索引
// 情况2：仅 lhs 连续 → 为 rhs 计算步幅索引
// 情况3：仅 rhs 连续 → 为 lhs 计算步幅索引
// 情况4：都不连续 → 为两者计算步幅索引
```

### 规约操作内核（Reduce）

源码位置：`candle-kernels/src/reduce.cu`

以 `fast_sum` 为例：采用**并行规约**（parallel reduction）模式，使用共享内存：

```c
__shared__ T shr[BLOCK_SIZE];     // 共享内存 —— 线程块内所有线程可见
// 每个线程将元素加载到 shr[threadIdx.x]
// 然后进行树状规约：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();              // 屏障同步
    if (tid < s)
        shr[tid] += shr[tid + s]; // 步幅减半的规约
}
// shr[0] 包含该块的部分和
```

### LayerNorm / RmsNorm / Softmax

这些是从 llama.cpp 的 CUDA 实现改编而来，使用 **warp shuffle** 指令（`__shfl_xor_sync`）进行高效的跨线程规约，避免共享内存开销。

### RoPE（旋转位置编码）

直接在 GPU 上计算旋转位置编码，支持多种张量布局（`rope`、`rope_i`、`rope_thd`）。

## 8. 类型分发：`Map1` / `Map2` 特质

由于 `CudaStorageSlice` 是一个枚举体，candle 使用辅助特质来对 dtype 变体进行分发。

源码位置：`candle-core/src/cuda_backend/utils.rs`

```rust
pub trait Map1 {
    // 泛型方法 —— 针对每种类型参数化一次
    fn f<T: DeviceRepr + WithDType>(&self, src: &CudaSlice<T>, ...) -> Result<CudaSlice<T>>;

    // 分发方法 —— 对 CudaStorageSlice 变体进行匹配
    fn map(&self, s: &S, d: &CudaDevice, l: &Layout) -> Result<S> {
        match s {
            S::F32(s) => S::F32(self.f(s, d, l)?),
            S::F16(s) => S::F16(self.f(s, d, l)?),
            // ... 每种类型一个分支（共8种）
        }
    }
}
```

同样存在对应的特质：
- **`Map2`** —— 俩输入 → 一输出（二元操作，如 Add、Mul）
- **`Map3`** —— 三输入 → 一输出（三元操作）
- **`Map1Any`** —— 单输入，输出类型可能不同（如 Reduce 返回 U32）
- **`Map2Any`** —— 双输入，输出类型可能不同（如 Cmp 返回 U8）
- **`Map2InPlace`** —— 双输入，原地修改（如 ScatterAdd）

这意味着每个操作（Neg、Add、GELU 等）**为每种类型实现一次泛型方法**，而 `map` 方法为所有 8 种 dtype 变体进行匹配分发。

## 9. 内置内核操作清单

candle 在 11 个 PTX 模块中提供了约 200+ 个类型特化的内核函数：

| 模块 | .cu 文件 | 包含的操作 |
|------|---------|-----------|
| `AFFINE` | `affine.cu` | affine 变换 |
| `BINARY` | `binary.cu` | add, sub, mul, div, min, max, eq, ne, lt, le, gt, ge（所有类型组合）|
| `CAST` | `cast.cu` | 所有 dtype 转换组合 |
| `CONV` | `conv.cu` | conv1d/2d, conv_transpose1d/2d, im2col, col2im, pool2d, upsample |
| `FILL` | `fill.cu` | const_set, copy2d |
| `INDEXING` | `indexing.cu` | index_select, gather, index_add, scatter |
| `QUANTIZED` | `quantized.cu` | 量化矩阵乘法 |
| `REDUCE` | `reduce.cu` | sum, min, max, argmin, argmax, softmax, rmsnorm, layernorm, rope |
| `SORT` | `sort.cu` | 排序操作 |
| `TERNARY` | `ternary.cu` | where（条件选择）|
| `UNARY` | `unary.cu` | neg, exp, log, sin, cos, abs, recip, sqr, sqrt, gelu, relu, silu, tanh, erf, floor, ceil, round, sign, sigmoid, powf, elu 等 |

## 10. 特殊情况：矩阵乘法（MatMul）使用 cuBLAS

与直接使用自定义内核不同，矩阵乘法使用 cuBLAS。

源码位置：`mod.rs:1965-2019`

```rust
fn matmul(&self, rhs: &Self, (b, m, n, k): ..., lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
    // 为 cuBLAS 的 strided-batched GEMM 计算步幅配置
    let cfg = gemm_config(1., 0., (b, m, n, k), lhs_l, rhs_l)?;
    // 处理各种转置/步幅组合

    // 调用 cudarc 对 cublasGemmStridedBatchedEx 的封装
    unsafe { gemm_strided_batched_f32(&self.device.blas, cfg, rhs, lhs, &mut out) }.w()?;
}
```

`gemm_config` 函数（`mod.rs:1200-1290`）处理关键的 cuBLAS 布局逻辑：
- 检测是否需要转置（基于步幅模式）
- 确定 leading dimensions（`lda`、`ldb`、`ldc`）
- 为 strided-batched 操作计算批次步幅

**Tensor Core 支持**（`mod.rs:2215-2258`）：
```rust
pub fn set_gemm_reduced_precision_f32(b: bool)  // 启用 TF32
pub fn set_gemm_reduced_precision_f16(b: bool)  // 启用 FP16 累加
pub fn set_gemm_reduced_precision_bf16(b: bool) // 启用 BF16 快速模式
```

## 11. 卷积：im2col + MatMul，或 cuDNN

### 无 cuDNN（默认路径）

源码位置：`mod.rs:1745-1798`

1. 使用 `Im2Col` 内核将输入展开为列矩阵
2. 使用 cuBLAS 进行矩阵乘法（`matmul`）
3. 将结果转置/重塑为输出维度

### 有 cuDNN（`feature = "cudnn"`）

源码位置：`mod.rs:1800-1864`

直接调用 cuDNN 的卷积原语，按 dtype 分支：
- `U8` → `cudnn::launch_conv2d::<u8, u8>`
- `BF16` → `cudnn::launch_conv2d::<bf16, f32>`（伪 BF16，内部以 f32 计算）
- `F16` → `cudnn::launch_conv2d::<f16, f16>`
- `F32` → `cudnn::launch_conv2d::<f32, f32>`
- `F64` → `cudnn::launch_conv2d::<f64, f64>`

## 12. 内存传输

源码位置：`device.rs:50-106`

candle 通过 `cudarc` 提供了便捷的内存传输方法：

| 方法 | 方向 | cudarc 底层调用 |
|------|------|----------------|
| `memcpy_htod` | Host → Device | `cuMemcpyHtoDAsync` |
| `memcpy_dtov` | Device → Host | `cuMemcpyDtoHAsync` |
| `memcpy_dtod` | Device → Device | `cuMemcpyDtoDAsync` |
| `memcpy_stod` | Host Slice → Device | `cuMemcpyHtoDAsync` + 分配 |
| `alloc` | Device 分配 | `cuMemAlloc` |
| `alloc_zeros` | Device 分配+清零 | `cuMemAlloc` + 初始化 |

所有操作都是**异步的**（在默认 CUDA 流上），可通过 `synchronize()` 进行屏障同步。

## 13. 数据流：完整生命周期

以下是将 f32 张量从 CPU 移动到 GPU、相加再返回 CPU 的完整"生命周期"：

```
CPU:  [f32; 1000]  ← Rust Vec<f32>
        │
        │ memcpy_stod() ← cudarc 封装的 cuMemcpyHtoDAsync
        ▼
GPU:  CudaSlice<f32>  ← 包装在 CudaStorageSlice::F32 中
        │
        │ get_or_load_func("badd_f32", &BINARY)  ← 加载 PTX，获取函数句柄
        │ 分配输出 CudaSlice<f32>（unsafe alloc）
        │ builder.arg(...).launch(cfg)  ← cudarc 封装的 cuLaunchKernel
        │ 内核在 GPU 上执行（每个线程处理一个元素 / 网格步进）
        ▼
GPU:  CudaSlice<f32>  ← 结果
        │
        │ memcpy_dtov() ← cudarc 封装的 cuMemcpyDtoHAsync
        ▼
CPU:  Vec<f32>  ← CpuStorage::F32
```

## 14. 关键设计决策总结

1. **Driver API 而非 Runtime API**：`cudarc` 使用 CUDA driver API，PTX 模块在运行时加载。这也意味着 CUDA 工具包不需要在最终用户的机器上安装 —— PTX 由 GPU 驱动即时编译（JIT）为 SASS（机器码）。

2. **编译时 PTX 嵌入**：内核在 `build.rs` 中编译，并通过 `include_str!()` 嵌入到二进制文件中。用户运行 candle 程序时不需要任何 `.ptx` 文件。

3. **延迟模块加载**：PTX 模块在首次调用 `get_or_load_func` 时才加载到 GPU，之后通过 `RwLock` 缓存 —— 每个内核类型（共 11 种）永远只加载一次。

4. **内核名称按类型特化**：命名约定为 `<操作>_<dtype>`（例如 `badd_f32`、`ugelu_f16`）。Rust 侧的 `kernel_name::<T>()` 辅助函数据此构造正确的内核名称。

5. **通过 `Map1`/`Map2` 进行基于特质的类型分发**：这些特质将类型特定的泛型代码与 `CudaStorageSlice` 枚举体的运行时匹配分离，避免了大规模 `match` 语句的代码膨胀。

6. **连续数组快速路径**：每个内核都在运行时检查张量是否物理上连续（行主序）。如果是，使用直接索引；如果不是，则计算步幅索引。`nullptr` info 指针发出"一切连续"的信号。

7. **将重度操作委托给 NVIDIA 库**：矩阵乘法使用 cuBLAS（`cublasGemmStridedBatchedEx`，可选启用 Tensor Core），卷积可选使用 cuDNN。自定义内核仅用于元素级操作、规约和索引操作。
