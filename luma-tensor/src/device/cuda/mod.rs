mod device;
mod error;
mod launch;
#[allow(unused)]
mod ops;
mod storage;
pub use device::Cuda;
pub use error::*;
use luma_cuda_kernel as kernel;
pub use storage::*;

// // ---------------------------------------------------------------------------
// // Cuda: constructors & kernel helpers
// // ---------------------------------------------------------------------------

// impl Cuda {
//     pub fn new(ordinal: usize) -> Result<Self> {
//         let context = cudarc::driver::CudaContext::new(ordinal)
//             .map_err(|e| Error::Cuda(e.to_string()))?;
//         let stream = context.default_stream();
//         let blas = cudarc::cublas::CudaBlas::new(stream.clone())
//             .map_err(|e| Error::Cuda(e.to_string()))?;
//         let curand = cudarc::curand::CudaRng::new(299_792_458, stream.clone())
//             .map_err(|e| Error::Cuda(e.to_string()))?;

//         Ok(Cuda {
//             inner: Arc::new(CudaInner {
//                 ordinal,
//                 context,
//                 stream,
//                 modules: RwLock::new(Vec::new()),
//                 blas: Arc::new(blas),
//                 curand: std::sync::Mutex::new(curand),
//             }),
//         })
//     }

//     /// Synchronise the default stream (blocking).
//     pub fn synchronize(&self) -> Result<()> {
//         self.inner
//             .stream
//             .synchronize()
//             .map_err(|e| Error::Cuda(e.to_string()))
//     }

//     // ---- memory allocation ----

//     /// Allocate `n` elements of type `T` on the device (uninitialized).
//     pub(crate) unsafe fn alloc<T: cudarc::driver::DeviceRepr>(&self, n: usize) -> Result<CudaSlice<T>> {
//         self.inner
//             .stream
//             .alloc::<T>(n)
//             .map_err(|e| Error::Cuda(e.to_string()))
//     }

//     /// Host → device memcpy (async, on default stream). Allocates and copies.
//     pub(crate) fn memcpy_htod<T: cudarc::driver::DeviceRepr>(&self, src: &[T]) -> Result<CudaSlice<T>> {
//         let mut dst = unsafe { self.alloc::<T>(src.len()) }?;
//         self.inner
//             .stream
//             .memcpy_htod(src, &mut dst)
//             .map_err(|e| Error::Cuda(e.to_string()))?;
//         Ok(dst)
//     }

//     /// Device → host memcpy (async, on default stream).
//     pub(crate) fn memcpy_dtov<T: cudarc::driver::DeviceRepr>(&self, src: &CudaSlice<T>) -> Result<Vec<T>> {
//         self.inner
//             .stream
//             .memcpy_dtov(src)
//             .map_err(|e| Error::Cuda(e.to_string()))
//     }

//     // ---- kernel loading ----

//     /// Load or retrieve a cached [`CudaFunction`] by kernel name and module.
//     pub(crate) fn get_or_load_func(&self, kernel_name: &str, module: &kernels::Module) -> Result<CudaFunction> {
//         let idx = module.index;
//         {
//             let modules = self.inner.modules.read().unwrap();
//             if idx < modules.len() {
//                 if let Some(ref cached) = modules[idx] {
//                     return cached
//                         .load_function(kernel_name)
//                         .map_err(|e| Error::Cuda(e.to_string()));
//                 }
//             }
//         }
//         {
//             let mut modules = self.inner.modules.write().unwrap();
//             while modules.len() <= idx {
//                 modules.push(None);
//             }
//             if modules[idx].is_none() {
//                 let ptx = cudarc::nvrtc::Ptx::from(module.ptx.to_string());
//                 let cuda_module = self
//                     .inner
//                     .context
//                     .load_module(ptx)
//                     .map_err(|e| Error::Cuda(e.to_string()))?;
//                 modules[idx] = Some(cuda_module);
//             }
//             modules[idx]
//                 .as_ref()
//                 .unwrap()
//                 .load_function(kernel_name)
//                 .map_err(|e| Error::Cuda(e.to_string()))
//         }
//     }

//     pub(crate) fn launch_config(&self, num_elems: usize) -> LaunchConfig {
//         LaunchConfig::for_num_elems(num_elems as u32)
//     }

//     /// Materialise a potentially non-contiguous tensor view into a contiguous
//     /// `CudaSlice<T>` (GPU-side copy via host round-trip, or D2D copy).
//     pub(crate) fn gather_to_contiguous<T: cudarc::driver::DeviceRepr + Copy>(
//         &self,
//         src: &CudaSlice<T>,
//         layout: &crate::Layout,
//     ) -> Result<CudaSlice<T>> {
//         if layout.is_contiguous() && layout.start_offset() == 0 {
//             return Ok(src.clone());
//         }
//         let v = self.memcpy_dtov(src)?;
//         let gathered: Vec<T> = layout.storage_indices().map(|i| v[i]).collect();
//         self.memcpy_htod(&gathered)
//     }

//     /// Convenience: get function by name, then build & launch with custom args.
//     pub(crate) fn launch_kernel(
//         &self,
//         module: &kernels::Module,
//         kernel_name: &str,
//         numel: usize,
//         push_args: impl FnOnce(&mut cudarc::driver::LaunchArgs<'_>),
//     ) -> Result<()> {
//         use cudarc::driver::PushKernelArg;
//         let func = self.get_or_load_func(kernel_name, module)?;
//         let cfg = self.launch_config(numel);
//         let mut builder = self.inner.stream.launch_builder(&func);
//         push_args(&mut builder);
//         unsafe { builder.launch(cfg) }
//             .map_err(|e: cudarc::driver::result::DriverError| Error::Cuda(e.to_string()))?;
//         Ok(())
//     }
// }

// impl Device for Cuda {
//     type FloatStorage = CudaFloatStorage;
//     type IntStorage = CudaIntStorage;
//     type BoolStorage = CudaBoolStorage;

//     fn name(&self) -> String {
//         format!("cuda:{}", self.inner.ordinal)
//     }
// }

// impl From<CudaSlice<f32>> for CudaFloatSlice { fn from(s: CudaSlice<f32>) -> Self { Self::F32(s) } }
// impl From<CudaSlice<f64>> for CudaFloatSlice { fn from(s: CudaSlice<f64>) -> Self { Self::F64(s) } }
// impl From<CudaSlice<i32>> for CudaIntSlice { fn from(s: CudaSlice<i32>) -> Self { Self::I32(s) } }
// impl From<CudaSlice<u32>> for CudaIntSlice { fn from(s: CudaSlice<u32>) -> Self { Self::U32(s) } }
// impl From<CudaSlice<u8>> for CudaIntSlice { fn from(s: CudaSlice<u8>) -> Self { Self::U8(s) } }

// impl Default for Cuda {
//     fn default() -> Self {
//         Cuda::new(0).expect("Failed to create default CUDA device (GPU 0)")
//     }
// }
