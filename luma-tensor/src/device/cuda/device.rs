use cudarc::{cublas::CudaBlas, curand::CudaRng, driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, HostSlice, LaunchArgs, ValidAsZeroBits}, nvrtc::Ptx};
use std::{collections::HashMap, sync::{Arc, Mutex, RwLock}};
use super::kernel;
use crate::{Device, device::cuda::{CudaBoolStorage, CudaError, CudaFloatStorage, CudaIntStorage, CudaResult}};

#[derive(Clone)]
pub struct Cuda(pub(crate) Arc<CudaImpl>);

pub struct CudaImpl {
    pub(crate) ordinal: usize,
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) curand: Arc<Mutex<CudaRng>>,
    pub(crate) modules: RwLock<HashMap<&'static str, Arc<CudaModule>>>,
    pub(crate) blas: Mutex<CudaBlas>,
}

pub struct CudaBindedFunction {
    pub(crate) function: CudaFunction,
    pub(crate) stream: Arc<CudaStream>,
}

impl Cuda {
    pub fn new(ordinal: usize) -> CudaResult<Self> {
        let context = CudaContext::new(ordinal)?;
        let stream = context.default_stream();
        let curand = CudaRng::new(299792458, stream.clone())?;
        let blas = CudaBlas::new(stream.clone())?;
        Ok(Cuda(Arc::new(CudaImpl { 
            ordinal, 
            context, 
            stream, 
            curand: Arc::new(Mutex::new(curand)),
            modules: RwLock::new(HashMap::new()),
            blas: Mutex::new(blas),
        })))
    }

    pub fn same_ordinal(&self, other: &Self, op: impl ToString) -> CudaResult<()> {
        if self.0.ordinal != other.0.ordinal {
            Err(CudaError::DiffCudaInBinary(self.name(), other.name(), op.to_string()))?;
        }
        Ok(())
    }

    pub fn ordinal(&self) -> usize {
        self.0.ordinal
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        self.0.stream.synchronize()?;
        Ok(())
    }

    pub fn alloc<T: DeviceRepr>(&self, n: usize) -> CudaResult<CudaSlice<T>> {
        let slice = unsafe { self.0.stream.alloc::<T>(n)? };
        Ok(slice)
    }

    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, n: usize) -> CudaResult<CudaSlice<T>> {
        let slice = self.0.stream.alloc_zeros::<T>(n)?;
        Ok(slice)
    }

    pub fn memcpy_stod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(&self, src: &Src) -> CudaResult<CudaSlice<T>> {
        let slice = self.0.stream.clone_htod(src)?;
        Ok(slice)
    }

    pub fn memcpy_dtov<T: DeviceRepr>(&self, src: &CudaSlice<T>) -> CudaResult<Vec<T>> {
        let v = self.0.stream.clone_dtoh(src)?;
        Ok(v)
    }

    pub fn load_function(&self, kernel_name: &str, module: &kernel::Module) -> CudaResult<CudaBindedFunction> {
        {
            let modules = self.0.modules.read().expect("read modules");
            if let Some(ref cached) = modules.get(module.name()) {
                let f = cached.load_function(kernel_name)?;
                return Ok( CudaBindedFunction { function: f, stream: self.0.stream.clone() } );
            }
        }
        {
            let mut modules = self.0.modules.write().expect("read modules");
            let ptx = Ptx::from(module.ptx().to_string());
            let cuda_module = self.0.context.load_module(ptx)?;
            let f = cuda_module.load_function(kernel_name)?;
            modules.insert(module.name(), cuda_module);
            Ok( CudaBindedFunction { function: f, stream: self.0.stream.clone() } )
        }
    }
}

impl CudaBindedFunction {
    pub fn builder(&self) -> LaunchArgs<'_> {
        self.stream.launch_builder(&self.function)
    }
}

impl Device for Cuda {
    type BoolStorage = CudaBoolStorage;
    type FloatStorage = CudaFloatStorage;
    type IntStorage = CudaIntStorage;
    
    fn name(&self) -> String {
        format!("cuda:{}", self.0.ordinal)
    }
}

impl Default for Cuda {
    fn default() -> Self {
        Cuda::new(0).expect("Failed to create default CUDA device (GPU 0)")
    }
}

unsafe impl Sync for Cuda {}
unsafe impl Send for Cuda {}