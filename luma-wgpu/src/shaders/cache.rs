use crate::shaders::Shader;
use crate::{ShaderKind, Wgpu, WgpuResult};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Per-device cache of compiled shaders.
#[derive(Default)]
pub struct PipelineCache {
    shaders: RwLock<HashMap<ShaderKind, Arc<Shader>>>,
}

impl PipelineCache {
    /// Get a cached shader, compiling it on first use.
    pub fn get(&self, device: &Wgpu, key: ShaderKind) -> WgpuResult<Arc<Shader>> {
        if let Some(shader) = self.shaders.read().expect("pipeline cache poisoned").get(&key) {
            return Ok(shader.clone());
        }

        let shader = Arc::new(super::build_shader(device, key)?);
        let mut shaders = self.shaders.write().expect("pipeline cache poisoned");
        Ok(shaders.entry(key).or_insert(shader).clone())
    }
}
