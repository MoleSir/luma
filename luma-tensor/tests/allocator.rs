//! 池化 allocator 的端到端测试：pooled Cpu 实例必须贯通整条计算链。

use std::sync::{Arc, RwLock};

use luma_tensor::device::cpu::allocator::{CpuAllocator, PoolAllocator};
use luma_tensor::dtype::FloatDType;
use luma_tensor::{Cpu, Tensor};

/// 读共享句柄里的池计数。
fn allocs(shared: &Arc<RwLock<dyn CpuAllocator>>) -> usize {
    let g = shared.read().unwrap();
    let pool = g.as_any().downcast_ref::<PoolAllocator>().expect("shared allocator is a pool");
    pool.system_allocs()
}

/// 池化实例贯通：输入 → op 输出 全程携带同一个 allocator；
/// 释放后同尺寸分配命中池（系统分配计数不增）。
#[test]
fn pooled_instance_threads_through_chain() {
    let shared: Arc<RwLock<dyn CpuAllocator>> = Arc::new(RwLock::new(PoolAllocator::new()));
    let cpu = Cpu::with_allocator_shared(shared.clone());
    let opts = (&cpu, FloatDType::F32);

    // 构造两个输入（各一次系统分配）
    let x = Tensor::<Cpu>::from_slice(&[1.0; 64], (8, 8), opts).unwrap();
    let y = Tensor::<Cpu>::from_slice(&[2.0; 64], (8, 8), opts).unwrap();
    assert_eq!(allocs(&shared), 2, "两个输入各分配一次");

    // op 输出携带同一实例（Arc::ptr_eq 验证）
    let z = x.add(&y).unwrap();
    assert!(Arc::ptr_eq(&shared, z.device().allocator()), "op 输出必须继承输入设备实例的 allocator");

    // z 的计算本身又分配一次
    assert_eq!(allocs(&shared), 3);

    // 释放 z → Storage Drop → dealloc → 回池
    drop(z);
    // 同尺寸再算一次 → 命中池，计数不增
    let z2 = x.add(&y).unwrap();
    assert_eq!(allocs(&shared), 3, "释放后的同尺寸分配应命中池");
    drop(z2);
}
