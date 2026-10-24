use cudarc::driver::CudaContext;
use cudarc::group;
use cudarc::nccl::result;
use cudarc::nccl::{Comm, ReduceOp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N_DEVICE: usize = 3;
    const N_SIZE: usize = 5;

    // 获取每个 GPU 的 context
    let devices: Vec<_> = (0..N_DEVICE).map(|i| CudaContext::new(i).unwrap()).collect();
    // 获取默认 stream
    let streams: Vec<_> = devices.iter().map(|d| d.default_stream()).collect();
    // 从 stream 申请 comm 对象，通过这个 stream 提交任务
    let comms = Comm::from_devices(streams.clone()).unwrap();
    // 对每个 gpu 申请发送、接受 buffs
    let mut sendbuffs: Vec<_> = streams.iter().map(|s| s.alloc_zeros::<f32>(N_SIZE).unwrap()).collect();
    let mut recvbuffs: Vec<_> = streams.iter().map(|s| s.alloc_zeros::<f32>(N_SIZE).unwrap()).collect();

    // gpu i 的 发送值 为 i
    // gpu 0: [0, 0, 0, 0, 0]
    // gpu 1: [1, 1, 1, 1, 1]
    // gpu 2: [2, 2, 2, 2, 2]
    // =>     [3, 3, 3, 3, 3]
    for (i, (stream, sendbuff)) in streams.iter().zip(sendbuffs.iter_mut()).enumerate() {
        stream.memcpy_htod(&[i as f32; N_SIZE], sendbuff)?;
    }

    group!({
        // 每个 gpu 放置自己的输入输出 buff
        for (comm, (sendbuff, recvbuff)) in comms.iter().zip(sendbuffs.iter().zip(recvbuffs.iter_mut())) {
            comm.all_reduce(sendbuff, recvbuff, &ReduceOp::Sum).unwrap();
        }
    });

    // 检查结果
    for (i, (stream, recvbuff)) in streams.iter().zip(recvbuffs.iter()).enumerate() {
        stream.synchronize()?;
        let mut values = vec![0.0f32; N_SIZE];
        stream.memcpy_dtoh(recvbuff, &mut values)?;
        println!("GPU {}: result = {:?}", i, values);
    }

    Ok(())
}
