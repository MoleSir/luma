use cudarc::driver::CudaContext;
use cudarc::group;
use cudarc::nccl::Comm;
use cudarc::nccl::result;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N_DEVICE: usize = 2;
    const N: usize = 4;

    let devices: Vec<_> = (0..N_DEVICE).map(|i| CudaContext::new(i).unwrap()).collect();
    let streams: Vec<_> = devices.iter().map(|d| d.default_stream()).collect();
    let comms = Comm::from_devices(streams.clone()).unwrap();

    let mut sendbuffs: Vec<_> = streams.iter().map(|s| s.alloc_zeros::<f32>(N).unwrap()).collect();

    let mut recvbuffs: Vec<_> = streams.iter().map(|s| s.alloc_zeros::<f32>(N).unwrap()).collect();

    // GPU 0 的输入数据
    let input = vec![10.0f32, 20.0, 30.0, 40.0];
    streams[0].memcpy_htod(&input, &mut sendbuffs[0])?;

    // GPU 0 发送，GPU 1 接收
    group!({
        comms[0].send(&sendbuffs[0], 1).unwrap();
        comms[1].recv(&mut recvbuffs[1], 0).unwrap();
    });

    // 等待接收完成并读取 GPU 1 的结果
    streams[1].synchronize()?;

    let mut output = vec![0.0f32; N];
    streams[1].memcpy_dtoh(&recvbuffs[1], &mut output)?;

    println!("GPU 1 received: {:?}", output);

    Ok(())
}
