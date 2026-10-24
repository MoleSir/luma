use cudarc::driver::CudaContext;
use cudarc::group;
use cudarc::nccl::Comm;
use cudarc::nccl::result;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N_DEVICE: usize = 3;
    const N_SEND_BUFF: usize = 2;
    const N_RECV_BUFF: usize = N_DEVICE * N_SEND_BUFF;

    // 获取每个 GPU 的 context 与对应的 stream
    let devices: Vec<_> = (0..N_DEVICE).map(|i| CudaContext::new(i).unwrap()).collect();
    let streams: Vec<_> = devices.iter().map(|d| d.default_stream()).collect();
    // 从 stream 申请 comm 对象，通过这个 stream 提交任务
    let comms = Comm::from_devices(streams.clone()).unwrap();

    // 对每个 gpu 申请发送、接受 buffs
    let mut sendbuffs: Vec<_> = streams.iter().map(|s| s.alloc_zeros::<f32>(N_SEND_BUFF).unwrap()).collect();
    let mut recvbuffs: Vec<_> = streams.iter().map(|s| s.alloc_zeros::<f32>(N_RECV_BUFF).unwrap()).collect();

    // gpu i 的 发送值 为 vec![i + 0.5]
    for (i, (stream, sendbuff)) in streams.iter().zip(sendbuffs.iter_mut()).enumerate() {
        stream.memcpy_htod(&[i as f32 + 0.5; N_SEND_BUFF], sendbuff)?;
    }

    group!({
        for (comm, (sendbuff, recvbuff)) in comms.iter().zip(sendbuffs.iter().zip(recvbuffs.iter_mut())) {
            comm.all_gather(sendbuff, recvbuff).unwrap();
        }
    });

    // 检查结果
    for (i, (stream, recvbuff)) in streams.iter().zip(recvbuffs.iter()).enumerate() {
        stream.synchronize()?;
        let mut values = vec![0.0f32; N_RECV_BUFF];
        stream.memcpy_dtoh(recvbuff, &mut values)?;
        println!("GPU {}: result = {:?}", i, values);
    }

    Ok(())
}
