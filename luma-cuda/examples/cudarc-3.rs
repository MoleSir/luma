use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // ptx 文件转为 module
    let module = ctx.load_module(Ptx::from_file("./exampless/sin.ptx"))?;
    // 选择其中一个函数
    let f = module.load_function("sin_kernel")?;

    // 准备参数
    let a_host = [1.0, 2.0, 3.0];
    let a_dev = stream.clone_htod(&a_host)?;
    let mut b_dev = a_dev.clone();

    let n = 3i32;
    // cfg 得到 grad size 和 block size
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch_args = stream.launch_builder(&f);
    launch_args.arg(&mut b_dev);
    launch_args.arg(&a_dev);
    launch_args.arg(&n);
    unsafe { launch_args.launch(cfg) }?;

    let a_host_2 = stream.clone_dtoh(&a_dev)?;
    let b_host = stream.clone_dtoh(&b_dev)?;

    println!("Found {b_host:?}");
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(&a_host, a_host_2.as_slice());

    Ok(())
}