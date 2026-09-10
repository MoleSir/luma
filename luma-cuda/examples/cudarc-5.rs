use cudarc::{driver::*, nvrtc::compile_ptx};

#[repr(C)]
struct MyCoolRestStruct {
    a: f32,
    b: f64,
    c: u32,
    d: usize,
}

unsafe impl DeviceRepr for MyCoolRestStruct {}

const PTX_SRC: &str = "
struct MyCoolStruct {
    float a;
    double b;
    unsigned int c;
    size_t d;
};

extern \"C\" __global__ void my_custom_kernel(MyCoolStruct thing) {
    assert(thing.a == 1.0);
    assert(thing.b == 2.34);
    assert(thing.c == 57);
    assert(thing.d == 420);
}
";

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(PTX_SRC).unwrap();
    let module = ctx.load_module(ptx)?;
    let f = module.load_function("my_custom_kernel")?;

    let thing = MyCoolRestStruct {
        a: 1.0,
        b: 2.34,
        c: 57,
        d: 420,
    };

    let mut builder = stream.launch_builder(&f);
    builder.arg(&thing);
    unsafe { builder.launch(LaunchConfig::for_num_elems(1)) }?;

    Ok(())
}