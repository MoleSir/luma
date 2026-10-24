struct Params {
    n: u32,
    op: u32,
    arg0_bits: u32,
    arg1_bits: u32,
    arg0_opt: u32,
    arg1_opt: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@group(0) @binding(4) var<storage, read> metadata: array<u32>;

fn elem_offset(i: u32, start: u32, rank: u32, dims_base: u32, strides_base: u32) -> u32 {
    var rem = i;
    var off = start;
    for (var d: u32 = rank; d > 0u; d = d - 1u) {
        let idx = d - 1u;
        let dim = metadata[dims_base + idx];
        let c = rem % dim;
        rem = rem / dim;
        off += c * metadata[strides_base + idx];
    }
    return off;
}

fn signum(x: f32) -> f32 {
    if x > 0.0 { return 1.0; }
    if x < 0.0 { return -1.0; }
    return select(1.0, -1.0, (bitcast<u32>(x) & 0x80000000u) != 0u);
}

fn pow_signed(x: f32, e: f32) -> f32 {
    // WGSL pow(0, 0) may produce NaN; Rust's powf(0, 0) is 1.
    if e == 0.0 {
        return 1.0;
    }
    if x >= 0.0 {
        return pow(x, e);
    }
    let ei = select(floor(e + 0.5), ceil(e - 0.5), e < 0.0);
    if abs(e - ei) > 1e-6 {
        return bitcast<f32>(0x7fc00000u);
    }
    let odd = (ei - 2.0 * floor(ei * 0.5)) != 0.0;
    let mag = pow(-x, e);
    return select(mag, -mag, odd);
}

fn apply(op: u32, x: f32, arg0: f32, arg1: f32, arg0_opt: u32, arg1_opt: u32) -> f32 {
    switch op {
        case 0u: { return -x; }
        case 1u: { return abs(x); }
        case 2u: { return signum(x); }
        case 3u: { return arg0 * x + arg1; }
        case 4u: { return pow_signed(x, arg0); }
        case 5u: {
            var v = x;
            if arg0_opt != 0 {
                v = max(arg0, v);
            }
            if arg1_opt != 0 {
                v = min(arg1, v);
            }
            return v;
        }
        default: { return x; }
    }
}

@compute @workgroup_size(256)
fn unary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        if (params.contiguous != 0u) {
            out[i] = apply(params.op, inp[params.start + i], bitcast<f32>(params.arg0_bits), bitcast<f32>(params.arg1_bits), params.arg0_opt, params.arg1_opt);
        } else {
            let ii = elem_offset(i, params.start, params.rank, params.dims_base, params.strides_base);
            out[i] = apply(params.op, inp[ii], bitcast<f32>(params.arg0_bits), bitcast<f32>(params.arg1_bits), params.arg0_opt, params.arg1_opt);
        }
    }
}
