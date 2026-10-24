enable f16;

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

@group(0) @binding(0) var<storage, read_write> buf: array<f16>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> metadata: array<u32>;

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

fn signum(x: f16) -> f16 {
    let xf = f32(x);
    if xf > 0.0 { return f16(1.0); }
    if xf < 0.0 { return f16(-1.0); }
    return select(f16(1.0), f16(-1.0), (bitcast<u32>(xf) & 0x80000000u) != 0u);
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

fn apply(op: u32, x: f16, arg0: f32, arg1: f32, arg0_opt: u32, arg1_opt: u32) -> f16 {
    let a0 = f16(arg0);
    let a1 = f16(arg1);
    switch op {
        case 0u: { return -x; }
        case 1u: { return abs(x); }
        case 2u: { return signum(x); }
        case 3u: { return a0 * x + a1; }
        case 4u: { return f16(pow_signed(f32(x), arg0)); }
        case 5u: {
            var v = x;
            if arg0_opt != 0 {
                v = max(a0, v);
            }
            if arg1_opt != 0 {
                v = min(a1, v);
            }
            return v;
        }
        default: { return x; }
    }
}

@compute @workgroup_size(256)
fn unary_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        if (params.contiguous != 0u) {
            let ii = params.start + i;
            buf[ii] = apply(params.op, buf[ii], bitcast<f32>(params.arg0_bits), bitcast<f32>(params.arg1_bits), params.arg0_opt, params.arg1_opt);
        } else {
            let ii = elem_offset(i, params.start, params.rank, params.dims_base, params.strides_base);
            buf[ii] = apply(params.op, buf[ii], bitcast<f32>(params.arg0_bits), bitcast<f32>(params.arg1_bits), params.arg0_opt, params.arg1_opt);
        }
    }
}
