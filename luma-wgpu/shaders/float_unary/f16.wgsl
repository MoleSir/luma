enable f16;

struct Params {
    n: u32,
    op: u32,
    arg0: f32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
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

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7.
fn erf_approx(x: f32) -> f32 {
    let s = sign(x);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
            - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return s * y;
}

fn gelu(x: f32) -> f32 {
    let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

fn gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865476));
}

fn round_away(x: f32) -> f32 {
    if x >= 0.0 { return floor(x + 0.5); }
    return ceil(x - 0.5);
}

fn apply(op: u32, xf: f16, arg0: f32) -> f16 {
    let x = f32(xf);
    switch op {
        case 0u: { return f16(exp(x)); }
        case 1u: { return f16(log(x)); }
        case 2u: { return f16(sin(x)); }
        case 3u: { return f16(cos(x)); }
        case 4u: { return f16(tanh(x)); }
        case 5u: { return f16(x * x); }
        case 6u: { return f16(sqrt(x)); }
        case 7u: { return f16(1.0 / x); }
        case 8u: { return f16(gelu(x)); }
        case 9u: { return f16(gelu_erf(x)); }
        case 10u: { return f16(erf_approx(x)); }
        case 11u: { return f16(max(x, 0.0)); }
        case 12u: { return f16(select(arg0 * x, x, x >= 0.0)); }
        case 13u: { return f16(x * sigmoid(x)); }
        case 14u: { return f16(sigmoid(x)); }
        case 15u: { return f16(floor(x)); }
        case 16u: { return f16(ceil(x)); }
        case 17u: { return f16(round_away(x)); }
        default: { return xf; }
    }
}

@compute @workgroup_size(256)
fn float_unary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        if (params.contiguous != 0u) {
            out[i] = apply(params.op, inp[params.start + i], params.arg0);
        } else {
            let ii = elem_offset(i, params.start, params.rank, params.dims_base, params.strides_base);
            out[i] = apply(params.op, inp[ii], params.arg0);
        }
    }
}
