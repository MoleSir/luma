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

fn apply(op: u32, x: f32, arg0: f32) -> f32 {
    switch op {
        case 0u: { return exp(x); }
        case 1u: { return log(x); }
        case 2u: { return sin(x); }
        case 3u: { return cos(x); }
        case 4u: { return tanh(x); }
        case 5u: { return x * x; }
        case 6u: { return sqrt(x); }
        case 7u: { return 1.0 / x; }
        case 8u: { return gelu(x); }
        case 9u: { return gelu_erf(x); }
        case 10u: { return erf_approx(x); }
        case 11u: { return max(x, 0.0); }
        case 12u: { return select(arg0 * x, x, x >= 0.0); }
        case 13u: { return x * sigmoid(x); }
        case 14u: { return sigmoid(x); }
        case 15u: { return floor(x); }
        case 16u: { return ceil(x); }
        case 17u: { return round_away(x); }
        default: { return x; }
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
