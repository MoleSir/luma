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

@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
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

fn pow_int(base: u32, exp: u32) -> u32 {
    var result: u32 = 1;
    var b = base;
    var e = exp;
    if (e < 0) {
        return 0;
    }
    loop {
        if (e == 0) { break; }
        if ((e & 1) != 0) {
            result = result * b;
        }
        b = b * b;
        e = e >> 1;
    }
    return result;
}

fn apply(op: u32, x: u32, arg0: u32, arg1: u32, arg0_opt: u32, arg1_opt: u32) -> u32 {
    switch op {
        case 0u: { return 0u - x; }
        case 1u: { return x; }
        case 2u: { return select(0u, 1u, x != 0u); }
        case 3u: { return arg0 * x + arg1; }
        case 4u: { return pow_int(x, arg0); }
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
        let a0 = params.arg0_bits;
        let a1 = params.arg1_bits;
        if (params.contiguous != 0u) {
            out[i] = apply(params.op, inp[params.start + i], a0, a1, params.arg0_opt, params.arg1_opt);
        } else {
            let ii = elem_offset(i, params.start, params.rank, params.dims_base, params.strides_base);
            out[i] = apply(params.op, inp[ii], a0, a1, params.arg0_opt, params.arg1_opt);
        }
    }
}
