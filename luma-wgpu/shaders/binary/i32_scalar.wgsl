struct Params {
    n: u32,
    op: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    scalar_is_lhs: u32,
    scalar_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<i32>;
@group(0) @binding(1) var<storage, read_write> out: array<i32>;
@group(0) @binding(2) var<uniform> params: Params;

@group(0) @binding(3) var<storage, read> metadata: array<u32>;

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

fn apply(op: u32, a: i32, b: i32) -> i32 {
    switch op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return max(a, b); }
        case 5u: { return min(a, b); }
        default: { return a; }
    }
}

@compute @workgroup_size(256)
fn binary_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        var x: i32;
        if (params.contiguous != 0u) {
            x = inp[params.start + i];
        } else {
            x = inp[elem_offset(i, params.start, params.rank, params.dims_base, params.strides_base)];
        }

        let s = bitcast<i32>(params.scalar_bits);
        if (params.scalar_is_lhs != 0u) {
            out[i] = apply(params.op, s, x);
        } else {
            out[i] = apply(params.op, x, s);
        }
    }
}
