enable f16;

struct Params {
    n: u32,
    op: u32,
    scalar_bits: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
};

@group(0) @binding(0) var<storage, read> lhs: array<f16>;
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

fn cmp(op: u32, a: f16, b: f16) -> bool {
    switch op {
        case 0u: { return a == b; }
        case 1u: { return a != b; }
        case 2u: { return a <= b; }
        case 3u: { return a >= b; }
        case 4u: { return a < b; }
        case 5u: { return a > b; }
        default: { return false; }
    }
}

@compute @workgroup_size(256)
fn cmp_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        if (params.contiguous != 0u) {
            out[i] = select(0u, 1u, cmp(params.op, lhs[params.start + i], f16(f16(bitcast<f32>(params.scalar_bits)))));
        } else {
            let ii = elem_offset(i, params.start, params.rank, params.dims_base, params.strides_base);
            out[i] = select(0u, 1u, cmp(params.op, lhs[ii], f16(f16(bitcast<f32>(params.scalar_bits)))));
        }
    }
}
