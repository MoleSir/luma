struct Params {
    n: u32,
    op: u32,
    rank: u32,
    lhs_start: u32,
    rhs_start: u32,
    dims_base: u32,
    lhs_strides_base: u32,
    rhs_strides_base: u32,
    lhs_contiguous: u32,
    rhs_contiguous: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> lhs: array<i32>;
@group(0) @binding(1) var<storage, read> rhs: array<i32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

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

fn cmp(op: u32, a: i32, b: i32) -> bool {
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
fn cmp_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        if (params.lhs_contiguous != 0u) {
            if (params.rhs_contiguous != 0u) {
                out[i] = select(0u, 1u, cmp(params.op, lhs[params.lhs_start + i], rhs[params.rhs_start + i]));
            } else {
                let ri = elem_offset(i, params.rhs_start, params.rank, params.dims_base, params.rhs_strides_base);
                out[i] = select(0u, 1u, cmp(params.op, lhs[params.lhs_start + i], rhs[ri]));
            }
        } else {
            let li = elem_offset(i, params.lhs_start, params.rank, params.dims_base, params.lhs_strides_base);
            if (params.rhs_contiguous != 0u) {
                out[i] = select(0u, 1u, cmp(params.op, lhs[li], rhs[params.rhs_start + i]));
            } else {
                let ri = elem_offset(i, params.rhs_start, params.rank, params.dims_base, params.rhs_strides_base);
                out[i] = select(0u, 1u, cmp(params.op, lhs[li], rhs[ri]));
            }
        }
    }
}
