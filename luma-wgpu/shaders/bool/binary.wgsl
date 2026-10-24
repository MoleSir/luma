// Boolean and/or/xor over two u32 bool buffers.
// op codes: 0=and, 1=or, 2=xor.

struct BoolBinaryParams {
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

@group(0) @binding(0) var<storage, read> lhs: array<u32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> params: BoolBinaryParams;
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

fn apply(op: u32, a: u32, b: u32) -> u32 {
    let av = select(0u, 1u, a != 0u);
    let bv = select(0u, 1u, b != 0u);
    switch op {
        case 0u: { return av & bv; }
        case 1u: { return av | bv; }
        case 2u: { return av ^ bv; }
        default: { return av; }
    }
}

@compute @workgroup_size(256)
fn bool_binary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        var li = params.lhs_start + i;
        if (params.lhs_contiguous == 0u) {
            li = elem_offset(i, params.lhs_start, params.rank, params.dims_base, params.lhs_strides_base);
        }
        var ri = params.rhs_start + i;
        if (params.rhs_contiguous == 0u) {
            ri = elem_offset(i, params.rhs_start, params.rank, params.dims_base, params.rhs_strides_base);
        }
        out[i] = apply(params.op, lhs[li], rhs[ri]);
    }
}
