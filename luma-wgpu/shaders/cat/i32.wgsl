// Copy one source's slice into its position in a concatenated output.

struct CatParams {
    n: u32,
    rank: u32,
    dim: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    dst_dim_base: u32,
    cat_size: u32,
    right: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<i32>;
@group(0) @binding(1) var<storage, read_write> out: array<i32>;
@group(0) @binding(2) var<uniform> params: CatParams;
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

@compute @workgroup_size(256)
fn cat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        var ii = params.start + i;
        if (params.contiguous == 0u) {
            ii = elem_offset(i, params.start, params.rank, params.dims_base, params.strides_base);
        }
        let src_dim = metadata[params.dims_base + params.dim];
        let block = src_dim * params.right;
        let outer = i / block;
        let rem = i % block;
        let dst = (outer * params.cat_size + params.dst_dim_base) * params.right + rem;
        out[dst] = inp[ii];
    }
}
