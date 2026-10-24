enable f16;

// Gather a strided layout into a contiguous buffer.

struct CopyParams {
    n: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> params: CopyParams;
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
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        var ii = params.start + i;
        if (params.contiguous == 0u) {
            ii = elem_offset(i, params.start, params.rank, params.dims_base, params.strides_base);
        }
        out[i] = inp[ii];
    }
}
