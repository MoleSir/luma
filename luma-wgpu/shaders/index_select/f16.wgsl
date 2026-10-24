enable f16;

// Select slices along `dim` using a 1-D index tensor.

struct IndexSelectParams {
    n: u32,
    rank: u32,
    src_start: u32,
    src_dims_base: u32,
    src_strides_base: u32,
    src_contiguous: u32,
    ids_start: u32,
    ids_stride: u32,
    src_dim: u32,
    ids_count: u32,
    right: u32,
    pad: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;
@group(0) @binding(3) var<uniform> params: IndexSelectParams;
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

@compute @workgroup_size(256)
fn index_select(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        let id_i = (i / params.right) % params.ids_count;
        let left_i = i / (params.ids_count * params.right);
        let right_i = i % params.right;
        let id = ids[params.ids_start + id_i * params.ids_stride];
        if (id == params.pad) {
            out[i] = f16(0.0);
        } else {
            let src_i = (left_i * params.src_dim + id) * params.right + right_i;
            var ii = params.src_start + src_i;
            if (params.src_contiguous == 0u) {
                ii = elem_offset(src_i, params.src_start, params.rank, params.src_dims_base, params.src_strides_base);
            }
            out[i] = inp[ii];
        }
    }
}
