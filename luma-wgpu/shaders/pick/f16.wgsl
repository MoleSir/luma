enable f16;

// Pick from two buffers according to a bool mask.

struct PickParams {
    n: u32,
    rank: u32,
    mask_start: u32,
    true_start: u32,
    false_start: u32,
    dims_base: u32,
    mask_strides_base: u32,
    true_strides_base: u32,
    false_strides_base: u32,
    mask_contiguous: u32,
    true_contiguous: u32,
    false_contiguous: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> mask: array<u32>;
@group(0) @binding(1) var<storage, read> on_true: array<f16>;
@group(0) @binding(2) var<storage, read> on_false: array<f16>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
@group(0) @binding(4) var<uniform> params: PickParams;
@group(0) @binding(5) var<storage, read> metadata: array<u32>;

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
fn pick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        var mi = params.mask_start + i;
        if (params.mask_contiguous == 0u) {
            mi = elem_offset(i, params.mask_start, params.rank, params.dims_base, params.mask_strides_base);
        }
        var ti = params.true_start + i;
        if (params.true_contiguous == 0u) {
            ti = elem_offset(i, params.true_start, params.rank, params.dims_base, params.true_strides_base);
        }
        var fi = params.false_start + i;
        if (params.false_contiguous == 0u) {
            fi = elem_offset(i, params.false_start, params.rank, params.dims_base, params.false_strides_base);
        }
        let m = mask[mi];
        let tv = on_true[ti];
        let fv = on_false[fi];
        out[i] = select(fv, tv, m != 0u);
    }
}
