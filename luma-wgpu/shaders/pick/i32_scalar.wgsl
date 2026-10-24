// Pick between a scalar and a buffer according to a bool mask.

struct PickScalarParams {
    n: u32,
    rank: u32,
    mask_start: u32,
    other_start: u32,
    dims_base: u32,
    mask_strides_base: u32,
    other_strides_base: u32,
    mask_contiguous: u32,
    other_contiguous: u32,
    scalar_bits: u32,
    scalar_is_true: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
};

@group(0) @binding(0) var<storage, read> mask: array<u32>;
@group(0) @binding(1) var<storage, read> other: array<i32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;
@group(0) @binding(3) var<uniform> params: PickScalarParams;
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
fn pick_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        var mi = params.mask_start + i;
        if (params.mask_contiguous == 0u) {
            mi = elem_offset(i, params.mask_start, params.rank, params.dims_base, params.mask_strides_base);
        }
        var oi = params.other_start + i;
        if (params.other_contiguous == 0u) {
            oi = elem_offset(i, params.other_start, params.rank, params.dims_base, params.other_strides_base);
        }
        let m = mask[mi] != 0u;
        let ov = other[oi];
        let sv = bitcast<i32>(params.scalar_bits);
        var v = ov;
        if (params.scalar_is_true != 0u) {
            v = select(ov, sv, m);
        } else {
            v = select(sv, ov, m);
        }
        out[i] = v;
    }
}
