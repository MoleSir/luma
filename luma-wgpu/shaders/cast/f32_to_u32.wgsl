// Cast from f32 to u32.

struct CastParams {
    n: u32,
    rank: u32,
    start: u32,
    dims_base: u32,
    strides_base: u32,
    contiguous: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> params: CastParams;
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

fn to_u32(x: f32) -> u32 {
    if ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u) { return 0u; }
    if (x <= 0.0) { return 0u; }
    if (x >= 4294967296.0) { return 0xffffffffu; }
    return u32(x);
}

@compute @workgroup_size(256)
fn cast_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        var ii = params.start + i;
        if (params.contiguous == 0u) {
            ii = elem_offset(i, params.start, params.rank, params.dims_base, params.strides_base);
        }
        let x = inp[ii];
        out[i] = to_u32(x);
    }
}
