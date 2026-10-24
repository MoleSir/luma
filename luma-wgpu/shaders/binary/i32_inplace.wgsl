struct Params {
    n: u32,
    op: u32,
    rank: u32,
    dst_start: u32,
    src_start: u32,
    dims_base: u32,
    dst_strides_base: u32,
    src_strides_base: u32,
    dst_contiguous: u32,
    src_contiguous: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> dst: array<i32>;
@group(0) @binding(1) var<storage, read> src: array<i32>;
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
fn binary_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        if (params.dst_contiguous != 0u) {
            if (params.src_contiguous != 0u) {
                let di = params.dst_start + i;
                dst[di] = apply(params.op, dst[di], src[params.src_start + i]);
            } else {
                let di = params.dst_start + i;
                let si = elem_offset(i, params.src_start, params.rank, params.dims_base, params.src_strides_base);
                dst[di] = apply(params.op, dst[di], src[si]);
            }
        } else {
            let di = elem_offset(i, params.dst_start, params.rank, params.dims_base, params.dst_strides_base);
            if (params.src_contiguous != 0u) {
                dst[di] = apply(params.op, dst[di], src[params.src_start + i]);
            } else {
                let si = elem_offset(i, params.src_start, params.rank, params.dims_base, params.src_strides_base);
                dst[di] = apply(params.op, dst[di], src[si]);
            }
        }
    }
}
