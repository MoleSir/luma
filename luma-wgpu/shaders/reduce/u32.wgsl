// u32 reduction: one workgroup per output element.
//
// Binding and metadata layout match reduce/f32.wgsl. `mean` is rejected by the
// host for int types, so the shader only implements sum/min/max/prod.

const OP_SUM:  u32 = 0;
const OP_MIN:  u32 = 1;
const OP_MAX:  u32 = 2;
const OP_MEAN: u32 = 3;
const OP_PROD: u32 = 4;

struct ReduceParams {
    out_count: u32,
    rank: u32,
    start: u32,
    red_count: u32,
    red_total: u32,
    op: u32,
    dims_base: u32,
    strides_base: u32,
    mask_base: u32,
    red_dims_base: u32,
    red_strides_base: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> params: ReduceParams;
@group(0) @binding(3) var<storage, read> metadata: array<u32>;

const BLOCK_SIZE: u32 = 256u;

var<workgroup> scratch: array<u32, BLOCK_SIZE>;

fn output_base(out_idx: u32) -> u32 {
    var rem = out_idx;
    var off = params.start;

    for (var d: u32 = params.rank; d > 0u; d = d - 1u) {
        let idx = d - 1u;
        if (metadata[params.mask_base + idx] == 0u) {
            let dim = metadata[params.dims_base + idx];
            off += (rem % dim) * metadata[params.strides_base + idx];
            rem = rem / dim;
        }
    }

    return off;
}

fn reduced_offset(i: u32) -> u32 {
    var rem = i;
    var off = 0u;

    for (var k: u32 = params.red_count; k > 0u; k = k - 1u) {
        let idx = k - 1u;
        let dim = metadata[params.red_dims_base + idx];
        off += (rem % dim) * metadata[params.red_strides_base + idx];
        rem = rem / dim;
    }

    return off;
}

fn identity(op: u32) -> u32 {
    switch op {
        case OP_SUM:  { return 0u; }
        case OP_MIN:  { return 0xffffffffu; }
        case OP_MAX:  { return 0u; }
        case OP_MEAN: { return 0u; }
        case OP_PROD: { return 1u; }
        default: { return 0u; }
    }
}

fn combine(acc: u32, value: u32) -> u32 {
    switch params.op {
        case OP_SUM:  { return acc + value; }
        case OP_MIN:  { return min(acc, value); }
        case OP_MAX:  { return max(acc, value); }
        case OP_MEAN: { return acc + value; }
        case OP_PROD: { return acc * value; }
        default: { return acc; }
    }
}

@compute @workgroup_size(BLOCK_SIZE)
fn reduce(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let out_idx = wid.x;
    let tid = lid.x;
    let base = output_base(out_idx);

    var acc = identity(params.op);
    for (var i: u32 = tid; i < params.red_total; i = i + BLOCK_SIZE) {
        acc = combine(acc, inp[base + reduced_offset(i)]);
    }

    scratch[tid] = acc;
    workgroupBarrier();

    for (var stride: u32 = BLOCK_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            scratch[tid] = combine(scratch[tid], scratch[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        out[out_idx] = scratch[0];
    }
}
