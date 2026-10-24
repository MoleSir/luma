// u32 argmin/argmax over one dim. One workgroup per output element.
// Ties keep the smaller index.

struct ArgReduceParams {
    out_count: u32,
    rank: u32,
    start: u32,
    dim_size: u32,
    dim_stride: u32,
    dim: u32,
    take_max: u32,
    dims_base: u32,
    strides_base: u32,
    mask_base: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> params: ArgReduceParams;
@group(0) @binding(3) var<storage, read> metadata: array<u32>;

const BLOCK_SIZE: u32 = 256u;
const NO_INDEX: u32 = 0xffffffffu;

var<workgroup> scratch_value: array<u32, BLOCK_SIZE>;
var<workgroup> scratch_index: array<u32, BLOCK_SIZE>;

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

fn identity_value() -> u32 {
    if (params.take_max != 0u) {
        return 0u;
    }
    return 0xffffffffu;
}

fn better(value: u32, index: u32, best_value: u32, best_index: u32) -> bool {
    if (best_index == NO_INDEX) {
        return true;
    }
    if (params.take_max != 0u) {
        return value > best_value || (value == best_value && index < best_index);
    }
    return value < best_value || (value == best_value && index < best_index);
}

@compute @workgroup_size(BLOCK_SIZE)
fn arg_reduce(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let out_idx = wid.x;
    let tid = lid.x;
    let base = output_base(out_idx);

    var best_value = identity_value();
    var best_index = NO_INDEX;
    for (var i: u32 = tid; i < params.dim_size; i = i + BLOCK_SIZE) {
        let value = inp[base + i * params.dim_stride];
        if (better(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    scratch_value[tid] = best_value;
    scratch_index[tid] = best_index;
    workgroupBarrier();

    for (var stride: u32 = BLOCK_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            let other_value = scratch_value[tid + stride];
            let other_index = scratch_index[tid + stride];
            if (better(other_value, other_index, scratch_value[tid], scratch_index[tid])) {
                scratch_value[tid] = other_value;
                scratch_index[tid] = other_index;
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let index = scratch_index[0];
        out[out_idx] = select(0u, index, index != NO_INDEX);
    }
}
