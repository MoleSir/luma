// Workgroup reduction: one workgroup per output element.
//
// Each thread accumulates a strided subset of the reduced elements, then the
// workgroup folds the per-thread partials in shared memory. Layout metadata is
// shared with the other shape-aware kernels.
//
// Bindings:
//   0: var<storage, read> inp: array<f32>;
//   1: var<storage, read_write> out: array<f32>;
//   2: var<uniform> params: ReduceParams;
//   3: var<storage, read> metadata: array<u32>;
//
// meta layout:
//   dims[rank] | strides[rank] | mask[rank] |
//   reduced_dims[red_count] | reduced_strides[red_count]
//
// op codes: 0=sum, 1=min, 2=max, 3=mean, 4=prod.

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

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: ReduceParams;
@group(0) @binding(3) var<storage, read> metadata: array<u32>;

const BLOCK_SIZE: u32 = 256u;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

// Storage offset of output `out_idx`, applying the input layout start and the
// strides of every dimension that is *not* being reduced.
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

// Offset from `output_base(out_idx)` to the `i`-th element of the reduced
// subspace. Dimensions are folded from the innermost one outwards.
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

fn identity(op: u32) -> f32 {
    switch op {
        case OP_SUM:  { return 0.0; }
        case OP_MIN:  { return bitcast<f32>(0x7f800000u); }
        case OP_MAX:  { return bitcast<f32>(0xff800000u); }
        case OP_MEAN: { return 0.0; }
        case OP_PROD: { return 1.0; }
        default: { return 0.0; }
    }
}

fn combine(acc: f32, value: f32) -> f32 {
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
    // 每个 workgroup 负责一个输出的计算，workgroup 的索引相当于当前线程负责的输出的逻辑索引
    // 例如如果输出 shape 是 [4, 5]，逻辑索引从 0 -> 19
    let out_idx = wid.x;
    // 局部线程 id 
    let tid = lid.x;
    // 计算输出逻辑索引对应的输出地址偏移，例如 shape [4, 5]，索引是 14，对应 (2, 4)，再根据 stride 计算逻辑输出 2 * s1 + 4 * s2
    let base = output_base(out_idx);

    // 一共 256 个 invocation，负责 red_total 个计算，先在每个 thread 局部求和
    var acc = identity(params.op);
    for (var i: u32 = tid; i < params.red_total; i = i + BLOCK_SIZE) {
        acc = combine(acc, inp[base + reduced_offset(i)]);
    }

    // 每个 thread 计算出一个局部和，放到共享内存中
    scratch[tid] = acc;
    workgroupBarrier();

    // thread 树归约
    for (var stride: u32 = BLOCK_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            scratch[tid] = combine(scratch[tid], scratch[tid + stride]);
        }
        workgroupBarrier();
    }

    // 让 tid 0 写入
    if (tid == 0u) {
        if params.op == OP_MEAN {
            out[out_idx] = scratch[0] / f32(params.red_total);
        } else {
            out[out_idx] = scratch[0];
        }
    }
}
