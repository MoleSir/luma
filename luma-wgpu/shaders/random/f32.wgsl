// Counter-based random generation. Each invocation derives independent values
// from (seed, index), so the kernel has no mutable RNG state.

struct RandomParams {
    n: u32,
    op: u32,
    seed: u32,
    p0: f32,
    p1: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: RandomParams;

const INV_2_24: f32 = 1.0 / 16777216.0;

fn hash32(x: u32) -> u32 {
    var v = x;
    v = v ^ (v >> 16u);
    v = v * 0x7feb352du;
    v = v ^ (v >> 15u);
    v = v * 0x846ca68bu;
    v = v ^ (v >> 16u);
    return v;
}

fn uniform01(seed: u32, index: u32) -> f32 {
    return f32(hash32(seed ^ hash32(index)) >> 8u) * INV_2_24;
}

fn standard_normal(seed: u32, index: u32) -> f32 {
    let u1 = max(uniform01(seed, index), INV_2_24);
    let u2 = uniform01(seed ^ 0x9e3779b9u, index);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

@compute @workgroup_size(256)
fn random(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        var value: f32;
        if (params.op == 0u) {
            value = params.p0 + params.p1 * uniform01(params.seed, i);
        } else {
            value = params.p0 + params.p1 * standard_normal(params.seed, i);
        }
        out[i] = value;
    }
}
