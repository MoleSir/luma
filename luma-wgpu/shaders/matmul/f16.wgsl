enable f16;

// Tiled f16 matmul. Storage is f16; shared tiles and accumulation are f32.

struct MatmulParams {
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    lhs_start: u32,
    rhs_start: u32,
    dst_start: u32,
    lhs_meta: u32,
    rhs_meta: u32,
    lhs_batch_ndim: u32,
    rhs_batch_ndim: u32,
    lhs_sm: u32,
    lhs_sk: u32,
    rhs_sm: u32,
    rhs_sk: u32,
    accumulate: u32,
};

@group(0) @binding(0) var<storage, read> lhs: array<f16>;
@group(0) @binding(1) var<storage, read> rhs: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;
@group(0) @binding(3) var<uniform> p: MatmulParams;
@group(0) @binding(4) var<storage, read> metadata: array<u32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

const IVC_M: u32 = 4u;
const IVC_N: u32 = 4u;

const NUM_IVC_M: u32 = BM / IVC_M; // 16
const NUM_IVC_N: u32 = BN / IVC_N; // 16
const NUM_INVOCATIONS: u32 = NUM_IVC_M * NUM_IVC_N; // 256

var<workgroup> tile_a: array<array<f32, BK>, BM>; // [BM][BK]
var<workgroup> tile_b: array<array<f32, BN>, BK>; // [BK][BN]

fn batch_offset(b: u32, base: u32, ndim: u32, start: u32) -> u32 {
    var off = start;
    var t = b;

    for (var i: u32 = 0u; i < ndim; i += 1u) {
        let dim = metadata[base + i];
        let idx = t % dim;
        t = t / dim;
        off += idx * metadata[base + ndim + i];
    }

    return off;
}

@compute @workgroup_size(NUM_IVC_M, NUM_IVC_N, 1)
fn matmul(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let M = p.m;
    let N = p.n;
    let K = p.k;
    let batch = wid.z;

    let row_wg = wid.y * BM;
    let col_wg = wid.x * BN;

    let row_base = row_wg + lid.y * IVC_M;
    let col_base = col_wg + lid.x * IVC_N;

    let iid = lid.y * NUM_IVC_N + lid.x;

    let lhs_off = batch_offset(batch, p.lhs_meta, p.lhs_batch_ndim, p.lhs_start);
    let rhs_off = batch_offset(batch, p.rhs_meta, p.rhs_batch_ndim, p.rhs_start);

    var acc: array<array<f32, IVC_N>, IVC_M>;
    for (var i: u32 = 0u; i < IVC_M; i += 1u) {
        for (var j: u32 = 0u; j < IVC_N; j += 1u) {
            acc[i][j] = 0.0;
        }
    }

    let num_tiles = (K + BK - 1u) / BK;
    for (var t: u32 = 0u; t < num_tiles; t += 1u) {
        let k_base = t * BK;

        for (var i: u32 = iid; i < BM * BK; i += NUM_INVOCATIONS) {
            let r = i / BK;
            let c = i % BK;
            let rg = row_wg + r;
            let cg = k_base + c;

            if (rg < M && cg < K) {
                tile_a[r][c] = f32(lhs[lhs_off + rg * p.lhs_sm + cg * p.lhs_sk]);
            } else {
                tile_a[r][c] = 0.0;
            }
        }

        for (var i: u32 = iid; i < BK * BN; i += NUM_INVOCATIONS) {
            let r = i / BN;
            let c = i % BN;
            let rg = k_base + r;
            let cg = col_wg + c;

            if (rg < K && cg < N) {
                tile_b[r][c] = f32(rhs[rhs_off + rg * p.rhs_sm + cg * p.rhs_sk]);
            } else {
                tile_b[r][c] = 0.0;
            }
        }

        workgroupBarrier();

        for (var kk: u32 = 0u; kk < BK; kk += 1u) {
            var arow: array<f32, IVC_M>;
            var bcol: array<f32, IVC_N>;

            for (var i: u32 = 0u; i < IVC_M; i += 1u) {
                arow[i] = tile_a[lid.y * IVC_M + i][kk];
            }
            for (var j: u32 = 0u; j < IVC_N; j += 1u) {
                bcol[j] = tile_b[kk][lid.x * IVC_N + j];
            }

            for (var i: u32 = 0u; i < IVC_M; i += 1u) {
                for (var j: u32 = 0u; j < IVC_N; j += 1u) {
                    acc[i][j] += arow[i] * bcol[j];
                }
            }
        }

        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < IVC_M; i += 1u) {
        for (var j: u32 = 0u; j < IVC_N; j += 1u) {
            let row = row_base + i;
            let col = col_base + j;

            if (row < M && col < N) {
                let idx = p.dst_start + batch * M * N + row * N + col;
                if (p.accumulate != 0u) {
                    out[idx] = f16(f32(out[idx]) + acc[i][j]);
                } else {
                    out[idx] = f16(acc[i][j]);
                }
            }
        }
    }
}
