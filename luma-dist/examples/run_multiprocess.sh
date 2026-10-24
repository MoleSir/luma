#!/usr/bin/env bash
# 拉起 WORLD_SIZE 个进程，每个进程一个 rank / 一张卡(GPU ordinal = RANK)。
# 用法: WORLD_SIZE=4 ./run_multiprocess.sh
#
# 注意: 例子用 Cuda::new(rank)，即 rank 用「可见设备序号」选卡。
# 想指定物理卡请设 CUDA_VISIBLE_DEVICES，例如:
#   CUDA_VISIBLE_DEVICES=1,4 WORLD_SIZE=2 ./run_multiprocess.sh
# CUDA 默认按 FASTEST_FIRST 排序，序号可能和 nvidia-smi 的顺序不同。
set -euo pipefail

cd "$(dirname "$0")"

WORLD_SIZE="${WORLD_SIZE:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# 先编译一次，避免多个 cargo run 抢构建锁
cargo build --example nccl-multiprocess --features nccl

# workspace 根 target
BIN="../../target/debug/examples/nccl-multiprocess"
[[ -x "$BIN" ]] || { echo "找不到 $BIN"; exit 1; }

pids=()
for ((r = 0; r < WORLD_SIZE; r++)); do
  RANK=$r WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
    "$BIN" &
  pids+=($!)
  # rank0 先起来监听，避免其它 rank 的 connect 抢跑 (fetch_id 没有重试)
  if ((r == 0)); then sleep 0.5; fi
done

status=0
for p in "${pids[@]}"; do
  wait "$p" || status=$?
done
exit "$status"
