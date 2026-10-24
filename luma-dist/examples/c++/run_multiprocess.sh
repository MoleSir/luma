#!/usr/bin/env bash
# 启动 WORLD_SIZE 个进程，每个进程一个 rank / 一张卡。
set -euo pipefail

WORLD_SIZE="${WORLD_SIZE:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

pids=()
for ((r = 0; r < WORLD_SIZE; r++)); do
  RANK=$r LOCAL_RANK=$r WORLD_SIZE=$WORLD_SIZE \
  MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
    ./nccl_multiprocess &
  pids+=($!)
done

status=0
for p in "${pids[@]}"; do
  wait "$p" || status=$?
done
exit "$status"
