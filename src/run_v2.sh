#!/bin/bash
# Shard v2 redocking across login + node01 + node02. shard i -> node (i%3).
set -u
PY=~/.conda/envs/fieldscreen/bin/python
T=${1:-30}; LIMIT=${2:-100}; NROT=${3:-600}
cd ~/screen
mkdir -p out/v2/shards

launch() {
  local mod=$1
  for i in $(seq 0 $((T-1))); do
    if [ $((i % 3)) -eq "$mod" ]; then
      nohup $PY src/v2_dock.py --pb-root data/posebusters_x \
        --which posebusters_benchmark_set --limit $LIMIT --n-rot $NROT \
        --out out/v2/shards/dock_$i.jsonl --shard $((i+1))/$T \
        > out/v2/shards/log_$i.txt 2>&1 &
    fi
  done
  wait
}

launch 0 &
ssh node01 "cd ~/screen && PY=$PY; T=$T; LIMIT=$LIMIT; NROT=$NROT; $(declare -f launch); launch 1" &
ssh node02 "cd ~/screen && PY=$PY; T=$T; LIMIT=$LIMIT; NROT=$NROT; $(declare -f launch); launch 2" &
wait
echo "V2_ALL_DONE"
