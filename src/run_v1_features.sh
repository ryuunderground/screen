#!/bin/bash
# Shard v1 feature extraction across login(96) + node01(64) + node02(64).
# shard i -> node (i % 3). Each shard is one background python process.
set -u
PY=~/.conda/envs/fieldscreen/bin/python
T=${1:-210}
cd ~/screen
mkdir -p out/v1/shards

launch() {   # $1=mod
  local mod=$1
  for i in $(seq 0 $((T-1))); do
    if [ $((i % 3)) -eq "$mod" ]; then
      nohup $PY src/v1_features.py --ids out/v1/ids.txt \
        --out out/v1/shards/feat_$i.npz --shard $((i+1))/$T \
        > out/v1/shards/log_$i.txt 2>&1 &
    fi
  done
  wait
}

# login node = mod 0 (local); node01 = mod 1; node02 = mod 2
launch 0 &
ssh node01 "cd ~/screen && PY=$PY; $(declare -f launch); T=$T; launch 1" &
ssh node02 "cd ~/screen && PY=$PY; $(declare -f launch); T=$T; launch 2" &
wait
echo "ALL_SHARDS_DONE"
