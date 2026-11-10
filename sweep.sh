#!/usr/bin/env bash
# Usage: ./sweep.sh MODEL "LAYOUT FLAGS" TAG ["CONCURRENCY LIST"]
# Optional env vars: IN, OUT (prompt/output lengths, default 1024/256), EXTRA (extra server flags)
set -u
MODEL=$1; LAYOUT=$2; TAG=$3; CONCS=${4:-"1 4 16 64 128"}
IN=${IN:-1024}; OUT=${OUT:-256}; PORT=${PORT:-8001}
DIR=results/${TAG}_in${IN}_out${OUT}; mkdir -p "$DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

vllm serve "$MODEL" $LAYOUT --max-model-len 4096 --gpu-memory-utilization 0.92 \
  --no-enable-prefix-caching --port $PORT ${EXTRA:-} > "$DIR/server.log" 2>&1 &
PID=$!

until curl -sf localhost:$PORT/health > /dev/null; do
  kill -0 $PID 2>/dev/null || { echo "server died - see $DIR/server.log"; exit 1; }
  sleep 5
done

# warm-up, not saved
vllm bench serve --model "$MODEL" --port $PORT --dataset-name random --random-input-len $IN \
  --random-output-len 32 --num-prompts 8 --max-concurrency 4 > /dev/null 2>&1

for C in $CONCS; do
  N=$(( C*4 > 10 ? C*4 : 10 ))
  vllm bench serve --model "$MODEL" --dataset-name random \
    --random-input-len $IN --random-output-len $OUT --ignore-eos \
    --num-prompts $N --max-concurrency $C \
    --save-result --result-dir "$DIR" --result-filename "c${C}.json" | tee "$DIR/c${C}.txt"
done

kill $PID; wait $PID 2>/dev/null; sleep 10   # give GPUs time to free memory
