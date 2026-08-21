#!/usr/bin/env bash
# Move a sequence from a native macOS worker into a Linux container, and check
# the token stream is the one the Mac would have produced on its own.
#
# The two ends differ in operating system, page size (16 KiB vs 4 KiB), dirty
# tracking mechanism (Mach exception ports vs mprotect + SIGSEGV) and C library.
# None of that is allowed to change a single token, which is why the engine owns
# its own exp/ln/sincos and fixes one accumulation order for every reduction.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/qwen2.5-0.5b-instruct-q4km.gguf}
HS=./target/release/hotseat
HOST_PORT=${HOST_PORT:-7401}
CTR_PORT=${CTR_PORT:-7412}
PROMPT=${PROMPT:-"Describe what happens when a program writes to a page that has been marked read-only."}
mkdir -p runs

cleanup() { docker rm -f hs-linux >/dev/null 2>&1 || true; pkill -f "listen 0.0.0.0:$HOST_PORT" 2>/dev/null || true; }
trap cleanup EXIT
cleanup

docker run -d --name hs-linux -p "$CTR_PORT":7400 \
  -v "$PWD/$MODEL":/models/model.gguf:ro hotseat \
  --model /models/model.gguf --listen 0.0.0.0:7400 --name linux-container \
  --threads 4 --max-ctx 1024 >/dev/null
./target/release/hs-worker --model "$MODEL" --listen 0.0.0.0:"$HOST_PORT" \
  --name macos-host --threads 4 --max-ctx 1024 > runs/crosshost.log 2>&1 &
for _ in $(seq 60); do
  $HS 127.0.0.1:"$HOST_PORT" info >/dev/null 2>&1 && $HS 127.0.0.1:"$CTR_PORT" info >/dev/null 2>&1 && break
  sleep 0.5
done

echo "the two ends:"
for p in "$HOST_PORT" "$CTR_PORT"; do
  $HS 127.0.0.1:"$p" info | awk -F' ' '/^name/{n=$2} /^os/{os=$2; a=$4} /^page_size/{ps=$2} /^tracker/{t=$2} END{printf "  %-16s %s/%s  %s B pages  %s\n", n, os, a, ps, t}'
done
echo

$HS 127.0.0.1:"$HOST_PORT" start max=100 temp=0.85 seed=31337 "$PROMPT" >/dev/null
REF=$($HS 127.0.0.1:"$HOST_PORT" wait | grep '^tokens ')
echo "reference: $(echo "$REF" | tr ',' '\n' | wc -l | tr -d ' ') tokens, never left the Mac"

$HS 127.0.0.1:"$HOST_PORT" start max=100 temp=0.85 seed=31337 "$PROMPT" >/dev/null
$HS 127.0.0.1:"$HOST_PORT" migrate 127.0.0.1:"$CTR_PORT" at=60 verify | sed 's/^/  /'
GOT=$($HS 127.0.0.1:"$CTR_PORT" wait | grep '^tokens ')
echo
if [ "$REF" = "$GOT" ]; then
  echo "PASS  identical token stream across macOS -> Linux container"
else
  echo "FAIL  transcripts differ"; exit 1
fi
