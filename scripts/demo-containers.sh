#!/usr/bin/env bash
# Move a decoding sequence between two containers.
#
# Both run the same image on one Docker network, with the model mounted
# read-only -- the weights are not migrated and never should be, so each worker
# loads its own copy and the handshake checks they are the same file.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/qwen2.5-0.5b-instruct-q4km.gguf}
IMAGE=${IMAGE:-hotseat}
NET=${NET:-hotseat-net}
CTX=${CTX:-1024}
THREADS=${THREADS:-4}
PROMPT=${PROMPT:-"Explain, in a short paragraph, why a cache is faster than main memory."}

[ -f "$MODEL" ] || { echo "no model at $MODEL"; exit 1; }
MODEL_ABS=$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")

cleanup() { docker rm -f hs-a hs-b >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup
docker network create "$NET" >/dev/null 2>&1 || true

start() {  # $1 = container name, $2 = worker name
  docker run -d --name "$1" --network "$NET" --network-alias "$2" \
    -v "$MODEL_ABS":/models/model.gguf:ro \
    "$IMAGE" --model /models/model.gguf --listen 0.0.0.0:7400 \
    --name "$2" --threads "$THREADS" --max-ctx "$CTX" >/dev/null
}

hs() { local c=$1; shift; docker exec "$c" /usr/local/bin/hotseat "$@"; }

echo "starting two containers on the $NET network"
start hs-a a
start hs-b b
for c in hs-a hs-b; do
  for _ in $(seq 60); do
    docker logs "$c" 2>&1 | grep -q "listening on" && break
    sleep 0.5
  done
done
docker logs hs-a 2>&1 | sed 's/^/  hs-a: /'
echo

echo "starting a sequence in container hs-a"
hs hs-a a:7400 start max=120 temp=0.8 seed=99 "$PROMPT"
sleep 1.5
hs hs-a a:7400 status | sed 's/^/  /'
echo

echo "migrating it to container hs-b, without stopping it"
hs hs-a a:7400 migrate b:7400 verify | sed 's/^/  /'
echo

echo "hs-b finishes the sequence hs-a started:"
hs hs-b b:7400 wait | grep '^text ' | cut -c6- | fold -s -w 78 | sed 's/^/  /'
echo
echo "tokens decoded on each side:"
printf '  hs-a: %s\n' "$(hs hs-a a:7400 status | awk '/^decoded_here/{print $2}')"
printf '  hs-b: %s\n' "$(hs hs-b b:7400 status | awk '/^decoded_here/{print $2}')"
