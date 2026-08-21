#!/usr/bin/env bash
# What pre-copy is for, and when it stops helping.
#
# Loopback moves a KV cache far faster than a sequence can dirty it, so the
# algorithm converges in two rounds and never shows its work. Shaping the link
# makes the rounds mean something -- and taking the link below the sequence's
# dirty rate shows the case the convergence rule exists to detect.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/qwen2.5-0.5b-instruct-q4km.gguf}
A=${A:-127.0.0.1:7401}
B=${B:-127.0.0.1:7402}
HS=./target/release/hotseat
RATES=${RATES:-"1000 200 60 10"}
AT=${AT:-300}
mkdir -p runs

# A long prompt, so the live cache is megabytes rather than kilobytes.
LONG=$(python3 -c "print(open('bench/ppl.txt').read().strip().replace(chr(10),'\\\\n'))")

SRC=$A; DST=$B
for MB in $RATES; do
  # Both workers hold one sequence at a time, so clear whatever the previous
  # round of this loop left behind.
  $HS "$A" stop >/dev/null 2>&1; $HS "$B" stop >/dev/null 2>&1
  $HS "$SRC" start max=2500 temp=0.8 seed=5 raw "$LONG" >/dev/null || exit 1
  echo "======== link shaped to ${MB} Mbit/s"
  $HS "$SRC" migrate "$DST" at=$AT mbps=$MB rounds=8 verify 2>&1 | sed 's/^/  /'
  echo
  T=$SRC; SRC=$DST; DST=$T
done
$HS "$A" stop >/dev/null 2>&1; $HS "$B" stop >/dev/null 2>&1

cat <<'NOTE'
Read the round column, not the totals. Each round carries what the sequence
dirtied while the previous one was in flight, so the ratio between consecutive
rounds is (dirty rate / link rate). Below 1 it converges and the pause is the
residue; at or above 1 it never will, and the rule stops early rather than
spending another round to arrive somewhere worse.
NOTE
