#!/usr/bin/env bash
# Does a migrated sequence produce the same tokens it would have produced if it
# had never moved?
#
# Reference run: A alone, start to finish.
# Test runs:     A -> B, with the handover pinned to an exact cache position,
#                including one inside the prompt (mid-prefill) and several
#                inside generation, plus a there-and-back-again.
# Control:       the same handover with the sampler RNG deliberately left
#                behind. It must diverge, or the test is not testing anything.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/qwen2.5-0.5b-instruct-q4km.gguf}
A=${A:-127.0.0.1:7401}
B=${B:-127.0.0.1:7402}
HS=./target/release/hotseat
SEED=${SEED:-424242}
MAX=${MAX:-96}
TEMP=${TEMP:-0.9}
PROMPT=${PROMPT:-"Explain what a page fault is, in two sentences."}
OUT=runs/verify
mkdir -p $OUT

tokens_of() { grep '^tokens ' | head -1; }

run_ref() {
  $HS "$A" start max=$MAX temp=$TEMP seed=$SEED top_k=40 top_p=0.95 "$PROMPT" >/dev/null || return 1
  $HS "$A" wait | tokens_of
}

run_migrated() {  # $1 = position to migrate at, $2 = extra migrate flags
  local at=$1 flags=${2:-}
  $HS "$A" start max=$MAX temp=$TEMP seed=$SEED top_k=40 top_p=0.95 "$PROMPT" >/dev/null || return 1
  if ! $HS "$A" migrate "$B" at="$at" $flags > "$OUT/migrate-$at${flags:+-$flags}.txt" 2>&1; then
      cat "$OUT/migrate-$at${flags:+-$flags}.txt" >&2; return 1
  fi
  $HS "$B" wait | tokens_of
}

run_round_trip() {  # A -> B -> A
  $HS "$A" start max=$MAX temp=$TEMP seed=$SEED top_k=40 top_p=0.95 "$PROMPT" >/dev/null || return 1
  local p1 p2 p3
  read -r p1 p2 p3 <<<"$(python3 -c "n=$NTOK; print(int(n*.3), int(n*.55), int(n*.8))")"
  $HS "$A" migrate "$B" at=$p1 > "$OUT/rt-1.txt" 2>&1 || { cat "$OUT/rt-1.txt" >&2; return 1; }
  $HS "$B" migrate "$A" at=$p2 > "$OUT/rt-2.txt" 2>&1 || { cat "$OUT/rt-2.txt" >&2; return 1; }
  $HS "$A" migrate "$B" at=$p3 > "$OUT/rt-3.txt" 2>&1 || { cat "$OUT/rt-3.txt" >&2; return 1; }
  $HS "$B" wait | tokens_of
}

echo "reference run on A (no migration)"
REF=$(run_ref) || exit 1
echo "$REF" > $OUT/reference.txt
NTOK=$(echo "$REF" | tr ',' '\n' | wc -l | tr -d ' ')
echo "  $NTOK tokens"
echo

# Migration points scaled to the reference run: one inside the prompt, the rest
# spread through generation. A point past the end of the sequence would just
# find it already finished.
POINTS=$(python3 -c "n=$NTOK; print(' '.join(str(max(4,int(n*f))) for f in (0.18,0.4,0.6,0.8,0.95)))")
echo "migration points: $POINTS"
echo

fail=0
for at in $POINTS; do
  GOT=$(run_migrated "$at") || { fail=1; continue; }
  if [ "$GOT" = "$REF" ]; then
    stw=$(grep -o 'stop-the-world [0-9.]* us' "$OUT/migrate-$at.txt" | head -1)
    printf 'PASS  migrate at position %-4s identical transcript   (%s)\n' "$at" "$stw"
  else
    printf 'FAIL  migrate at position %-4s transcript differs\n' "$at"
    diff <(echo "$REF" | tr ',' '\n') <(echo "$GOT" | tr ',' '\n') | head -5
    fail=1
  fi
done

GOT=$(run_round_trip) || fail=1
if [ "$GOT" = "$REF" ]; then
  echo "PASS  A->B->A->B round trip      identical transcript"
else
  echo "FAIL  A->B->A->B round trip      transcript differs"
  fail=1
fi

echo
echo "control: hand over the KV cache but not the sampler state"
CTRL=$(python3 -c "print(int($NTOK*.6))")
GOT=$(run_migrated $CTRL norng) || fail=1
if [ "$GOT" = "$REF" ]; then
  echo "FAIL  dropping the RNG changed nothing -- the test above proves nothing"
  fail=1
else
  common=$(awk 'NR==FNR{a[FNR]=$0;next}{if($0!=a[FNR]){print FNR-1;exit}}END{print FNR}' \
      <(echo "$REF" | tr ',' '\n') <(echo "$GOT" | tr ',' '\n') | head -1)
  echo "PASS  transcripts diverge as they must (first $common tokens shared, then they part)"
fi

echo
[ $fail -eq 0 ] && echo "all determinism checks passed" || echo "FAILURES"
exit $fail
