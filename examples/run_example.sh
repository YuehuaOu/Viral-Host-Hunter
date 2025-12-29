#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

MODEL_DIR=${1:?Usage: ./examples/run_example.sh <model_dir> [prott5_dir]}
PROTT5_DIR=${2:-}

EMBED_DIR="$PROJECT_ROOT/examples/embedding"
OUT_DIR="$PROJECT_ROOT/examples/output"

PROTEIN="$PROJECT_ROOT/examples/gut/tail/protein.fasta"
DNA="$PROJECT_ROOT/examples/gut/tail/dna.fasta"

if [[ ! -f "$PROTEIN" ]]; then
  echo "Protein FASTA not found: $PROTEIN" >&2
  exit 1
fi
if [[ ! -f "$DNA" ]]; then
  echo "DNA FASTA not found: $DNA" >&2
  exit 1
fi

mkdir -p "$EMBED_DIR" "$OUT_DIR"

vhh-predict \
--protein "$PROTEIN" \
--dna "$DNA" \
--phage_type gut --seq_type tail \
--embedding_dir "$EMBED_DIR" \
--output_dir "$OUT_DIR" \
--model_dir "$MODEL_DIR" \
${PROTT5_DIR:+--prott5_dir "$PROTT5_DIR"}
