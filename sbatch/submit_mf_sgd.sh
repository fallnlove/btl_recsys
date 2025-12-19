#!/bin/bash

SEARCH_DIR="sbatch/mf_sgd"

if [ ! -d "$SEARCH_DIR" ]; then
  echo "Directory $SEARCH_DIR does not exist."
  exit 1
fi

for file in "$SEARCH_DIR"/*.sbatch; do
  if [ -f "$file" ]; then
    echo "Submitting $file..."
    sbatch "$file"
  fi
done
