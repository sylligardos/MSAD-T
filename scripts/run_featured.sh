#!/bin/bash

sizes=(16 32)

for size in "${sizes[@]}"; do
    echo "Running for size $size..."
    python3 src/generate_featured_dataset.py -p "data/TSB_${size}/" -s "data/features/" -f catch22
done