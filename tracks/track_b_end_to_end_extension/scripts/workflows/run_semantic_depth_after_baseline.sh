#!/bin/bash

echo "Waiting for baseline training to finish..."

while pgrep -f "baseline_identity_output" >/dev/null
do
    sleep 60
done

echo "Baseline complete."
date

cd /media/sherifdeen/ext_hd250/projects/deepdct-vo

source .venv/bin/activate

python scripts/train_deepdct_vo.py \
    --data-root data \
    --train-sequences 00 01 02 03 04 05 06 07 08 \
    --validation-sequences 09 \
    --epochs 15 \
    --batch-size 1 \
    --num-workers 0 \
    --experiment-name semantic_depth_identity_output \
    --checkpoint-dir experiments/semantic_depth_identity_output \
    --use-semantic-cues \
    --use-depth-cues \
    --freeze-semantic \
    --freeze-depth \
    --depth-checkpoint-dir weights/lite-mono-tiny-640x192 \
    --depth-model-name lite-mono-tiny \
    --depth-output-mode normalized_depth
