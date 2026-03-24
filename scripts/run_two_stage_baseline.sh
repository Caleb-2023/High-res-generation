#!/bin/bash
# Description: Two-stage train-free high-resolution video generation baseline.

python3 sample_video_two_stage.py \
    --lr-size 544 960 \
    --hr-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --capture-step 25 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results
