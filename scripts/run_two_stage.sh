#!/bin/bash
# Description: Unified two-stage train-free high-resolution video generation script.

python3 sample_video_two_stage.py \
    --lr-size 544 960 \
    --hr-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --capture-step 35 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --cfg-scale 1.0 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --hr-start-mode continue \
    --renoise-mode scheduler_sigma \
    --use-cpu-offload \
    --save-path ./results \
    "$@"
