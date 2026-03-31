#!/bin/bash
# Description: Image-space two-stage debug baseline with latent dumps and optional direct-HR comparison.

python3 sample_video_two_stage_debug.py \
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
    --use-cpu-offload \
    --interpolation-mode bilinear \
    --run-direct-hr \
    --capture-dir /root/autodl-tmp/HunyuanVideo/HunyuanVideo/capture_latents \
    --save-path ./results
