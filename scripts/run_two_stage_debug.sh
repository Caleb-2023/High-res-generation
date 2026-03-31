#!/bin/bash
# Description: Image-space two-stage debug baseline with latent dumps and optional direct-HR comparison.

bash scripts/run_two_stage.sh \
    --debug-latent-dir ./capture_latents \
    --log-latent-stats \
    --match-init-stats \
    --run-direct-hr \
    "$@"
