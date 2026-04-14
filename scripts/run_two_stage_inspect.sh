#!/bin/bash
# Description: Latent inspection mode for the two-stage baseline.
# Saves intermediate latent artifacts and prints latent statistics.

bash scripts/run_two_stage.sh \
    --latent-dump-dir ./capture_latents \
    --log-latent-stats \
    "$@"
