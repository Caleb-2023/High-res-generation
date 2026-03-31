import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from hyvideo.config import (
    add_denoise_schedule_args,
    add_extra_models_args,
    add_inference_args,
    add_network_args,
    add_parallel_args,
    sanity_check_args,
)
from hyvideo.constants import PRECISION_TO_TYPE
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.utils.data_utils import align_to
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.utils.latent_utils import (
    decode_latents_to_video,
    encode_video_to_latents,
    resize_video_frames_framewise,
    sample_noise_like,
)


def build_two_stage_parser():
    parser = argparse.ArgumentParser(
        description="Two-stage train-free high-resolution video generation baseline"
    )
    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_inference_args(parser)
    parser = add_parallel_args(parser)

    parser.add_argument(
        "--lr-size",
        type=int,
        nargs=2,
        default=(544, 960),
        metavar=("HEIGHT", "WIDTH"),
        help="Low-resolution stage size.",
    )
    parser.add_argument(
        "--hr-size",
        type=int,
        nargs=2,
        default=(720, 1280),
        metavar=("HEIGHT", "WIDTH"),
        help="High-resolution stage size.",
    )
    parser.add_argument(
        "--capture-step",
        type=int,
        default=25,
        help="Absolute denoising step index used to capture the intermediate LR latent z_t.",
    )
    parser.add_argument(
        "--capture-save-path",
        type=str,
        default="",
        help="Optional explicit file path for the captured LR latent payload.",
    )
    parser.add_argument(
        "--debug-latent-dir",
        "--capture-dir",
        dest="debug_latent_dir",
        type=str,
        default="",
        help="Optional directory used to save LR z_t, mapped HR latent, and noised HR init latent payloads.",
    )
    parser.add_argument(
        "--interpolation-mode",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "area"],
        help="Spatial interpolation mode for decoded LR video resizing before re-encoding.",
    )
    parser.add_argument(
        "--mapping-source",
        type=str,
        default="clean_estimate",
        choices=["clean_estimate", "z_t"],
        help="Which LR latent to map into HR space. `clean_estimate` uses the denoiser-derived clean "
        "latent at capture step; `z_t` uses the raw captured noisy latent.",
    )
    parser.add_argument(
        "--renoise-mode",
        type=str,
        default="scheduler_sigma",
        choices=["scheduler_sigma", "step_ratio", "none"],
        help="How to re-noise the mapped HR latent before resume. `scheduler_sigma` matches the "
        "actual FlowMatch scheduler sigma at `capture_step`. `step_ratio` uses a linear "
        "approximation `capture_step / infer_steps * latent + (1 - capture_step / infer_steps) * noise`.",
    )
    parser.add_argument(
        "--noise-seed-offset",
        type=int,
        default=0,
        help="Offset added to each sample seed when drawing HR re-noise.",
    )
    parser.add_argument(
        "--match-init-stats",
        action="store_true",
        help="Match per-channel mean/std of the mapped HR latent to the captured LR latent before re-noising.",
    )
    parser.add_argument(
        "--log-latent-stats",
        action="store_true",
        help="Log shape/mean/std/min/max for captured, mapped, and noised HR latents.",
    )
    parser.add_argument(
        "--run-direct-hr",
        action="store_true",
        help="Also run a direct HR baseline from pure noise for comparison.",
    )
    parser.add_argument(
        "--hr-start-mode",
        type=str,
        default="continue",
        choices=["continue"],
        help="HR resume mode. Only `continue` is supported for step-matched re-noising.",
    )
    return parser


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")


def save_video_outputs(outputs, save_dir, tag):
    samples = outputs["samples"]
    if samples is None:
        raise ValueError(f"No decoded samples were returned for tag `{tag}`.")

    saved_paths = []
    for i, sample in enumerate(samples):
        sample = sample.unsqueeze(0)
        prompt_prefix = outputs["prompts"][i][:100].replace("/", "")
        video_path = (
            f"{save_dir}/{timestamp()}_{tag}_seed{outputs['seeds'][i]}_{prompt_prefix}.mp4"
        )
        save_videos_grid(sample, video_path, fps=24)
        logger.info(f"Saved {tag} video to: {video_path}")
        saved_paths.append(video_path)
    return saved_paths


def validate_resume_timestep(args, capture_step, captured_timestep, device):
    scheduler = FlowMatchDiscreteScheduler(
        shift=args.flow_shift,
        reverse=args.flow_reverse,
        solver=args.flow_solver,
    )
    scheduler.set_timesteps(args.infer_steps, device=device)
    expected_timestep = scheduler.timesteps[capture_step]

    if torch.is_tensor(captured_timestep):
        captured_timestep = captured_timestep.to(
            device=device, dtype=expected_timestep.dtype
        )
    else:
        captured_timestep = torch.tensor(
            captured_timestep, device=device, dtype=expected_timestep.dtype
        )

    if not torch.isclose(captured_timestep, expected_timestep):
        raise ValueError(
            f"Captured timestep {captured_timestep.item()} does not match scheduler timestep "
            f"{expected_timestep.item()} at step {capture_step}."
        )

    return float(scheduler.sigmas[capture_step].item())


def summarize_latents(name, latents):
    latents = latents.detach().float().cpu()
    stats = {
        "shape": tuple(latents.shape),
        "mean": latents.mean().item(),
        "std": latents.std().item(),
        "min": latents.min().item(),
        "max": latents.max().item(),
    }
    logger.info(
        f"{name}: shape={stats['shape']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
        f"min={stats['min']:.6f}, max={stats['max']:.6f}"
    )
    return stats


def match_latent_stats(source_latents, target_latents, eps=1e-6):
    reduce_dims = (2, 3, 4)
    src_mean = source_latents.mean(dim=reduce_dims, keepdim=True)
    src_std = source_latents.std(dim=reduce_dims, keepdim=True)
    tgt_mean = target_latents.mean(dim=reduce_dims, keepdim=True)
    tgt_std = target_latents.std(dim=reduce_dims, keepdim=True)
    normalized = (target_latents - tgt_mean) / torch.clamp(tgt_std, min=eps)
    return normalized * src_std + src_mean


def maybe_log_latents(name, latents, enabled):
    if enabled:
        summarize_latents(name, latents)


def save_latent_payload(path, payload, label):
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info(f"Saved {label} payload to: {path}")


def main():
    parser = build_two_stage_parser()
    args = sanity_check_args(parser.parse_args())
    print(args)

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    save_path = (
        args.save_path
        if args.save_path_suffix == ""
        else f"{args.save_path}_{args.save_path_suffix}"
    )
    os.makedirs(save_path, exist_ok=True)

    debug_latent_dir = Path(args.debug_latent_dir) if args.debug_latent_dir else None
    if debug_latent_dir is not None:
        debug_latent_dir.mkdir(parents=True, exist_ok=True)

    sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = sampler.args

    run_tag = f"step{args.capture_step}_seed{args.seed if args.seed is not None else 'auto'}"
    lr_height, lr_width = args.lr_size
    hr_height, hr_width = args.hr_size

    auto_capture_path = (
        debug_latent_dir / f"{run_tag}_lr_z_t.pt" if debug_latent_dir is not None else None
    )
    capture_save_path = (
        Path(args.capture_save_path)
        if args.capture_save_path
        else auto_capture_path
    )
    hr_encoded_path = (
        debug_latent_dir / f"{run_tag}_hr_encoded.pt" if debug_latent_dir is not None else None
    )
    hr_init_path = (
        debug_latent_dir / f"{run_tag}_hr_init_noisy.pt" if debug_latent_dir is not None else None
    )

    logger.info(
        f"Running two-stage baseline with capture_step={args.capture_step}, "
        f"lr_size={tuple(args.lr_size)}, hr_size={tuple(args.hr_size)}, "
        f"mapping_source={args.mapping_source}, interpolation={args.interpolation_mode}, "
        f"renoise_mode={args.renoise_mode}, "
        f"match_init_stats={args.match_init_stats}, noise_seed_offset={args.noise_seed_offset}"
    )

    lr_outputs = sampler.predict(
        prompt=args.prompt,
        height=lr_height,
        width=lr_width,
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        capture_step=args.capture_step,
        capture_save_path=str(capture_save_path) if capture_save_path is not None else None,
        return_captured_latents=True,
        stop_after_capture=True,
    )

    captured_latents = lr_outputs["captured_latents"]
    captured_clean_latents = lr_outputs["captured_clean_latents"]
    captured_step = lr_outputs["captured_step"]
    captured_timestep = lr_outputs["captured_timestep"]
    if captured_latents is None or captured_step is None or captured_timestep is None:
        raise ValueError("Failed to capture LR intermediate latent z_t.")
    if args.mapping_source == "clean_estimate" and captured_clean_latents is None:
        raise ValueError("Failed to capture LR clean latent estimate at the selected step.")

    resume_sigma = validate_resume_timestep(
        args,
        capture_step=captured_step,
        captured_timestep=captured_timestep,
        device=captured_latents.device,
    )
    maybe_log_latents("Captured LR z_t", captured_latents, args.log_latent_stats)
    maybe_log_latents(
        "Captured LR clean estimate",
        captured_clean_latents,
        args.log_latent_stats and captured_clean_latents is not None,
    )
    logger.info(
        f"Resolved capture sigma={resume_sigma:.6f} at step={captured_step} "
        f"(signal_scale={1.0 - resume_sigma:.6f}, noise_scale={resume_sigma:.6f})"
    )

    if args.mapping_source == "clean_estimate":
        mapping_source_latents = captured_clean_latents
        mapping_source_label = "clean_estimate"
    else:
        mapping_source_latents = captured_latents
        mapping_source_label = "z_t"
    logger.info(f"Using `{mapping_source_label}` as the LR->HR mapping source.")

    hr_height = align_to(hr_height, 16)
    hr_width = align_to(hr_width, 16)
    vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
    vae_autocast_enabled = (
        vae_dtype != torch.float32
    ) and not args.disable_autocast

    lr_decoded_video = decode_latents_to_video(
        sampler.pipeline.vae,
        mapping_source_latents,
        vae_dtype=vae_dtype,
        autocast_enabled=vae_autocast_enabled,
        enable_tiling=args.vae_tiling,
    )
    logger.info(
        f"Decoded LR `{mapping_source_label}` to pixel-space video with shape={tuple(lr_decoded_video.shape)}"
    )
    hr_resized_video = resize_video_frames_framewise(
        lr_decoded_video,
        target_size=(hr_height, hr_width),
        mode=args.interpolation_mode,
    )
    logger.info(
        f"Resized pixel-space video to HR shape={tuple(hr_resized_video.shape)}"
    )
    hr_encoded_latents = encode_video_to_latents(
        sampler.pipeline.vae,
        hr_resized_video,
        vae_dtype=vae_dtype,
        autocast_enabled=vae_autocast_enabled,
        enable_tiling=args.vae_tiling,
    )
    maybe_log_latents(
        "Re-encoded HR latent before stat match",
        hr_encoded_latents,
        args.log_latent_stats,
    )

    if args.match_init_stats:
        hr_encoded_latents = match_latent_stats(mapping_source_latents, hr_encoded_latents)
        maybe_log_latents(
            "Re-encoded HR latent after stat match",
            hr_encoded_latents,
            args.log_latent_stats,
        )

    save_latent_payload(
        hr_encoded_path,
        {
            "latents": hr_encoded_latents.detach().cpu(),
            "source_capture_path": str(capture_save_path)
            if capture_save_path is not None
            else None,
            "capture_step": captured_step,
            "capture_timestep": captured_timestep.cpu()
            if torch.is_tensor(captured_timestep)
            else captured_timestep,
            "mapping_source": mapping_source_label,
            "hr_size": (hr_height, hr_width),
            "video_length": args.video_length,
            "interpolation_mode": args.interpolation_mode,
            "match_init_stats": args.match_init_stats,
            "mapping_space": "image",
        },
        label="HR encoded latent",
    )

    if args.renoise_mode == "scheduler_sigma":
        signal_scale = 1.0 - resume_sigma
        noise_scale = resume_sigma
        resume_noise = sample_noise_like(
            hr_encoded_latents,
            seeds=lr_outputs["seeds"],
            seed_offset=args.noise_seed_offset,
        )
        hr_init_latents = (
            hr_encoded_latents * signal_scale + resume_noise * noise_scale
        )
    elif args.renoise_mode == "step_ratio":
        signal_scale = float(captured_step) / float(args.infer_steps)
        noise_scale = 1.0 - signal_scale
        resume_noise = sample_noise_like(
            hr_encoded_latents,
            seeds=lr_outputs["seeds"],
            seed_offset=args.noise_seed_offset,
        )
        hr_init_latents = (
            hr_encoded_latents * signal_scale + resume_noise * noise_scale
        )
    else:
        hr_init_latents = hr_encoded_latents
        resume_noise = None
        signal_scale = 1.0
        noise_scale = 0.0

    logger.info(
        f"Prepared HR init latent with signal_scale={signal_scale:.6f}, "
        f"noise_scale={noise_scale:.6f}, start_step={captured_step}"
    )
    maybe_log_latents("Noised HR init latent", hr_init_latents, args.log_latent_stats)

    save_latent_payload(
        hr_init_path,
        {
            "latents": hr_init_latents.detach().cpu(),
            "source_capture_path": str(capture_save_path)
            if capture_save_path is not None
            else None,
            "capture_step": captured_step,
            "capture_timestep": captured_timestep.cpu()
            if torch.is_tensor(captured_timestep)
            else captured_timestep,
            "mapping_source": mapping_source_label,
            "hr_size": (hr_height, hr_width),
            "video_length": args.video_length,
            "interpolation_mode": args.interpolation_mode,
            "match_init_stats": args.match_init_stats,
            "mapping_space": "image",
            "renoise_mode": args.renoise_mode,
            "signal_scale": signal_scale,
            "noise_scale": noise_scale,
            "resume_sigma": resume_sigma,
            "noise_seed_offset": args.noise_seed_offset,
            "per_sample_seeds": lr_outputs["seeds"],
            "has_resume_noise": resume_noise is not None,
        },
        label="HR init latent",
    )

    hr_outputs = sampler.predict(
        prompt=args.prompt,
        height=hr_height,
        width=hr_width,
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        start_step=captured_step,
        init_latents=hr_init_latents,
    )
    save_video_outputs(
        hr_outputs,
        save_path,
        tag=f"two_stage_{args.renoise_mode}_{run_tag}",
    )

    if args.run_direct_hr:
        logger.info("Running direct HR baseline from pure noise for comparison")
        direct_hr_outputs = sampler.predict(
            prompt=args.prompt,
            height=hr_height,
            width=hr_width,
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
        )
        save_video_outputs(direct_hr_outputs, save_path, tag=f"direct_hr_{run_tag}")


if __name__ == "__main__":
    main()
