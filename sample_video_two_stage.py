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
        help="Optional path to save the captured LR latent payload.",
    )
    parser.add_argument(
        "--interpolation-mode",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "area"],
        help="Spatial interpolation mode for decoded LR video resizing before re-encoding.",
    )
    parser.add_argument(
        "--hr-start-mode",
        type=str,
        default="continue",
        choices=["restart", "continue"],
        help="How HR sampling starts from the mapped latent. `restart` runs a full HR denoising schedule; "
        "`continue` resumes from `capture_step`.",
    )
    return parser


def validate_resume_timestep(args, capture_step, captured_timestep, device):
    scheduler = FlowMatchDiscreteScheduler(
        shift=args.flow_shift,
        reverse=args.flow_reverse,
        solver=args.flow_solver,
    )
    scheduler.set_timesteps(args.infer_steps, device=device)
    expected_timestep = scheduler.timesteps[capture_step]

    if torch.is_tensor(captured_timestep):
        captured_timestep = captured_timestep.to(device=device, dtype=expected_timestep.dtype)
    else:
        captured_timestep = torch.tensor(
            captured_timestep, device=device, dtype=expected_timestep.dtype
        )

    if not torch.isclose(captured_timestep, expected_timestep):
        raise ValueError(
            f"Captured timestep {captured_timestep.item()} does not match scheduler timestep "
            f"{expected_timestep.item()} at step {capture_step}."
        )


def save_outputs(outputs, save_dir, tag):
    samples = outputs["samples"]
    if samples is None:
        raise ValueError("No decoded samples were returned.")

    for i, sample in enumerate(samples):
        sample = sample.unsqueeze(0)
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        prompt_prefix = outputs["prompts"][i][:100].replace("/", "")
        cur_save_path = (
            f"{save_dir}/{time_flag}_{tag}_seed{outputs['seeds'][i]}_{prompt_prefix}.mp4"
        )
        save_videos_grid(sample, cur_save_path, fps=24)
        logger.info(f"Sample saved to: {cur_save_path}")


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

    sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = sampler.args

    lr_height, lr_width = args.lr_size
    hr_height, hr_width = args.hr_size
    capture_save_path = args.capture_save_path or None

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
        capture_save_path=capture_save_path,
        return_captured_latents=True,
        stop_after_capture=True,
    )

    captured_latents = lr_outputs["captured_latents"]
    captured_step = lr_outputs["captured_step"]
    captured_timestep = lr_outputs["captured_timestep"]
    if captured_latents is None or captured_step is None or captured_timestep is None:
        raise ValueError("Failed to capture LR intermediate latent z_t.")

    validate_resume_timestep(
        args,
        capture_step=captured_step,
        captured_timestep=captured_timestep,
        device=captured_latents.device,
    )

    hr_height = align_to(hr_height, 16)
    hr_width = align_to(hr_width, 16)
    vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
    vae_autocast_enabled = (
        vae_dtype != torch.float32
    ) and not args.disable_autocast

    lr_decoded_video = decode_latents_to_video(
        sampler.pipeline.vae,
        captured_latents,
        vae_dtype=vae_dtype,
        autocast_enabled=vae_autocast_enabled,
        enable_tiling=args.vae_tiling,
    )
    hr_resized_video = resize_video_frames_framewise(
        lr_decoded_video,
        target_size=(hr_height, hr_width),
        mode=args.interpolation_mode,
    )
    hr_init_latents = encode_video_to_latents(
        sampler.pipeline.vae,
        hr_resized_video,
        vae_dtype=vae_dtype,
        autocast_enabled=vae_autocast_enabled,
        enable_tiling=args.vae_tiling,
    )

    hr_predict_kwargs = {}
    if args.hr_start_mode == "continue":
        hr_predict_kwargs["start_step"] = captured_step

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
        init_latents=hr_init_latents,
        **hr_predict_kwargs,
    )

    save_outputs(
        hr_outputs,
        save_path,
        tag=f"two_stage_image_space_{args.hr_start_mode}_step{captured_step}",
    )


if __name__ == "__main__":
    main()
