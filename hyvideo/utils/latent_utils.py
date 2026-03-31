from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def interpolate_spatial_latents_framewise(
    latents: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize latent spatial dimensions frame by frame without mixing time."""
    if latents.ndim != 5:
        raise ValueError(
            f"`latents` must have shape (batch, channels, frames, height, width), got {tuple(latents.shape)}."
        )

    target_height, target_width = target_size
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"`target_size` must be positive, got {target_size}.")

    batch, channels, frames, _, _ = latents.shape
    frame_latents = latents.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, latents.shape[-2], latents.shape[-1])

    needs_upcast = frame_latents.device.type == "cpu" and frame_latents.dtype == torch.bfloat16
    if needs_upcast:
        frame_latents = frame_latents.float()

    interpolate_kwargs = {"size": (target_height, target_width), "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        interpolate_kwargs["align_corners"] = False

    resized = F.interpolate(frame_latents, **interpolate_kwargs)
    if needs_upcast:
        resized = resized.to(latents.dtype)

    return resized.reshape(batch, frames, channels, target_height, target_width).permute(0, 2, 1, 3, 4)


def resize_video_frames_framewise(
    video: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize decoded video frames frame by frame in pixel space without mixing time."""
    if video.ndim != 5:
        raise ValueError(
            f"`video` must have shape (batch, channels, frames, height, width), got {tuple(video.shape)}."
        )

    target_height, target_width = target_size
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"`target_size` must be positive, got {target_size}.")

    batch, channels, frames, _, _ = video.shape
    frame_video = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, video.shape[-2], video.shape[-1])

    needs_upcast = frame_video.device.type == "cpu" and frame_video.dtype == torch.bfloat16
    if needs_upcast:
        frame_video = frame_video.float()

    interpolate_kwargs = {"size": (target_height, target_width), "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        interpolate_kwargs["align_corners"] = False

    resized = F.interpolate(frame_video, **interpolate_kwargs)
    if needs_upcast:
        resized = resized.to(video.dtype)

    return resized.reshape(batch, frames, channels, target_height, target_width).permute(0, 2, 1, 3, 4)


def decode_latents_to_video(
    vae,
    latents: torch.Tensor,
    vae_dtype: torch.dtype,
    autocast_enabled: bool,
    enable_tiling: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Decode diffusion latents into VAE pixel space video in [-1, 1]."""
    scaled_latents = latents / vae.config.scaling_factor
    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
        scaled_latents = scaled_latents + vae.config.shift_factor

    if enable_tiling:
        vae.enable_tiling()

    with torch.autocast(
        device_type=latents.device.type,
        dtype=vae_dtype,
        enabled=autocast_enabled and latents.device.type == "cuda",
    ):
        return vae.decode(scaled_latents, return_dict=False, generator=generator)[0]


def encode_video_to_latents(
    vae,
    video: torch.Tensor,
    vae_dtype: torch.dtype,
    autocast_enabled: bool,
    sample_posterior: bool = False,
    enable_tiling: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Encode a decoded video tensor in [-1, 1] back to diffusion latents."""
    if enable_tiling:
        vae.enable_tiling()

    with torch.autocast(
        device_type=video.device.type,
        dtype=vae_dtype,
        enabled=autocast_enabled and video.device.type == "cuda",
    ):
        posterior = vae.encode(video).latent_dist
        vae_latents = (
            posterior.sample(generator=generator)
            if sample_posterior
            else posterior.mode()
        )

    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
        vae_latents = vae_latents - vae.config.shift_factor

    return vae_latents * vae.config.scaling_factor


def compute_step_ratio_mix_scales(
    capture_step: int,
    total_steps: int,
) -> Tuple[float, float]:
    """Compute the signal/noise blend weights for HR resume latents.

    The default train-free baseline follows the user's step-ratio rule:
    signal_scale = capture_step / total_steps
    noise_scale = (total_steps - capture_step) / total_steps
    """
    if total_steps <= 0:
        raise ValueError(f"`total_steps` must be positive, got {total_steps}.")
    if capture_step < 0 or capture_step >= total_steps:
        raise ValueError(
            f"`capture_step` must be in [0, {total_steps - 1}], got {capture_step}."
        )

    signal_scale = float(capture_step) / float(total_steps)
    noise_scale = 1.0 - signal_scale
    return signal_scale, noise_scale


def sample_noise_like(
    latents: torch.Tensor,
    seeds: Optional[Sequence[int]] = None,
    seed_offset: int = 0,
) -> torch.Tensor:
    """Sample deterministic per-sample Gaussian noise matching the latent shape."""
    sample_dtype = latents.dtype
    if latents.device.type == "cpu" and sample_dtype == torch.bfloat16:
        sample_dtype = torch.float32

    if seeds is None:
        noise = torch.randn(latents.shape, device=latents.device, dtype=sample_dtype)
        if noise.dtype != latents.dtype:
            noise = noise.to(latents.dtype)
        return noise

    if len(seeds) != latents.shape[0]:
        raise ValueError(
            f"`seeds` must have length {latents.shape[0]}, got {len(seeds)}."
        )

    generator_device = latents.device.type

    noise_batches = []
    for batch_idx, seed in enumerate(seeds):
        generator = torch.Generator(device=generator_device).manual_seed(
            int(seed) + int(seed_offset)
        )
        noise = torch.randn(
            latents[batch_idx : batch_idx + 1].shape,
            generator=generator,
            device=latents.device,
            dtype=sample_dtype,
        )
        if noise.dtype != latents.dtype:
            noise = noise.to(latents.dtype)
        noise_batches.append(noise)

    return torch.cat(noise_batches, dim=0)


def blend_latents_with_noise(
    latents: torch.Tensor,
    noise: torch.Tensor,
    signal_scale: float,
    noise_scale: float,
) -> torch.Tensor:
    """Blend encoded HR latents with Gaussian noise using explicit scalar weights."""
    if latents.shape != noise.shape:
        raise ValueError(
            f"`latents` and `noise` must have the same shape, got {tuple(latents.shape)} and {tuple(noise.shape)}."
        )

    return latents * signal_scale + noise * noise_scale


def renoise_latents_with_step_ratio(
    latents: torch.Tensor,
    capture_step: int,
    total_steps: int,
    seeds: Optional[Sequence[int]] = None,
    seed_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Add step-consistent noise to the mapped HR latent before resuming denoising."""
    signal_scale, noise_scale = compute_step_ratio_mix_scales(
        capture_step=capture_step,
        total_steps=total_steps,
    )
    noise = sample_noise_like(latents, seeds=seeds, seed_offset=seed_offset)
    mixed_latents = blend_latents_with_noise(
        latents,
        noise,
        signal_scale=signal_scale,
        noise_scale=noise_scale,
    )
    return mixed_latents, noise, signal_scale, noise_scale
