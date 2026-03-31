from typing import Optional, Tuple

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
