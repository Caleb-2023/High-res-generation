from typing import Tuple

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
