import torch

from hyvideo.utils.latent_utils import interpolate_spatial_latents_framewise


def test_interpolate_spatial_latents_framewise_preserves_batch_channel_time():
    latents = torch.randn(2, 4, 3, 5, 7)

    resized = interpolate_spatial_latents_framewise(latents, target_size=(9, 11))

    assert resized.shape == (2, 4, 3, 9, 11)


def test_interpolate_spatial_latents_framewise_matches_per_frame_interpolation():
    latents = torch.randn(1, 2, 2, 4, 4)

    resized = interpolate_spatial_latents_framewise(latents, target_size=(6, 8))

    expected_frames = []
    for frame_idx in range(latents.shape[2]):
        expected_frames.append(
            torch.nn.functional.interpolate(
                latents[:, :, frame_idx],
                size=(6, 8),
                mode="bilinear",
                align_corners=False,
            )
        )
    expected = torch.stack(expected_frames, dim=2)

    torch.testing.assert_close(resized, expected)
