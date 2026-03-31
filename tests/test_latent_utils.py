import torch

from hyvideo.utils.latent_utils import (
    compute_step_ratio_mix_scales,
    flowmatch_clean_latent_estimate,
    interpolate_spatial_latents_framewise,
    renoise_latents_with_step_ratio,
    resize_video_frames_framewise,
    sample_noise_like,
)


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


def test_resize_video_frames_framewise_matches_per_frame_interpolation():
    video = torch.randn(1, 3, 2, 4, 4)

    resized = resize_video_frames_framewise(video, target_size=(6, 8))

    expected_frames = []
    for frame_idx in range(video.shape[2]):
        expected_frames.append(
            torch.nn.functional.interpolate(
                video[:, :, frame_idx],
                size=(6, 8),
                mode="bilinear",
                align_corners=False,
            )
        )
    expected = torch.stack(expected_frames, dim=2)

    torch.testing.assert_close(resized, expected)


def test_compute_step_ratio_mix_scales_matches_expected_example():
    signal_scale, noise_scale = compute_step_ratio_mix_scales(
        capture_step=10,
        total_steps=25,
    )

    assert signal_scale == 10 / 25
    assert noise_scale == 15 / 25


def test_renoise_latents_with_step_ratio_matches_linear_blend_rule():
    latents = torch.ones(2, 1, 1, 2, 2)
    seeds = [123, 456]

    mixed, noise, signal_scale, noise_scale = renoise_latents_with_step_ratio(
        latents,
        capture_step=2,
        total_steps=4,
        seeds=seeds,
    )

    expected_noise = sample_noise_like(latents, seeds=seeds)
    expected_mixed = latents * 0.5 + expected_noise * 0.5

    assert signal_scale == 0.5
    assert noise_scale == 0.5
    torch.testing.assert_close(noise, expected_noise)
    torch.testing.assert_close(mixed, expected_mixed)


def test_flowmatch_clean_latent_estimate_uses_xt_minus_sigma_u():
    noisy_latents = torch.full((1, 1, 1, 2, 2), 3.0)
    model_output = torch.full((1, 1, 1, 2, 2), 2.0)

    clean = flowmatch_clean_latent_estimate(
        noisy_latents,
        model_output,
        sigma=0.25,
    )

    expected = torch.full((1, 1, 1, 2, 2), 2.5)
    torch.testing.assert_close(clean, expected)
