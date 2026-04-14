# Two-Stage HR Video Baseline

本仓库当前收敛的 baseline 是一个基于 `HunyuanVideo 13B` 的 train-free 两阶段高分辨率视频生成流程。

固定主流程如下：

1. 在 LR 分辨率下正常采样。
2. 在中间 step `t` 截取当前 `z_t`。
3. 在同一个 `t` 上额外跑一次 denoiser，构造 `clean latent estimate`。
4. 将这个 `clean estimate` 送入 VAE 解码到图像空间。
5. 在图像空间做逐帧 spatial resize，从 LR 放大到 HR。
6. 用 VAE 编码回 HR latent。
7. 按当前 step 对应的 scheduler `sigma_t` 重新加噪。
8. 从同一个 step `t` 开始继续 HR 去噪，得到最终 HR 视频。

当前 baseline 已固定为：

- `mapping source = clean_estimate`
- `re-noise = scheduler_sigma`
- `HR resume = continue`

## 目录入口

- 主脚本：`sample_video_two_stage.py`
- 一键启动脚本：`scripts/run_two_stage.sh`
- Latent Inspection 脚本：`scripts/run_two_stage_inspect.sh`
- 兼容别名脚本：`scripts/run_two_stage_debug.sh`

## 启动前准备

在仓库根目录运行，并确保模型权重可用。默认会读取：

- `./ckpts`
- `./ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt`

如果你的权重路径不同，请在手动命令中覆盖：

- `--model-base`
- `--dit-weight`

## 快速启动

直接运行默认 baseline：

```bash
bash scripts/run_two_stage.sh
```

运行 Latent Inspection Mode：

这个模式会：

- 保存中间 latent 文件
- 打印 latent 统计信息

```bash
bash scripts/run_two_stage_inspect.sh
```

## 手动模式

推荐手动命令模板：

```bash
python3 sample_video_two_stage.py \
  --model-base ./ckpts \
  --dit-weight ./ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
  --model-resolution 720p \
  --lr-size 544 960 \
  --hr-size 720 1280 \
  --video-length 129 \
  --infer-steps 25 \
  --capture-step 15 \
  --prompt "A cat walks on the grass, realistic style." \
  --seed 42 \
  --cfg-scale 1.0 \
  --embedded-cfg-scale 6.0 \
  --flow-shift 7.0 \
  --flow-reverse \
  --use-cpu-offload \
  --save-path ./results/manual_run
```

如果要手动启用 Latent Inspection Mode：

```bash
python3 sample_video_two_stage.py \
  --model-base ./ckpts \
  --dit-weight ./ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
  --model-resolution 720p \
  --lr-size 544 960 \
  --hr-size 720 1280 \
  --video-length 129 \
  --infer-steps 25 \
  --capture-step 15 \
  --prompt "A cat walks on the grass, realistic style." \
  --seed 42 \
  --cfg-scale 1.0 \
  --embedded-cfg-scale 6.0 \
  --flow-shift 7.0 \
  --flow-reverse \
  --latent-dump-dir ./capture_latents \
  --log-latent-stats \
  --use-cpu-offload \
  --save-path ./results/manual_inspect
```

## 手动模式常用参数

### Baseline 相关参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--lr-size H W` | LR 生成分辨率 | `544 960` |
| `--hr-size H W` | HR 续采样分辨率 | `720 1280` |
| `--capture-step` | 在第几个 denoising step 切换到 HR | `25` |
| `--interpolation-mode` | 图像空间 resize 插值方式，可选 `nearest` `bilinear` `bicubic` `area` | `bilinear` |
| `--latent-dump-dir` | 保存中间 latent 文件的目录，包括 `z_t`、HR encoded latent、HR init latent | 空 |
| `--debug-latent-dir` / `--capture-dir` | `--latent-dump-dir` 的兼容别名 | 空 |
| `--capture-save-path` | 单独指定 LR capture payload 的保存路径 | 空 |
| `--log-latent-stats` | 打印中间 latent 的 shape/mean/std/min/max | 关闭 |

### 推理常用参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-base` | 模型根目录 | `ckpts` |
| `--dit-weight` | DiT 权重路径 | `ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt` |
| `--model-resolution` | 模型分辨率，可选 `540p` `720p` | `540p` |
| `--video-length` | 视频帧数，当前 3D VAE 要求 `4n+1` | `129` |
| `--infer-steps` | 推理步数 | `50` |
| `--prompt` | 正向提示词 | 无 |
| `--neg-prompt` | 负向提示词 | 默认内置 negative prompt |
| `--seed` | 随机种子 | 空 |
| `--cfg-scale` | classifier-free guidance | `1.0` |
| `--embedded-cfg-scale` | embedded guidance scale | `6.0` |
| `--flow-shift` | FlowMatch scheduler shift | `7.0` |
| `--flow-reverse` | 是否启用 reverse schedule | 关闭 |
| `--flow-solver` | FlowMatch 求解器 | `euler` |
| `--use-cpu-offload` | 是否启用 CPU offload | 关闭 |
| `--save-path` | 视频保存目录 | `./results` |
| `--save-path-suffix` | 保存目录后缀 | 空 |
| `--batch-size` | batch size | `1` |
| `--num-videos` | 每个 prompt 生成多少个视频 | `1` |
| `--disable-autocast` | 关闭 autocast | 关闭 |

### 进阶参数

这些参数平时不建议动，除非你在做显存、精度或并行实验：

| 参数 | 说明 |
| --- | --- |
| `--precision` | 主干模型精度 |
| `--vae-precision` | VAE 精度 |
| `--vae-tiling` | VAE tiling |
| `--ulysses-degree` | 并行相关参数 |
| `--ring-degree` | 并行相关参数 |
| `--use-fp8` | FP8 推理加速 |
| `--reproduce` | 尽量开启可复现设置 |

## 当前推荐配置

如果你只是想稳定复现当前 baseline，推荐先固定下面这组：

- `--lr-size 544 960`
- `--hr-size 720 1280`
- `--video-length 129`
- `--infer-steps 25`
- `--capture-step 15`
- `--cfg-scale 1.0`
- `--embedded-cfg-scale 6.0`
- `--flow-shift 7.0`
- `--flow-reverse`
- `--interpolation-mode bilinear`

## 输出内容

默认生成的视频会保存在：

- `./results/...`

如果开启 Latent Inspection Mode，或手动传入 `--latent-dump-dir`，还会额外保存：

- `*_lr_z_t.pt`
- `*_hr_encoded.pt`
- `*_hr_init_noisy.pt`

## 备注

- 如果你想看完整参数集合，可以直接阅读 `sample_video_two_stage.py` 和 `hyvideo/config.py`。
- 现阶段 README 只描述当前收敛下来的 baseline，不再覆盖已经移除的实验分叉开关。
- `scripts/run_two_stage_debug.sh` 仍然保留，但现在只是 `scripts/run_two_stage_inspect.sh` 的兼容入口。
