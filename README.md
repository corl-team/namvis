<div align="center">

# NAMVIS

### Next-Scale Autoregressive Multi-View Image Synthesis

Ramil Khafizov<sup>1</sup>, Ilya Statsenko<sup>2</sup>, Ruslan Rakhimov<sup>2</sup>,
Artem Komarichev<sup>1</sup>, Peter Wonka<sup>3</sup>, Evgeny Burnaev<sup>1,4</sup>

<sup>1</sup> Applied AI Institute · <sup>2</sup> T-Tech · <sup>3</sup> KAUST · <sup>4</sup> AXXX

*NeurIPS 2026*

**[📄 arXiv](https://arxiv.org/abs/2610.04722) | [🌐 Project Page](https://corl-team.github.io/namvis/) | [🤗 Checkpoints](https://huggingface.co/smileyenot983/NAMVIS) | [🗂️ Dataset Part 1](https://huggingface.co/datasets/smileyenot983/objaversexl_sketchfab_pmap) | [🗂️ Dataset Part 2](https://huggingface.co/datasets/smileyenot983/objaversexl_github6.5_pmap) | [🐦 Twitter thread](https://x.com/RamilKhafizov11/status/2108127219910377983)**

</div>

## Abstract

Sparse-view novel view synthesis is a central problem in 3D content creation, but diffusion-based approaches remain limited by iterative denoising, making multi-view generation expensive at inference time. We introduce NAMVIS, a diffusion-free framework that reformulates multi-view image synthesis as geometry-conditioned next-scale autoregression. Instead of generating target views through repeated denoising, NAMVIS predicts discrete visual tokens through a small number of coarse-to-fine scale steps, while sampling all tokens within each scale and across target views in parallel. To anchor this generation process to explicit camera geometry, we propose Multi-scale Projective Pose Encoding, which injects source and target camera transformations into both target-view self-attention and source-to-target cross-attention at every resolution. NAMVIS further combines global conditioning with dense geometry-aware cross-attention, enabling the model to preserve source-view appearance while maintaining target-view consistency. Across Objaverse, GSO, and OmniObject3D, NAMVIS outperforms diffusion-based baselines in PSNR, SSIM, and LPIPS, while running over 3× faster than the evaluated diffusion baselines under the same evaluation setting. These results suggest that geometry-conditioned next-scale autoregression is a promising and efficient alternative to diffusion for sparse-view multi-view synthesis.

![NAMVIS architecture](assets/architecture.png)

This repository contains the training, inference, and evaluation code for the
released 1B model.

## 🛠️ Installation

Requirements: Linux x86_64, an NVIDIA GPU supported by
[FlashAttention-2](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3#nvidia-cuda-support)
(Ampere or newer), and a driver for CUDA 12.1 or later. There is no CPU path.

The environment is managed with [uv](https://docs.astral.sh/uv/) and pinned in
`uv.lock` (Python 3.10, PyTorch 2.5.1 + CUDA 12.1, FlashAttention 2.8.3.post1
from its prebuilt wheel, so nothing is compiled):

```bash
git clone https://github.com/corl-team/namvis.git && cd namvis
uv sync
```

Prefix the commands below with `uv run`, or activate the environment with
`source .venv/bin/activate`.

<details>
<summary>Docker</summary>

The image contains only the locked environment; the repository is mounted at
run time, so weights, data, and outputs stay in your checkout. Install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then:

```bash
docker build -t namvis .
docker run --rm -it --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$PWD":/workspace/namvis namvis bash
```

Inside the container, run the commands below without `uv run`. Use
`--gpus '"device=0,1"'` to select GPUs, and add
`-v /path/to/data:/workspace/namvis/data` if the training data lives elsewhere.

</details>

## 📦 Checkpoints

The [Hugging Face repository](https://huggingface.co/smileyenot983/NAMVIS)
holds the NAMVIS transformer and the matching visual tokenizer:

```bash
uv run hf download smileyenot983/NAMVIS namvis_1b.pth infinity_vae_d32reg.pth \
  --revision ba8f58ee46a7077af80dea4621239c9389435518 --local-dir weights
sha256sum --check <<'EOF'
716b78418826e23e9f6498e89752370d7b6c7d7d79236f0b19a4549ce0a56247  weights/namvis_1b.pth
7a37fa3ea1b2a1ebd23de61d91a5e68202825e5a67edaef4b7c55f5fd5b9cf26  weights/infinity_vae_d32reg.pth
EOF
```

## 🚀 Inference

`data_eval/rendered_objaverse8_wdepth_pitch30/` contains 30 rendered Objaverse
scenes with eight views each. Each scene has a `transforms.json` with Blender
camera-to-world poses; view indices are positions in its `frames` list.

Generate three target views from two source views of one scene:

```bash
uv run python inference/infer_ext.py \
  --data_path=data_eval/rendered_objaverse8_wdepth_pitch30 \
  --model_path=weights/namvis_1b.pth --vae_path=weights/infinity_vae_d32reg.pth \
  --N_views_src=2 --N_views_tgt=3 --src_indices 0 3 --tgt_indices 1 2 4 \
  --cfg=1 --tau=0.5 --seed=0 --max_scenes=1 \
  --out_dir=inference_results/example --grid_out_dir=inference_grid/example
uv run python inference/calc_metric.py --root=inference_results/example
```

Each scene folder gets `input/`, `gt/`, `gen/`, and `grid_comparison.png`
(rows: source, generated, ground truth). `calc_metric.py` reports pixel MSE,
LPIPS (AlexNet), SSIM, and PSNR.

To evaluate every combination of one to three source and one to three target
views on all bundled scenes:

```bash
uv run bash inference/infer_batched.sh
```

It uses source views `0 3 7` and target views `1 2 4 5 6`, taking the first N
of each, and writes to `inference_results_namvis256/` and
`inference_grid_namvis256/`. Expected averages over the 30 scenes (seed 0,
one H100):

| Source views | Target views | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|
| 1 | 1 | 0.1016 | 0.8551 | 21.77 |
| 1 | 2 | 0.1126 | 0.8502 | 21.25 |
| 1 | 3 | 0.1089 | 0.8520 | 21.45 |
| 2 | 1 | 0.0688 | 0.8742 | 23.79 |
| 2 | 2 | 0.0632 | 0.8785 | 24.08 |
| 2 | 3 | 0.0618 | 0.8821 | 24.35 |
| 3 | 1 | 0.0642 | 0.8769 | 24.09 |
| 3 | 2 | 0.0617 | 0.8795 | 24.14 |
| 3 | 3 | 0.0607 | 0.8838 | 24.42 |

Use them to check an installation. They do not match the paper's Objaverse
results exactly, because the paper's evaluation renders may differ slightly
from the bundled ones.

All examples use `--cfg=1`, i.e. no classifier-free guidance, which worked
better than guidance in the authors' evaluations.

## 🏋️ Training

Training reads WebDataset TAR shards in which each sample holds the view images
and a JSON with camera metadata (see
[infinity/dataset/webdataset_utils.py](infinity/dataset/webdataset_utils.py)).
The released training data is on Hugging Face in two parts, both rendered from
Objaverse-XL objects:
[Part 1](https://huggingface.co/datasets/smileyenot983/objaversexl_sketchfab_pmap),
objects from the Sketchfab source (1.1 TB), and
[Part 2](https://huggingface.co/datasets/smileyenot983/objaversexl_github6.5_pmap),
objects from the GitHub source with an object-quality score above 6.5 (0.3 TB).
For a first run, a single shard (162 scenes) is enough:

```bash
uv run hf download smileyenot983/objaversexl_sketchfab_pmap \
  objaversexl_sketchfab-w00-000000.tar --repo-type dataset --local-dir data/sketchfab
```

`train_namvis.sh` is an example for fine-tuning from the released checkpoint;
it does not reproduce the paper's training schedule. Launch it on one node with
two GPUs:

```bash
DATA_PATH=data/sketchfab TRAIN_SCENES=162 NPROC_PER_NODE=2 RUN_NAME=namvis_1b \
  uv run bash train_namvis.sh
```

`DATA_PATH` accepts comma-separated shard directories, shard files, or brace
patterns. Set `TRAIN_SCENES` to the number of scenes in your data (default:
218186, the full release); it defines the epoch length. Set `RUSH_RESUME=` (empty)
to train the transformer from scratch; the visual tokenizer is always required.
For multiple nodes, run the launcher on each node with the same `NNODES`,
`MASTER_ADDR`, and `MASTER_PORT` and a distinct `NODE_RANK`.

Outputs:

- `checkpoints/<RUN_NAME>/`: full `*-last.pth` checkpoints, used for automatic
  resume when the launcher is rerun, and weights-only `*-statedict.pth`
  exports, which `--model_path` and `RUSH_RESUME` accept.
- `outputs_<RUN_NAME>/evaluation/`: metrics and grids from the rendered-scene
  evaluation that runs before every epoch; see
  [data_eval/README.md](data_eval/README.md).
- `trackio_logs/`: training curves
  (`TRACKIO_DIR=trackio_logs uv run trackio show --project namvis`).

## 🙏 Acknowledgements

The implementation builds on [Infinity](https://github.com/FoundationVision/Infinity).

## 📚 Citation

```bibtex
@inproceedings{khafizov2026namvis,
  title         = {{NAMVIS}: Next-Scale Autoregressive Multi-View Image Synthesis},
  author        = {Khafizov, Ramil and Statsenko, Ilya and Rakhimov, Ruslan and Komarichev, Artem and Wonka, Peter and Burnaev, Evgeny},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2026},
  eprint        = {2610.04722},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.04722}
}
```

## 📄 License

The code is released under the [MIT License](LICENSE), including the upstream
Infinity copyright notice. Datasets and third-party assets retain their
respective licenses.
