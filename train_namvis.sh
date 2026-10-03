#!/usr/bin/env bash

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Use one worker per selected GPU; set NNODES and NODE_RANK for multiple nodes.
nnodes="${NNODES:-1}"
node_rank="${NODE_RANK:-0}"
nproc_per_node="${NPROC_PER_NODE:-1}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-6661}"

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

# set up envs
export OMP_NUM_THREADS=8
export TRACKIO_DIR="${REPO_ROOT}/trackio_logs"

BED=checkpoints
LOCAL_OUT=local_output
mkdir -p $BED
mkdir -p $LOCAL_OUT

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export RUN_NAME="${RUN_NAME:-namvis_1b}"

export WANDB_MODE="${WANDB_MODE:-offline}"
exp_name="${RUN_NAME}"
bed_path=checkpoints/${RUN_NAME}/

# Comma-separated shard directories, shard files, or brace patterns.
export data_path="${DATA_PATH:?Set DATA_PATH to your WebDataset training shards}"
export eval_path="${EVAL_DATA_PATH:-data_eval/rendered_objaverse8_wdepth_pitch30}"

export eval_json="${EVAL_JSON:-data_eval/eval_objaverse_small.json}"


local_out_path=$LOCAL_OUT/${RUN_NAME}

out_dir=outputs_${RUN_NAME}


mkdir -p "${out_dir}" "${bed_path}" "${local_out_path}"

echo "Shell: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}'"
echo "Shell: CUDA_DEVICE_ORDER='${CUDA_DEVICE_ORDER:-<unset>}'"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv

torchrun \
--nproc_per_node="${nproc_per_node}" \
--nnodes="${nnodes}" \
--node_rank="${node_rank}" \
--master_addr="${master_addr}" \
--master_port="${master_port}" \
train.py \
--ep=10000 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=0 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--bed=${bed_path} \
--data_path=${data_path} \
--eval_path="${eval_path}" \
--eval_json="${eval_json}" \
--eval_backend=rendered \
--eval_max_scenes="${EVAL_MAX_SCENES:-100}" \
--eval_cfg=1.0 \
--eval_tau=0.5 \
--eval_seed=0 \
--eval_freq="${EVAL_FREQ:-1}" \
--train_scenes="${TRAIN_SCENES:-218186}" \
--exp_name=${exp_name} \
--tblr=6e-3 \
--pn 0.06M \
--model=1bc8 \
--lbs=2 \
--workers=1 \
--short_cap_prob 0.5 \
--online_t5=0 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 30000 \
--Ct5=2048 \
--t5_path="${T5_PATH:-weights/flan_t5}" \
--vae_type 32 \
--vae_ckpt="${VAE_CKPT:-weights/infinity_vae_d32reg.pth}" \
--rush_resume="${RUSH_RESUME-weights/namvis_1b.pth}" \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 5000 \
--log_freq=50 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--apply_spatial_patchify 0 \
--use_flex_attn=False \
--pad=0 \
--multiview=True \
--src2sos=True \
--N_views_src=2 \
--N_views_tgt=3 \
--out_dir=${out_dir} \
--run_name=${RUN_NAME} \
--use_prope=True \
--logger_name="trackio" \
"$@"
