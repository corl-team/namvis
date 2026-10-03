#!/usr/bin/env bash
# Evaluate every combination of 1-3 source and 1-3 target views on rendered scenes.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-weights/namvis_1b.pth}"
VAE_PATH="${VAE_PATH:-weights/infinity_vae_d32reg.pth}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data_eval/rendered_objaverse8_wdepth_pitch30}"
RESULTS_ROOT="${RESULTS_ROOT:-inference_results_namvis256}"
GRID_ROOT="${GRID_ROOT:-inference_grid_namvis256}"

MAX_VIEWS_SRC=3
MAX_VIEWS_TGT=3
# Each run takes the first N entries of these lists.
SRC_MASTER_INDICES=(0 3 7)
TGT_MASTER_INDICES=(1 2 4 5 6)

for (( src=1; src<=MAX_VIEWS_SRC; src++ )); do
    for (( tgt=1; tgt<=MAX_VIEWS_TGT; tgt++ )); do
        echo "=================================================="
        echo "Running inference with N_views_src=${src} and N_views_tgt=${tgt}"
        echo "=================================================="

        out_dir="${RESULTS_ROOT}/src${src}_tgt${tgt}/objaverse"
        grid_dir="${GRID_ROOT}/src${src}_tgt${tgt}/objaverse"

        python inference/infer_ext.py \
            --data_path="${EVAL_DATA_PATH}" \
            --model_path="${MODEL_PATH}" \
            --vae_path="${VAE_PATH}" \
            --model=1b --use_prope=1 --cos=0 --pn=0.06M \
            --N_views_src="${src}" \
            --N_views_tgt="${tgt}" \
            --src_indices "${SRC_MASTER_INDICES[@]}" \
            --tgt_indices "${TGT_MASTER_INDICES[@]}" \
            --cfg=1 --tau=0.5 --seed=0 \
            --out_dir="${out_dir}" \
            --grid_out_dir="${grid_dir}"

        echo "Calculating metrics for src=${src}, tgt=${tgt}..."
        python inference/calc_metric.py --root="${out_dir}"
    done
done

echo "All view combinations completed!"
