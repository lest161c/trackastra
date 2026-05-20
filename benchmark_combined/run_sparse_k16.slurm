#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=tr_sparse_k16
#SBATCH --output=logs/slurm-sparse_k16-%j.out
#SBATCH --error=logs/slurm-sparse_k16-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G

export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export WANDB_DISABLED=true
export WANDB_MODE=disabled
unset PYTHONHOME

eval "$(conda shell.bash hook)"
TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
ENV_DIR="$TRK/trackastra_env"

# --- Quick test mode: sbatch --export=TEST=1 run_baseline.slurm ---
if [ "$TEST" = "1" ]; then
    echo "=== TEST MODE: env setup only ==="
    conda activate "$ENV_DIR" 2>/dev/null || {
        echo "Creating env..."
        conda create --prefix "$ENV_DIR" python=3.10 -y
        conda activate "$ENV_DIR"
    }
    python -m pip install setuptools
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
    cd "$TRK"
    git pull origin feature/sparse-attention-gather
    python -m pip install -e .
    python -c "import trackastra; from trackastra.model import TrackingTransformer; print('trackastra OK:', TrackingTransformer); import torch; print('CUDA:', torch.cuda.is_available())"
    echo "=== TEST PASSED ==="
    exit 0
fi

# --- Normal mode ---
conda activate "$ENV_DIR" 2>/dev/null || {
    echo "Creating env..."
    conda create --prefix "$ENV_DIR" python=3.10 -y
    conda activate "$ENV_DIR"
}
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export WANDB_DISABLED=true
export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .
mkdir -p logs

python scripts/train.py \
    --name sparse_k16 --ndim 2 --window 10 --epochs 100 --warmup_epochs 10 \
    --detection_folders TRA \
    --input_train /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/rpsM /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/recA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/pheA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/metA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/cib /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/trpL \
    --input_val /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/rpsM /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/recA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/pheA \
    --outdir runs --distributed False \
    --batch_size 48 --max_tokens 4096 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.01 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff 256 \
    --delta_cutoff 2 --knn_neighbors 16 \
    --mixedp True --num_workers 8 --seed 42 --augment 3
