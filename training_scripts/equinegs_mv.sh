#!/usr/bin/env bash
# Usage:
#   bash training_scripts/equinegs_mv.sh [exp_name] [num_gpus]
#
# Examples:
#   bash training_scripts/equinegs_mv.sh equinegs_mv_8gpu 8
#   bash training_scripts/equinegs_mv.sh equinegs_mv_4gpu 4
#   NUM_GPUS=4 bash training_scripts/equinegs_mv.sh equinegs_mv_4gpu
#
# Defaults:
#   exp_name: 4DEquinePP_mv
#   num_gpus: ${NUM_GPUS:-8}
#
# The script keeps the effective global batch close to the 8-GPU baseline by
# setting trainer.accumulate_grad_batches = ceil(8 * 8 / num_gpus).

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=headless

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_EXT_DIR="${REPO_ROOT}/.local_ext"
if [[ -d "${LOCAL_EXT_DIR}" ]]; then
  export PYTHONPATH="${LOCAL_EXT_DIR}:${PYTHONPATH}"
fi

EXP_NAME=${1:-4DEquinePP_mv}
NUM_GPUS=${2:-${NUM_GPUS:-8}}

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
  exit 1
fi

BASE_NUM_GPUS=${BASE_NUM_GPUS:-8}
BASE_ACCUMULATE_GRAD_BATCHES=${BASE_ACCUMULATE_GRAD_BATCHES:-1}
TARGET_ACCUMULATION_FACTOR=$((BASE_NUM_GPUS * BASE_ACCUMULATE_GRAD_BATCHES))
ACCUMULATE_GRAD_BATCHES=$(((TARGET_ACCUMULATION_FACTOR + NUM_GPUS - 1) / NUM_GPUS))

DATASET_SIZE=168318
TRAIN_BATCH_SIZE=4
EPOCHS_TO_TRAIN=6
SAMPLES_PER_RANK=$(((DATASET_SIZE + NUM_GPUS - 1) / NUM_GPUS))
TRAIN_BATCHES_PER_EPOCH=$((SAMPLES_PER_RANK / TRAIN_BATCH_SIZE))
OPT_STEPS_PER_EPOCH=$(((TRAIN_BATCHES_PER_EPOCH + ACCUMULATE_GRAD_BATCHES - 1) / ACCUMULATE_GRAD_BATCHES))
TOTAL_STEPS=$((OPT_STEPS_PER_EPOCH * EPOCHS_TO_TRAIN))
VAL_STEPS=${TRAIN_BATCHES_PER_EPOCH}
LOG_STEPS=$((OPT_STEPS_PER_EPOCH / 10))
if [[ "${LOG_STEPS}" -lt 50 ]]; then
  LOG_STEPS=50
fi

echo "exp_name=${EXP_NAME}"
echo "trainer.devices=${NUM_GPUS}"
echo "trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
echo "dataset_size=${DATASET_SIZE}"
echo "train_batches_per_epoch=${TRAIN_BATCHES_PER_EPOCH}"
echo "optimizer_steps_per_epoch=${OPT_STEPS_PER_EPOCH}"
echo "GENERAL.TOTAL_STEPS=${TOTAL_STEPS}"
echo "GENERAL.VAL_STEPS=${VAL_STEPS}"
echo "GENERAL.LOG_STEPS=${LOG_STEPS}"

python main_avatar.py exp_name=${EXP_NAME} \
       experiment=hrm_mv trainer=gpu_avatar launcher=local \
       trainer=ddp trainer.devices=${NUM_GPUS} \
       trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
       GENERAL.TOTAL_STEPS=${TOTAL_STEPS} \
       GENERAL.VAL_STEPS=${VAL_STEPS} \
       GENERAL.LOG_STEPS=${LOG_STEPS} \
       WANDB.MODE=offline
