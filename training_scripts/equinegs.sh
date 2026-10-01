export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=headless

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_EXT_DIR="${REPO_ROOT}/.local_ext"
if [[ -d "${LOCAL_EXT_DIR}" ]]; then
  export PYTHONPATH="${LOCAL_EXT_DIR}:${PYTHONPATH}"
fi

python main_avatar.py exp_name=${1:-4DEquinePP} experiment=hrm trainer=gpu_avatar launcher=local \
       trainer=ddp trainer.devices=4 \
       GENERAL.VAL_STEPS=10000 \
       TRAIN.BATCH_SIZE=2 \
       WANDB.MODE=online
