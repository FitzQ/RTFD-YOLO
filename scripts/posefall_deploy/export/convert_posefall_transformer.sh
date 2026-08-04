#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/developer14/anaconda3/envs/ultralytics/bin/python}"
CANN_SETENV="${CANN_SETENV:-/usr/local/Ascend/ascend-toolkit/6.10.t06spc030b110/x86_64-linux/script/setenv.sh}"
RUN_NAME="${RUN_NAME:-yolo26n-posefall-2}"
WEIGHTS="${WEIGHTS:-${ROOT}/runs/posefall/${RUN_NAME}/weights/best.pt}"
REFERENCE="${REFERENCE:-${ROOT}/exports/posefall-hi3516cv610/work/calibration/head_reference.npz}"
IMAGE_LIST="${IMAGE_LIST:-${ROOT}/exports/posefall-hi3516cv610/work/calibration/head_input_real.txt}"
OUT_DIR="${OUT_DIR:-${ROOT}/exports/posefall-hi3516cv610/work/picovision}"

mkdir -p "${OUT_DIR}/atc"

"${PYTHON}" "${ROOT}/scripts/posefall_deploy/export/export_posefall_picovision.py" \
  --extract \
  --weights "${WEIGHTS}" \
  --reference "${REFERENCE}" \
  --intermediate "${OUT_DIR}/posefall_transformer_tensors.pt"

"${PYTHON}" "${ROOT}/scripts/posefall_deploy/export/export_posefall_picovision.py" \
  --portable \
  --intermediate "${OUT_DIR}/posefall_transformer_tensors.pt" \
  --output "${OUT_DIR}/posefall_head_picovision.onnx"

set +u
source "${CANN_SETENV}"
set -u

atc \
  --framework=5 \
  --model="${OUT_DIR}/posefall_head_picovision.onnx" \
  --output="${OUT_DIR}/atc/posefall_head_picovision" \
  --input_shape="features:1,60,56" \
  --input_type="features:FP32" \
  --image_list="features:${IMAGE_LIST}" \
  --soc_version=Hi3516CV610 \
  --compile_mode=5 \
  --save_original_model=true

echo "Transformer OM: ${OUT_DIR}/atc/posefall_head_picovision_original.om"
