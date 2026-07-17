#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${ROOT}/.." && pwd)"
CANN_SETENV="${CANN_SETENV:-/usr/local/Ascend/ascend-toolkit/svp_latest/x86_64-linux/script/setenv.sh}"
if [[ ! -f "${CANN_SETENV}" ]]; then
  echo "CANN setenv.sh not found: ${CANN_SETENV}" >&2
  echo "Set CANN_SETENV to the installed SVP CANN setenv.sh path." >&2
  exit 1
fi
set +u
source "${CANN_SETENV}"
set -u
cd "${ROOT}"

atc \
  --framework=5 \
  --model="${PACKAGE_ROOT}/pose_yolo.onnx" \
  --output="${ROOT}/model/pose_yolo" \
  --input_shape="images:1,3,640,640" \
  --input_type="images:FP32" \
  --image_list="images:${ROOT}/calibration/pose_input.txt" \
  --soc_version=Hi3516CV610 \
  --save_original_model=true

atc \
  --framework=5 \
  --model="${PACKAGE_ROOT}/posefall_head.onnx" \
  --output="${ROOT}/model/posefall_head" \
  --input_shape="features:1,60,56" \
  --input_type="features:FP32" \
  --image_list="features:${ROOT}/calibration/head_input.txt" \
  --soc_version=Hi3516CV610 \
  --save_original_model=true

echo "SVP OM models saved under ${ROOT}/model"
