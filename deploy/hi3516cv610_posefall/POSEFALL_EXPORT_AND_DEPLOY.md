# PoseFall 导出与 Hi3516CV610 部署

本文只描述当前正式权重：

```text
runs/posefall/yolo26n-posefall-2/weights/best.pt
SHA256 d9bd12b969cdb3f957c2921ae5736aadec3e4915392d3bc37624d6ba41293773
```

历史权重、兼容头、普通 Transformer OM 和旧 ATC 目录均已删除，不再作为部署或回退路径。

## 目录

```text
exports/posefall-hi3516cv610/
├── README.md
├── board/                         # 只包含需要传到板端的文件
│   ├── pose_yolo_camera_yvu_fp16v.om
│   ├── posefall_head_picovision.om
│   ├── posefall_predict
│   ├── ffmpeg
│   ├── README.md
│   └── SHA256SUMS
└── work/                          # 导出和转换过程文件，不传板
    ├── onnx/
    │   └── pose_yolo.onnx
    ├── tracker/
    │   └── fall_botsort.yaml
    ├── calibration/
    └── picovision/
        ├── posefall_transformer_tensors.pt
        ├── posefall_head_picovision.onnx
        └── atc/
```

相关脚本：

```text
scripts/posefall_deploy/
├── export/
│   ├── export_posefall_onnx.py
│   ├── generate_posefall_head_calibration.py
│   ├── export_posefall_picovision.py
│   └── convert_posefall_transformer.sh
├── build/
│   └── build_ffmpeg_cv610.sh
└── validate/
    ├── audit_posefall_checkpoint_identity.py
    ├── audit_posefall_portable_equivalence.py
    ├── posefall_onnx_runtime.py
    └── validate_posefall_board_trace.py
```

## 当前模型结论

- Pose 前端来自当前权重；新旧 checkpoint 的 879 个 Pose 张量逐元素一致，因此复用已验证的摄像头 AIPP Pose OM。
- Head 是当前权重中原始三层 `PoseFallTransformer` 的 PicoVision 数学等价实现，不是 MLP。
- 输入为 `1x60x56`，保留 4 头注意力、256 维隐藏层、FFN、残差、LayerNorm、GELU、注意力池化及分类头。
- Linear/数据布局按 CV610 ATC Transformer 硬化要求改写，权重没有重新训练、蒸馏或替换。

最终 Head 在 64 个真实窗口上的转换一致率为 100%，板端 normal/fall 烟测输出分别约为 `0.000230789/1.000000`。

## 导出流程

以下命令仅用于以后更换权重；整理现有文件时没有重新运行。

```bash
cd /developer14/hefei/ultralytics
export PYTHON=/developer14/anaconda3/bin/python
export OUT=exports/posefall-hi3516cv610/work
```

导出 ONNX：

```bash
$PYTHON scripts/posefall_deploy/export/export_posefall_onnx.py \
  --weights runs/posefall/yolo26n-posefall-2/weights/best.pt \
  --out-dir "$OUT/onnx" \
  --imgsz 640 \
  --opset 13 \
  --device cpu \
  --svp
```

生成真实 Head 校准窗口：

```bash
$PYTHON scripts/posefall_deploy/export/generate_posefall_head_calibration.py \
  --cache-root runs/posefall/features/yolo26n-pose-posefall-5abce169acd4 \
  --output "$OUT/calibration/head_input_real.txt" \
  --reference "$OUT/calibration/head_reference.npz" \
  --sample-bin "$OUT/calibration/head_sample.bin" \
  --samples-per-class 32 \
  --seed 610
```

导出 PicoVision Head 并调用 ATC：

```bash
scripts/posefall_deploy/export/convert_posefall_transformer.sh
```

最终 ATC Head 位于：

```text
exports/posefall-hi3516cv610/work/picovision/atc/posefall_head_picovision_original.om
```

将它复制并重命名到 `board/posefall_head_picovision.om`。摄像头 Pose OM 必须使用已验证的静态 AIPP，不能用普通 NCHW Pose OM 替代。

## 上板

只复制 `board/` 中的四个二进制文件：

```sh
mkdir -p /root/posefall
```

```text
pose_yolo_camera_yvu_fp16v.om
posefall_head_picovision.om
posefall_predict
ffmpeg
```

设置权限：

```sh
chmod +x /root/posefall/posefall_predict /root/posefall/ffmpeg
```

视频：

```sh
POSEFALL_FFMPEG=/root/posefall/ffmpeg \
/root/posefall/posefall_predict \
  --source /mnt/usb/test.mp4 \
  --pose /root/posefall/pose_yolo_camera_yvu_fp16v.om \
  --head /root/posefall/posefall_head_picovision.om \
  --trace /mnt/usb/test_trace.csv
```

容器 FPS 自动读取。板上 Linux 内存约 33 MiB，建议使用不超过 640p 的视频，并把视频放在 USB/SD 存储中。

摄像头：

```sh
/root/posefall/posefall_predict \
  --source camera \
  --source-fps 30 \
  --pose /root/posefall/pose_yolo_camera_yvu_fp16v.om \
  --head /root/posefall/posefall_head_picovision.om
```

`--source 0` 与 `--source camera` 等价。摄像头模式要求相同固件中的 `sample_venc 0` 已建立 VI/ISP/VPSS；`posefall_predict` 再创建 AI channel 2。

## 验证

校验板端跟踪 trace：

```bash
$PYTHON scripts/posefall_deploy/validate/validate_posefall_board_trace.py \
  /path/to/trace.csv \
  --box-atol 0.01
```

当前实测结果：

```text
PASS: IDs agree and boxes are within 0.01 px
```

默认日志一帧一行：

```text
frame 123: (no detections), 80.12ms
frame 124: 1 person (nofall), 84.31ms
frame 125: 3 persons (nofall, FALL, nofall), 91.72ms
```
