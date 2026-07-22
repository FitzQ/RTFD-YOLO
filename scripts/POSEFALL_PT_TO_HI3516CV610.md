# PoseFall 原始 PT 到 Hi3516CV610 板端部署

本文记录 `/developer14/hefei/ultralytics` 中完整 PoseFall 权重从 `.pt` 导出、Transformer 硬化、ATC 编译、板端验证到实时运行的完整流程。

> **V1.3验证状态（2026-07-22）**：主结果权重 `runs/posefall/yolo26n-posefall-2/weights/best.pt`（SHA256 `d9bd12b969cdb3f957c2921ae5736aadec3e4915392d3bc37624d6ba41293773`）已完成Transformer Head等价硬化、ATC编译、板端烟测与空场景摄像头闭环。新旧checkpoint的879个Pose前端张量逐元素一致，因此复用已验证的摄像头Pose OM；旧 `train-6/best.pt` 数据只作历史比较。

## 1. 当前已验证结论

当前板端运行的不是 MLP 替代模型，而是 `.pt` 中原始 `PoseFallTransformer` 的 PicoVision 等价实现：

- 输入仍为 `1 x 60 x 56` 的姿态时序特征；
- 保留 3 层 Transformer Encoder；
- 保留 4 头自注意力和 256 维隐藏特征；
- 保留 FFN、残差连接、LayerNorm、GELU；
- 保留学习式注意力池化和最终分类头；
- 原始 Q/K/V、输出投影、FFN、池化和分类权重均逐项复制，没有重新训练或蒸馏；
- 仅按 CV610 的硬化要求把 NLC 数据流改为 CHW，并把 Linear 等价改写为 1x1 Conv。

新主结果权重在64个真实校准窗口上的转换前后验证结果：

```text
CPU max_abs=8.94069671631e-7, mean_abs=2.57468855125e-8
CUDA max_abs=7.89761543274e-7, mean_abs=2.49015101872e-8
class_agreement=1.000000
```

板端测试结果：

```text
normal sample: PyTorch=0.000213212, CV610=0.000230789
fall sample:   PyTorch=0.999946475, CV610=1.000000
Transformer head, 200 runs after 5 warm-ups:
  normal mean=4.159966 ms, fall mean=4.166595 ms
```

当前生效文件为Pose OM 4,238,920B（SHA256 `f71e4a264f22f055e49a95b66ecc5602274ed636c390263a9d54f45cfbddc2db`）与新Head OM 3,156,830B（SHA256 `85d26e3f179153d7e28457f34881fd9a3dd75454e0399d418cdc632a3581cdc0`），静态合计7,395,750B（7.05MiB）。静态文件大小不是统一口径NPU运行峰值。

摄像头闭环已加载新双OM并持续至少720帧。最终日志记录25个采样点，全部为`person=none`，`HEAD`行数为0；Pose推理范围79.75–81.87ms，稳态均值80.5129ms、中位数80.475ms。该记录只证明空场景下模型加载和持续推理稳定，不是含人跌倒事件或完整同帧端到端时延。

历史 `train-6` 对照为：64窗 `max_abs=7.7e-7`、分类一致率100%；normal/fall板端输出0.001701/1.000000；Pose P95 83.11ms、VPSS+Pose P95 86.39ms、Head单次约3.9–4.9ms。以上均明确标为历史，不作为新权重实测值。

## 2. 为什么不能直接把普通 Transformer ONNX 转成 OM

以下普通导出文件不能作为 CV610 的最终 Transformer 部署模型：

```text
exports/posefall-svp/posefall_head.onnx
exports/posefall-svp/svp/model/posefall_head_original.om
```

普通 ONNX 中 Q/K/V 仍是 Linear/MatMul，且包含大量通用 Reshape、Transpose、LayerNormalization 和 Softmax 调度。它虽然可以通过 ATC，也能在板端加载，但执行时会触发 AICore 超时。模型文件大小不是超时的判断依据。

根据以下 ATC 文档的“Transformer 网络硬化加速”章节：

```text
/developer14/hefei/Hi3516CV610/docs/06.工具中心/SVP_NPU/ATC工具使用指南/ATC工具使用指南.md
```

Transformer 必须做定制化等价修改：

1. 数据流从 HWC/NLC 转为 CHW；
2. Q、K、V 投影改为 Conv；
3. 使用 CV610 可识别的专用 LayerNorm、Gelu 节点；
4. 让 ATC 将注意力周围的 Reshape/Transpose 合并到硬件友好的 MatMul 分组结构。

本文生成的正确模型是：

```text
exports/posefall-svp/yolo26n-posefall-2/picovision/posefall_head_picovision.onnx
exports/posefall-svp/yolo26n-posefall-2/picovision/model/posefall_head_picovision_original.om
```

## 3. 环境与目录

开发节点：

```bash
ssh root@172.19.0.112 -p 30740
```

项目、SDK 和 Python：

```bash
export PROJECT=/developer14/hefei/ultralytics
export SDK=/developer14/hefei/Hi3516CV610
export PYTHON=/developer14/anaconda3/envs/ultralytics/bin/python
cd "$PROJECT"
```

V1.3板端证据链使用的完整权重：

```text
runs/posefall/yolo26n-posefall-2/weights/best.pt
```

确认权重中同时包含姿态网络和 Transformer 跌倒头：

```bash
$PYTHON - <<'PY'
import torch
p = "runs/posefall/yolo26n-posefall-2/weights/best.pt"
ckpt = torch.load(p, map_location="cpu", weights_only=False)
model = ckpt["model"]
print(type(model).__name__)
print(type(model.posefall_head).__name__)
print(model.posefall_head)
PY
```

## 4. 首次部署：导出姿态模型和基础 SVP 文件

从完整 `.pt` 导出静态姿态 ONNX、普通参考头 ONNX、metadata 和 SVP 目录：

```bash
cd /developer14/hefei/ultralytics

$PYTHON scripts/export_posefall_onnx.py \
  --weights runs/posefall/yolo26n-posefall-2/weights/best.pt \
  --out-dir exports/posefall-svp/yolo26n-posefall-2 \
  --imgsz 640 \
  --opset 13 \
  --device cpu \
  --svp
```

主要输出：

```text
exports/posefall-svp/yolo26n-posefall-2/pose_yolo.onnx
exports/posefall-svp/yolo26n-posefall-2/posefall_head.onnx  # 仅作 PC 参考，不直接部署
exports/posefall-svp/yolo26n-posefall-2/posefall_export_meta.json
exports/posefall-svp/yolo26n-posefall-2/svp/calibration/pose_input.txt
```

## 5. 生成真实 Transformer 校准窗口

不要用全零 `head_input.txt` 做最终转换。应从训练时缓存的真实姿态序列中抽取正常和跌倒样本。

当前 yolo26n-pose 的特征缓存目录为：

```text
runs/posefall/features/yolo26n-pose-posefall-5abce169acd4
```

生成 32 个正常和 32 个跌倒窗口：

```bash
$PYTHON scripts/generate_posefall_head_calibration.py \
  --cache-root runs/posefall/features/yolo26n-pose-posefall-5abce169acd4 \
  --output exports/posefall-svp/yolo26n-posefall-2/svp/calibration/head_input_real.txt \
  --reference exports/posefall-svp/yolo26n-posefall-2/svp/calibration/head_reference.npz \
  --sample-bin exports/posefall-svp/yolo26n-posefall-2/svp/calibration/head_sample.bin \
  --samples-per-class 32 \
  --seed 610
```

输出含义：

- `head_input_real.txt`：ATC 的真实校准输入列表，每行一个 `60 x 56` 窗口；
- `head_reference.npz`：保存相同窗口、标签和来源，用于原模型与硬化模型数值对齐；
- `head_sample.bin`：生成器提供的单样本烟测输入；V1.3验收另从同一reference固定导出一例normal和一例fall，并在`smoke_samples.json`中保存索引、来源、PT概率和输入SHA。

如果更换姿态前端或特征定义，必须重新生成特征缓存和校准窗口。不能复用维度相同但定义不同的数据。

## 6. 导出并编译原始 Transformer 的 PicoVision 等价模型

一键执行：

```bash
cd /developer14/hefei/ultralytics
./scripts/convert_posefall_transformer.sh
```

脚本依次执行：

1. 从 `best.pt` 提取原始 Transformer 的全部权重和位置编码；
2. 用原始 Transformer 计算真实参考样本概率；
3. 将 Q/K/V、投影层和 FFN 的 Linear 权重原值映射到 1x1 Conv；
4. 将 LayerNorm/GELU 导出为 PicoVision 自定义 ONNX 节点；
5. 比较改写模型与原模型的 64 组输出；
6. 使用 ATC `compile_mode=5` 生成 Hi3516CV610 OM。

如果使用其他权重或输出目录：

```bash
WEIGHTS=/path/to/new_best.pt \
REFERENCE=/path/to/head_reference.npz \
IMAGE_LIST=/path/to/head_input_real.txt \
OUT_DIR=/path/to/output \
./scripts/convert_posefall_transformer.sh
```

导出时必须看到类似结果：

```text
reference max_abs<9e-7 class_agreement=1.000000
```

建议接受条件：

```text
class_agreement == 1.0
max_abs < 1e-4
```

若不满足，不要部署，应先检查模型结构是否改变、权重键是否变化、window/input_dim/d_model/nhead/num_layers 是否仍为 `60/56/256/4/3`。

## 7. 检查 Transformer 硬化是否生效

检查 ONNX 自定义节点：

```bash
$PYTHON - <<'PY'
import collections
import onnx

p = "exports/posefall-svp/yolo26n-posefall-2/picovision/posefall_head_picovision.onnx"
m = onnx.load(p)
print(collections.Counter((n.domain, n.op_type) for n in m.graph.node))
PY
```

应包含：

```text
('custom_domain', 'LayerNorm'): 7
('custom_domain', 'Gelu'): 5
('', 'Conv'): 23
```

检查 ATC 适配后的图：

```bash
rg -n "MatMul \(inGroup\)|MatMul \(outGroup\)|type name: LayerNorm|type name: Gelu" \
  exports/posefall-svp/yolo26n-posefall-2/picovision/model/cnn_net_tree_adapt.dot
```

每层注意力应出现 `MatMul (inGroup)` 和 `MatMul (outGroup)`，并显示 `mergedOps: Permute;Reshape;`。这才说明注意力数据变形已被硬化合并。

## 8. 编译摄像头姿态 OM

板端实时程序直接从 VPSS 取得 `640 x 640 YVU420SP`，因此不能直接给检测程序使用普通 NCHW 浮点姿态 OM。姿态 OM 必须包含适配摄像头输入的静态 AIPP，并使用：

```text
compile_mode=5
```

当前已验证的板端文件为：

```text
pose_yolo_camera_yvu_fp16v.om
size=4,238,920 bytes
sha256=f71e4a264f22f055e49a95b66ecc5602274ed636c390263a9d54f45cfbddc2db
```

数据链路为：

```text
SC4336P -> VI/ISP -> VPSS 640x640 YVU420SP -> AIPP -> pose YOLO
```

本次新旧checkpoint审计覆盖879个Pose前端张量、3,700,640个参数，逐元素最大差异为0，聚合SHA均为`cb5570a0b2cec0500d77b65460cecac9d69908821c153e7bbd61137327aee6d9`。因此本次继续使用现有 `pose_yolo_camera_yvu_fp16v.om`，不重复编译姿态模型。这不是仅凭结构相同作出的假设，而是逐张量同一性结论。

若姿态前端也改变，则需从新的 `pose_yolo.onnx` 重新生成带静态 AIPP 的 OM。AIPP 必须开启 CSC，并与 VPSS 的 YVU420SP/NV21 排列一致；颜色矩阵错误会导致画面虽能推理，但人体框和关键点置信度严重下降。

## 9. 编译板端检测程序

```bash
cd /developer14/hefei/ultralytics/deploy/hi3516cv610_posefall
make -f Makefile.posefall_detector clean
make -f Makefile.posefall_detector
file posefall_detector
```

完整板端管线：

```text
SC4336P
  -> VPSS 640x640 YVU420SP
  -> pose_yolo_camera_yvu_fp16v.om
  -> CPU 选取最高置信度人体候选
     （当前板端为单主要人物模式，不执行 NMS、BoT-SORT 或持续 ID 跟踪）
  -> 56-value pose feature
  -> 60-frame temporal window
  -> posefall_head_picovision.om
  -> fall probability/debounce state
  -> RGN overlay
  -> H.264 RTSP
```

## 10. 下发到板子

本次直连复测的板子临时地址：

```text
169.254.131.168/16
```

Windows 直连网卡地址：

```text
169.254.131.219/16
```

先将以下文件放入 Windows TFTP 根目录：

```text
pose_yolo_camera_yvu_fp16v.om
posefall_head_picovision.om
posefall_detector
start_posefall.sh
stop_posefall.sh
posefall_om_benchmark
head_normal_yolo26n_posefall_2.bin
head_fall_yolo26n_posefall_2.bin
```

其中 Transformer OM 来源为：

```text
exports/posefall-svp/yolo26n-posefall-2/picovision/model/posefall_head_picovision_original.om
```

复制到 TFTP 根目录时将它重命名为：

```text
posefall_head_picovision.om
```

通过串口连接：

```powershell
py -m serial.tools.miniterm COM5 115200 --eol CRLF --filter direct
```

板端执行：

```sh
mkdir -p /root/posefall
cd /root/posefall

tftp -g -r posefall_head_picovision.om 169.254.131.219
tftp -g -r posefall_detector 169.254.131.219
tftp -g -r start_posefall.sh 169.254.131.219
tftp -g -r stop_posefall.sh 169.254.131.219
tftp -g -r posefall_om_benchmark 169.254.131.219
tftp -g -r head_normal_yolo26n_posefall_2.bin 169.254.131.219
tftp -g -r head_fall_yolo26n_posefall_2.bin 169.254.131.219

chmod +x posefall_detector start_posefall.sh stop_posefall.sh posefall_om_benchmark
ls -lh
```

切换前先把原Head OM备份为`posefall_head_picovision_train6.om`；不要删除`posefall_head_mlp.om`，它只作为故障回退文件保留。新Head OM的板端SHA256必须为`85d26e3f179153d7e28457f34881fd9a3dd75454e0399d418cdc632a3581cdc0`。

## 11. 先做板端单模型测试

在启动实时检测前先停止现有检测进程，但保留摄像头和 RTSP：

```sh
cd /root/posefall
./stop_posefall.sh
```

分别测试normal和fall窗口，并各运行200次（预热5次）：

```sh
./posefall_om_benchmark posefall_head_picovision.om head_normal_yolo26n_posefall_2.bin 200 5
./posefall_om_benchmark posefall_head_picovision.om head_fall_yolo26n_posefall_2.bin 200 5
```

成功标志：

```text
normal: PT=0.000213212, OM=0.000230789, mean=4.159966 ms
fall:   PT=0.999946475, OM=1.000000000, mean=4.166595 ms
wrote output_0.bin (4 bytes)
MODEL_SMOKE_OK
```

如果长时间无输出或出现 AICore timeout，不要切换实时启动脚本。首先检查使用的是否确实为新PicoVision OM（SHA `85d26e…1cdc0`），而不是普通 `posefall_head_original.om`。

## 12. 切换实时检测

确认 `/root/posefall/start_posefall.sh` 中配置为：

```sh
MODEL="$BASE/pose_yolo_camera_yvu_fp16v.om"
HEAD_MODEL="$BASE/posefall_head_picovision.om"
DETECTOR="$BASE/posefall_detector"
```

当前板端 C 程序从 8400 个姿态候选中仅选取置信度最高的人体，未执行 NMS、BoT-SORT 或持续 ID 跟踪。因此这一部署版本面向固定机位、单主要人物场景；Python 参考实现的多人跟踪能力不能视为已经部署到板端。

启动：

```sh
cd /root/posefall
./start_posefall.sh
```

检查进程和日志：

```sh
ps | grep -E 'posefall_detector|sample_venc|fixed_rtsp_server'
tail -f /run/posefall.log
```

有人的正常日志应包含：

```text
POSE frame=... infer_ms=...
HEAD frame=... prob=... infer_ms=...
STATE frame=... state=NORMAL|FALL ...
```

V1.3最终归档的摄像头日志为空场景：已持续至少720帧，25条采样全部`person=none`、没有`HEAD`行；Pose为79.75–81.87ms，稳态均值80.5129ms、中位数80.475ms。独立Head均值约4.16ms，不能与该空场景Pose值拼接成同帧端到端时延。

电脑端打开：

```text
rtsp://169.254.131.168:554/live.h264
```

## 13. 停止与回退

只停止检测，保留摄像头和 RTSP：

```sh
/root/posefall/stop_posefall.sh
```

同时停止检测、摄像头编码和 RTSP：

```sh
/root/posefall/stop_posefall.sh all
```

如果新 Transformer 头出现问题，可先显式回退到已备份的历史 `train-6` 头：

```sh
POSEFALL_HEAD_MODEL="/root/posefall/posefall_head_picovision_train6.om" \
  /root/posefall/start_posefall.sh
```

若仅为排查兼容性，也可把环境变量指向 `/root/posefall/posefall_head_mlp.om`。不设置 `POSEFALL_HEAD_MODEL` 时，启动脚本默认使用 `posefall_head_picovision.om` 符号链接指向的 `yolo26n-posefall-2` 新 Transformer 头。

## 14. 后续再次更新 PT 的最短流程

V1.3新主结果已按本流程完成。后续再次更新权重时，不能只凭结构相同复用Pose OM；应先逐张量验证姿态前端是否完全一致。若一致且只更新Head：

```bash
cd /developer14/hefei/ultralytics

# 如果特征缓存/数据集发生变化，先重新生成真实校准窗口。
$PYTHON scripts/generate_posefall_head_calibration.py \
  --cache-root runs/posefall/features/yolo26n-pose-posefall-5abce169acd4 \
  --output exports/posefall-svp/yolo26n-posefall-2/svp/calibration/head_input_real.txt \
  --reference exports/posefall-svp/yolo26n-posefall-2/svp/calibration/head_reference.npz \
  --sample-bin exports/posefall-svp/yolo26n-posefall-2/svp/calibration/head_sample.bin \
  --samples-per-class 32

WEIGHTS=runs/posefall/yolo26n-posefall-2/weights/best.pt \
  ./scripts/convert_posefall_transformer.sh
```

之后依次进行：

1. 确认 `class_agreement=1.0`；
2. 检查 ATC 图中存在 `MatMul (inGroup/outGroup)`；
3. 将新 OM 下发到板子；
4. 使用normal/fall两个固定窗口运行 `posefall_om_benchmark`；
5. 备份旧 OM 后替换实时模型；
6. 观察 `/run/posefall.log` 至少数百帧；
7. 用真实正常、蹲下、坐下、躺下和跌倒动作重新验证阈值和防抖策略；空场景日志不能替代该项。

## 15. 结果解释

对当前 `yolo26n-posefall-2` 证据链而言，“已部署原Transformer权重”指时序输入、3层4头结构和全部新Head权重都来自主结果 `.pt`，并且等价改写、64窗对齐、板端normal/fall烟测与空场景持续运行均已通过；它不表示ONNX算子图与PyTorch图逐节点字节相同。为在CV610上执行，Linear、数据布局和归一化节点必须做数学等价的PicoVision硬化改写。Pose OM的复用有879个张量逐元素一致的审计支持。

当前尚不能说“含人跌倒检测已用新权重完整复测”：最终摄像头日志为空场景、没有HEAD行。只有补录含人正常与跌倒动作、同时取得Pose/HEAD/STATE及完整链路时延后，才能作这一更强结论。旧 `train-6` 证据仅作历史比较。

部署成功只证明模型能够稳定执行。实际误报、漏报和阈值仍属于模型效果问题，需要用目标摄像机角度和真实场景数据单独评估。
