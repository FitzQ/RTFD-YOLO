# 板端文件

本目录只包含需要复制到相同 Hi3516CV610 固件板卡的四个文件：

```text
pose_yolo_camera_yvu_fp16v.om
posefall_head_picovision.om
posefall_predict
ffmpeg
```

复制后执行：

```sh
chmod +x posefall_predict ffmpeg
```

校验：

```sh
sha256sum -c SHA256SUMS
```

视频推理：

```sh
POSEFALL_FFMPEG=./ffmpeg ./posefall_predict \
  --source /mnt/usb/test.mp4 \
  --pose ./pose_yolo_camera_yvu_fp16v.om \
  --head ./posefall_head_picovision.om
```

摄像头推理使用 `--source camera` 或 `--source 0`，并需要先启动板上的 VI/ISP/VPSS 摄像头管线。
