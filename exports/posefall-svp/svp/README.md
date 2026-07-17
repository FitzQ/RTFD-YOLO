# PoseFall on Hi3516CV610 SVP NPU

This directory converts the two static ONNX networks to SVP offline `.om` models.

## Convert

```bash
CANN_SETENV=/path/to/ascend-toolkit/svp_latest/x86_64-linux/script/setenv.sh ./convert_om.sh
```

The bundled zero-valued calibration inputs are only conversion smoke-test data. Replace them with representative normalized pose images and real 60x56 PoseFall windows before final accuracy validation.

## Board pipeline

1. Decode camera/video frames and resize/letterbox to the exported image size.
2. Run `pose_yolo_original.om` through SVP ACL.
3. Perform confidence filtering, NMS, keypoint parsing, and person tracking on the ARM CPU.
4. Build the exact 56-value feature vector and maintain the fixed temporal window.
5. Run `posefall_head_original.om` through SVP ACL and apply the configured threshold.

`posefall_onnx_runtime.py` is the PC reference implementation. It is not the board SVP ACL application.
