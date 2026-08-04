# Hi3516CV610 PoseFall 导出

- `board/`：可直接复制到板端的最终四件套。
- `work/`：ONNX、校准数据、PicoVision 中间张量和 ATC 输出，不需要复制到板端。

当前权重：

```text
runs/posefall/yolo26n-posefall-2/weights/best.pt
```

完整流程见：

```text
deploy/hi3516cv610_posefall/POSEFALL_EXPORT_AND_DEPLOY.md
```
