# Hi3516CV610 PoseFall deployment

## Verified version (2026-07-22, V1.3)

The competition-document result now uses
`runs/posefall/yolo26n-posefall-2/weights/best.pt` (SHA256
`d9bd12b969cdb3f957c2921ae5736aadec3e4915392d3bc37624d6ba41293773`).
That checkpoint has now been converted and retested on the board. An elementwise
checkpoint audit found all 879 pose-front-end tensors identical to the previous
checkpoint, so the verified camera Pose OM is reused. All 48 Transformer-head
tensors changed and were exported into a new head OM. The old `train-6/best.pt`
results (SHA256 `a34689b5b1052cf8da94f158bdeda042f8d14612e3c9309eab6934f0cb4a8b48`)
remain only as explicitly labelled historical comparisons.

The board pipeline is:

`SC4336P -> VPSS 640x640 YVU420SP -> pose OM -> 56-value pose features -> 60-frame head OM -> overlay -> H.264 RTSP`

## Models

- Pose: `pose_yolo_camera_yvu_fp16v.om`, 4,238,920 bytes, SHA256
  `f71e4a264f22f055e49a95b66ecc5602274ed636c390263a9d54f45cfbddc2db`
- Temporal head: `posefall_head_picovision.om`, 3,156,830 bytes, SHA256
  `85d26e3f179153d7e28457f34881fd9a3dd75454e0399d418cdc632a3581cdc0`
- Active dual-OM static total: 7,395,750 bytes (7.05 MiB). This is file size,
  not the official NPU run-time peak-memory measurement.

The temporal head preserves the original trained 3-layer Transformer and its
weights. It is exported in the PicoVision CHW/Conv form required by the CV610;
the generic ONNX-to-OM graph is not used because that graph causes an AICore
timeout. `posefall_head_mlp.om` remains on the board only as a fallback.

The new checkpoint passed these checks:

- 64 real windows: `max_abs < 9e-7`, class agreement `1.0`;
- normal sample PyTorch/OM: `0.000213212 / 0.000230789`;
- fall sample PyTorch/OM: `0.999946475 / 1.000000`;
- 200 post-warm-up head runs: normal/fall mean `4.159966 / 4.166595 ms`;
- camera loop: at least 720 frames with both OMs loaded. All 25 logged samples
  were `person=none`, with zero `HEAD` lines; Pose inference ranged
  `79.75–81.87 ms` (steady-state mean `80.5129 ms`, median `80.475 ms`).

The last item proves stable loading and continuous empty-scene inference. It is
not a person-present fall event and not a same-frame end-to-end latency result.

The current board C runtime is intentionally a fixed-camera, single-primary-
person implementation: from the 8400 pose candidates it uses only the
highest-confidence person. It does not run NMS, BoT-SORT, or persistent ID
tracking on the ARM CPU. The Python reference runtime supports tracked
multi-person histories, but that capability must not be attributed to the
current board binary.

See `POSEFALL_PT_TO_HI3516CV610.md` for the complete PT-to-board
conversion, numerical validation, ATC hardening, smoke-test, and rollback steps.

## Build

```sh
cd /developer14/hefei/ultralytics/deploy/hi3516cv610_posefall
make -f Makefile.posefall_detector
```

## Board usage

```sh
/root/posefall/start_posefall.sh
tail -f /run/posefall.log
```

The original Transformer is the default. The MLP is an explicit emergency
fallback only:

```sh
POSEFALL_HEAD_MODEL=/root/posefall/posefall_head_mlp.om \
  /root/posefall/start_posefall.sh
```

For the verified direct link, open
`rtsp://169.254.131.168:554/live.h264` from the PC at
`169.254.131.219/16`. The board returns to `192.168.1.168/24` after reboot, so
the temporary direct-link address may need to be restored over COM5.
Green boxes are normal and red boxes are a fall. Stop only inference with
`/root/posefall/stop_posefall.sh`, or stop inference, camera, and RTSP with
`/root/posefall/stop_posefall.sh all`.

The startup script links encoder outputs to FIFOs under `/run`; do not replace
those links with regular files, or the small root filesystem will fill up.
