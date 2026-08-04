# Hi3516CV610 PoseFall deployment

## Verified version (2026-07-29, V1.4)

The competition-document result now uses
`runs/posefall/yolo26n-posefall-2/weights/best.pt` (SHA256
`d9bd12b969cdb3f957c2921ae5736aadec3e4915392d3bc37624d6ba41293773`).
That checkpoint has now been converted and retested on the board. An elementwise
checkpoint audit found all 879 pose-front-end tensors identical to the previous
checkpoint, so the verified camera Pose OM is reused. All 48 Transformer-head
tensors changed and were exported into the current head OM.

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
timeout.

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

The board runtime now preserves the end-to-end `PoseFallPredictor` semantics:

- pose confidence threshold `0.1` and IoU NMS threshold `0.7`;
- up to 32 people after NMS;
- BoT-SORT with `fall_botsort.yaml` thresholds, score-fused IoU, the
  `KalmanFilterXYWH` state model, high/low-confidence two-stage association,
  unconfirmed/lost/removed track lifecycle, and persistent IDs;
- `gmc_method=none` and `with_reid=False`, exactly as configured by the
  project, so no omitted GMC or ReID network exists;
- one independent 60x56 history per ID, resampled to 30 FPS with the same
  repeat-previous rule and 0.5 s reset gap as `append_resampled_feature`;
- Transformer Head inference on every valid tracked person in every processed
  frame, with direct `probability >= 0.5` classification and no stride or
  hysteresis.

The C `KalmanFilterXYWH` implementation was checked by feeding the board's raw
detections back into the Python `BOTSORT`: IDs agreed on every checked frame,
and the maximum tracked-box coordinate error over the five-frame numerical
check was about 0.002 pixels.

See `POSEFALL_EXPORT_AND_DEPLOY.md` for the current PT-to-board conversion,
directory layout, numerical validation, and deployment steps.

## Build

```sh
cd /developer14/hefei/ultralytics/deploy/hi3516cv610_posefall
make -f Makefile.posefall_predict
```

## Board usage

```sh
/root/posefall/start_posefall.sh
tail -f /run/posefall.log
```

The predictor also accepts a source directly:

```sh
# Camera
/root/posefall/posefall_predict \
  --source camera \
  --pose /root/posefall/pose_yolo_camera_yvu_fp16v.om \
  --head /root/posefall/posefall_head_picovision.om

# Video (or a preconverted 640x640 NV21 stream)
POSEFALL_FFMPEG=/root/posefall/ffmpeg \
/root/posefall/posefall_predict \
  --source /mnt/usb/test.mp4 \
  --source-fps 30 \
  --pose /root/posefall/pose_yolo_camera_yvu_fp16v.om \
  --head /root/posefall/posefall_head_picovision.om \
  --trace /mnt/usb/test_trace.csv
```

Container video is decoded and letterboxed before the OMs are loaded because
the board exposes only about 33 MiB of Linux memory. The temporary
`<source>.posefall_640.nv21` is created beside the video and deleted on exit;
therefore use USB/SD storage for a complete video. A 640x360 MPEG-4 test clip
ran successfully. A 1280x720 H.264 software decode exceeded this board image's
memory limit before model loading, so high-resolution inputs should be
pre-scaled to at most 640p. Container FPS is detected automatically;
`--source-fps` is an optional override. Raw `.nv21` input bypasses FFmpeg and
defaults to 30 FPS unless overridden.

For the verified direct link, open
`rtsp://169.254.131.168:554/live.h264` from the PC at
`169.254.131.219/16`. The board returns to `192.168.1.168/24` after reboot, so
the temporary direct-link address may need to be restored over COM5.
Green boxes are normal and red boxes are a fall. Stop only inference with
`/root/posefall/stop_posefall.sh`, or stop inference, camera, and RTSP with
`/root/posefall/stop_posefall.sh all`.

The startup script links encoder outputs to FIFOs under `/run`; do not replace
those links with regular files, or the small root filesystem will fill up.
