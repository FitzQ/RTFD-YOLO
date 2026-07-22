#!/bin/sh
set -u

BASE=/root/posefall
MODEL="$BASE/pose_yolo_camera_yvu_fp16v.om"
# The versioned yolo26n-posefall-2 Transformer is exposed through the default
# posefall_head_picovision.om symlink. Roll back explicitly without replacing it:
#   POSEFALL_HEAD_MODEL="$BASE/posefall_head_picovision_train6.om" ./start_posefall.sh
# The MLP remains an emergency compatibility fallback only:
#   POSEFALL_HEAD_MODEL="$BASE/posefall_head_mlp.om" ./start_posefall.sh
HEAD_MODEL="${POSEFALL_HEAD_MODEL:-$BASE/posefall_head_picovision.om}"
DETECTOR="$BASE/posefall_detector"
RTSP_SERVER="$BASE/fixed_rtsp_server"

if [ ! -x "$DETECTOR" ] || [ ! -s "$MODEL" ] || [ ! -s "$HEAD_MODEL" ] ||
   [ ! -x "$RTSP_SERVER" ]; then
    echo "posefall files are missing under $BASE" >&2
    exit 1
fi
cd "$BASE" || exit 2

if ! pidof sample_venc >/dev/null 2>&1; then
    echo "starting SC4336P camera, encoder and fixed RTSP..."
    rm -f /run/stream_chn0.h265 /run/stream_chn1.h264
    mkfifo /run/stream_chn0.h265 /run/stream_chn1.h264 || exit 2
    rm -f "$BASE/stream_chn0.h265" "$BASE/stream_chn1.h264"
    ln -s /run/stream_chn0.h265 "$BASE/stream_chn0.h265" || exit 2
    ln -s /run/stream_chn1.h264 "$BASE/stream_chn1.h264" || exit 2

    "$RTSP_SERVER" /run/stream_chn1.h264 554 > /run/fixed_rtsp.log 2>&1 &
    echo $! > /run/fixed_rtsp.pid
    cat /run/stream_chn0.h265 > /dev/null &
    echo $! > /run/h265_drain.pid

    (
        (printf '0\n'; sleep 1; printf 'c\n'; exec tail -f /dev/null) |
            /root/sample_venc 0 > /run/venc.log 2>&1
    ) &
    echo $! > /run/venc_wrapper.pid
    sleep 5
fi

if ! pidof sample_venc >/dev/null 2>&1; then
    echo "sample_venc failed; see /run/venc.log" >&2
    exit 3
fi
if ! pidof fixed_rtsp_server >/dev/null 2>&1; then
    echo "fixed RTSP server failed; see /run/fixed_rtsp.log" >&2
    exit 4
fi

old_pid="$(pidof posefall_detector 2>/dev/null || true)"
if [ -n "$old_pid" ]; then
    kill -TERM $old_pid 2>/dev/null || true
    sleep 2
fi

: > /run/posefall.log
"$DETECTOR" "$MODEL" "$HEAD_MODEL" > /run/posefall.log 2>&1 &
echo $! > /run/posefall.pid
sleep 2

if ! kill -0 "$(cat /run/posefall.pid)" 2>/dev/null; then
    echo "posefall detector failed; see /run/posefall.log" >&2
    tail -n 30 /run/posefall.log
    exit 5
fi

echo "posefall started pid=$(cat /run/posefall.pid)"
echo "models: $(basename "$MODEL") + $(basename "$HEAD_MODEL")"
echo "RTSP: rtsp://$(ip -4 addr show eth0 | awk '/inet / {sub(/\/.*/, "", $2); print $2; exit}'):554/live.h264"
echo "log: /run/posefall.log"
