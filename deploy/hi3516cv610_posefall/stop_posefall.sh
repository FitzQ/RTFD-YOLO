#!/bin/sh

pid="$(pidof posefall_detector 2>/dev/null || true)"
if [ -n "$pid" ]; then
    kill -TERM $pid 2>/dev/null || true
    sleep 2
fi
rm -f /run/posefall.pid
echo "posefall detector stopped"

if [ "${1:-}" = "all" ]; then
    venc_pid="$(pidof sample_venc 2>/dev/null || true)"
    if [ -n "$venc_pid" ]; then
        kill -INT $venc_pid 2>/dev/null || true
        sleep 2
    fi
    if [ -s /run/venc_wrapper.pid ]; then
        kill "$(cat /run/venc_wrapper.pid)" 2>/dev/null || true
    fi
    if [ -s /run/fixed_rtsp.pid ]; then
        kill "$(cat /run/fixed_rtsp.pid)" 2>/dev/null || true
    fi
    if [ -s /run/h265_drain.pid ]; then
        kill "$(cat /run/h265_drain.pid)" 2>/dev/null || true
    fi
    rm -f /run/venc_wrapper.pid /run/fixed_rtsp.pid /run/h265_drain.pid
    rm -f /run/stream_chn0.h265 /run/stream_chn1.h264
    rm -f /root/posefall/stream_chn0.h265 /root/posefall/stream_chn1.h264
    echo "camera and RTSP stopped"
fi
