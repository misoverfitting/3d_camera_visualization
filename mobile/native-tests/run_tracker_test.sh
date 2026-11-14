#!/bin/sh
# Compiles TareFeatureTracker.mm for macOS against a host OpenCV build (core,
# imgproc, video) and runs the synthetic triangulation check.
#   OPENCV_DIR=/path/to/opencv/install sh native-tests/run_tracker_test.sh
# A minimal host OpenCV can be built from the same release with:
#   cmake -S opencv-4.14.0 -B build -DBUILD_LIST=core,imgproc,video -DBUILD_SHARED_LIBS=OFF \
#     -DWITH_IPP=OFF -DWITH_OPENCL=OFF -DCMAKE_INSTALL_PREFIX=$OPENCV_DIR && cmake --build build --target install
set -eu
: "${OPENCV_DIR:?Set OPENCV_DIR to a host OpenCV install prefix}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
clang++ -std=c++17 -O2 -fobjc-arc \
  -I"$OPENCV_DIR/include/opencv4" \
  "$HERE/tracker_synthetic_test.mm" "$HERE/../modules/tare-capture/ios/TareFeatureTracker.mm" \
  -L"$OPENCV_DIR/lib" -L"$OPENCV_DIR/lib/opencv4/3rdparty" \
  -lopencv_video -lopencv_imgproc -lopencv_core $(ls "$OPENCV_DIR/lib/opencv4/3rdparty" 2>/dev/null | sed -n 's/^lib\(.*\)\.a$/-l\1/p') \
  -lz -framework Foundation -framework CoreVideo -framework Accelerate \
  -o "$OUT/tracker_test"
"$OUT/tracker_test"
