#!/bin/sh
# Downloads the pinned official OpenCV iOS framework that the native module links.
# Runs from npm postinstall so local and EAS builds vendor the same verified binary.
set -eu

VERSION=4.14.0
SHA256=f4f428fa9270adf95b3ed4705eb18ba3028ed2c2777a4f1a002739b869986a6a
DEST="$(cd "$(dirname "$0")/.." && pwd)/modules/tare-capture/ios/vendor"
STAMP="$DEST/opencv2.framework/.tare-version"

if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$VERSION" ]; then exit 0; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Fetching OpenCV $VERSION iOS framework…"
curl -fsSL -o "$TMP/opencv.zip" \
  "https://github.com/opencv/opencv/releases/download/$VERSION/opencv-$VERSION-ios-framework.zip"
echo "$SHA256  $TMP/opencv.zip" | shasum -a 256 -c - >/dev/null
unzip -q "$TMP/opencv.zip" -d "$TMP/x"
rm -rf "$DEST/opencv2.framework"
mkdir -p "$DEST"
mv "$TMP/x/opencv2.framework" "$DEST/"
echo "$VERSION" > "$STAMP"
