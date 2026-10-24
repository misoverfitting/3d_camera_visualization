#!/bin/sh
# Builds a self-contained Release app (JS bundled, no Metro needed), installs it
# on a connected iPhone and launches it. Uses a development bundle ID so it never
# collides with the App Store app's identifier or signing.
#   TEAM_ID=XXXXXXXXXX DEVICE=<UDID> sh scripts/run-device.sh
# List devices with: xcrun devicectl list devices
set -eu
: "${TEAM_ID:?Set TEAM_ID to your Apple development team ID}"
: "${DEVICE:?Set DEVICE to the iPhone UDID}"
BUNDLE_ID="${BUNDLE_ID:-com.tare.capture.dev}"
cd "$(dirname "$0")/.."

TARE_BUNDLE_ID="$BUNDLE_ID" TARE_APPLE_TEAM_ID="$TEAM_ID" CI=1 npx expo prebuild --platform ios --clean
xcodebuild -workspace ios/TareCapture.xcworkspace -scheme TareCapture -configuration Release \
  -destination "id=$DEVICE" -derivedDataPath ios/build -allowProvisioningUpdates build
xcrun devicectl device install app --device "$DEVICE" ios/build/Build/Products/Release-iphoneos/TareCapture.app
xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE_ID"
