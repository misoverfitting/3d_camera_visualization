# Tare iPhone capture prototype

Native ARKit capture for iPhone 16, delivered through EAS cloud builds and
TestFlight. Expo Go cannot load this app's custom Swift module. No cable or
working local Mac is required for the cloud workflow.

## Implemented

- Native camera preview and horizontal surface raycast on tap; world-aligned
  adjustable box extends upward from that surface. A raycast is a placement
  estimate, not measured object depth or an automatic segmentation mask.
- Start/stop RGB capture at up to two frames per second, capped at 120 seconds
  and 240 frames. Normally tracked frames only; at most one retained image
  buffer is queued for JPEG encoding. Native pixels never cross the JS bridge.
- Unrotated sensor JPEGs, per-frame intrinsics and camera-to-world poses from
  the same ARFrame, timestamps and tracking state; recording-level scan bounds
  and skip/stop diagnostics. Interruptions finalize the scan rather than combine
  observations across potentially incompatible tracking sessions.
- Atomic image/manifest writes, app Documents storage, ZIP export through the
  iOS share sheet. Recordings are accessible through Files/file sharing. No
  automatic server upload or deletion; manage old scans in Files.
- Desktop manifest/image validator and COLMAP known-pose TXT export.
- Cyan overlays show estimated horizontal plane extents; feature points and
  surface counts help diagnose tracking. A failed tap reports guidance instead
  of silently returning. Skip box permits normally tracked RGB/pose recording
  without waiting for plane detection; in that case `region` is JSON null.

## On-device depth

Both depth sources write into one 5 mm voxel grid in the ARKit world frame, shown
live over the camera (magenta = LiDAR, green = multi-view) and saved with each
recording as `points.ply` (binary PLY: x, y, z float meters; per-voxel observation
count). Points are kept inside the scan box when one is placed.

- **LiDAR** (`depth_mode: "lidar"`): ARKit `smoothedSceneDepth`, high-confidence
  pixels only, 0.05–1.5 m, sampled at 5 Hz. The live view hides voxels seen once.
- **Multi-view** (`depth_mode: "multiview"`): used on phones without LiDAR, and on
  LiDAR phones via *Force multi-view depth (benchmark)*. OpenCV pyramidal
  Lucas-Kanade tracks corners at 15 Hz on 960 x 720 luma with a forward-backward
  check. Tracks are triangulated (linear multi-view DLT) from fixed ARKit poses
  only after at least 6 degrees of parallax and 2 cm of baseline, with every view
  in front of the camera and under 1 px reprojection error (downscaled pixels).
  Sparse by design: textureless or glossy food produces few points. Tracking
  loss resets all tracks.

Scanning the same scene in both modes on one LiDAR phone gives a ground-truth
comparison for the multi-view path. Neither cloud is yet evaluated against
measured real objects.

OpenCV 4.14.0 (official iOS framework) is fetched by `scripts/fetch-opencv.sh` on
`npm install`/`npm ci`, verified by SHA-256, into the module's ignored `vendor/`.
`native-tests/run_tracker_test.sh` runs the tracker on macOS against a ray-cast
synthetic scene with known geometry.

Exported ZIPs can now be imported through the web app's Import scan form. The
server validates the recording, creates a native session, triangulates with fixed
ARKit poses and feeds the existing mesh/splat backends. Native mesh reconstruction
requires working CUDA dense stereo by default; coarse CPU preview is explicit.
The recorded box crops the reconstructed cloud, not the original images. Pose
refinement and texture baking remain future work. See docs/NATIVE_RECONSTRUCTION.md
for the complete workflow and verification limits.

## Install and check on Windows

Run from this `mobile` folder, using a supported Node LTS runtime:

```powershell
npm ci
npm run typecheck
npx expo install --check
npx expo-modules-autolinking resolve --platform apple
```

The autolinking output must contain `tare-capture-native`, `TareCaptureNative`, and
`TareCaptureModule`. Dependencies are locked in package-lock.json. ZIPFoundation
is linked by the local podspec for native ZIP export.
The native pod is named `TareCaptureNative` to avoid colliding with the app's
`TareCapture` Xcode scheme, which is explicitly selected in both EAS profiles.

## First cloud build and TestFlight

For each iterative iPhone update, run from `mobile/`:

```powershell
npm run testflight
```

This runs `eas build --platform ios --profile production --auto-submit`.
Production explicitly uses store distribution, remote auto-incrementing build
numbers and the existing bundle identifier `com.tare.capture`. Submission targets
the existing App Store Connect app `6824637632`, so new builds belong to the same
TestFlight app instead of creating another app. Do not change the bundle identifier
or run `eas init` for routine updates.

If needed, run `npx eas-cli@latest login` first and complete Apple's credential
prompts in your terminal. After Apple's processing completes, open TestFlight,
select Tare Capture, and choose Update (or Install if it is not installed).
In App Store Connect, enable automatic distribution on the internal tester group
to receive new eligible builds automatically, or add each new build to that group.
External tester groups may require beta review. Submission uploads to TestFlight;
it does not publish the app on the public App Store. No Metro server is needed.

If an existing **production/store** build finished but was not submitted:

```powershell
npx eas-cli@latest submit --platform ios --profile production
```

Choose the intended store build interactively. An ad hoc preview binary cannot
be submitted to TestFlight; create a production build using `npm run testflight`.

The `preview` profile remains an alternative ad hoc build requiring registered
device provisioning. Its installation links do not create TestFlight updates.

## Device acceptance check

1. Grant camera permission. Move the phone until tracking reads normal.
2. Look downward and move slowly sideways over a textured horizontal support
   surface. Tap a cyan patch near a stationary matte object, then adjust the box.
   Check failed-tap feedback and surface counts. If no plane is found, choose
   Skip box and verify that capture works with normal tracking and a null region.
3. Record a slow orbit with height changes. Stop and export the recording ZIP
   to Files/cloud storage. Ensure the JPEGs contain the real camera image only,
   without the preview's wireframe overlay.
4. Test camera denial, tracking loss, screen lock/background, session interruption,
   automatic time/frame cap, repeated scans, and export after an interruption.
5. Extract the ZIP on desktop. Point the validator at the folder containing
   `manifest.json`:

```powershell
python scripts/validate_arkit_recording.py PATH_TO_SCAN
python scripts/validate_arkit_recording.py PATH_TO_SCAN --colmap-output NEW_SPARSE_FOLDER
python scripts/test_arkit_recording.py
```

Those commands run from the repository root. Validation needs Pillow; optional
COLMAP export also needs NumPy (both are already server requirements).
Keep the scan root as the image root when using exported COLMAP image names.
The standalone export contains empty observations/points and is not itself a
reconstructed model. Import the recording ZIP through the server to extract and
match features, use database-assigned IDs, and triangulate those observations.

## Recording contract version 1

`manifest.json` identifies `source=arkit_rgb`, `schema_version=1`, meter units,
column-major matrices, and camera-to-world poses. ARKit camera axes are right/up/
back (+X/+Y/+Z); the camera looks along -Z. JPEGs remain in sensor orientation;
intrinsics refer to their full, unrotated resolution. Timestamps use ARFrame's
monotonic seconds clock, not wall time. No per-frame depth maps are included;
`depth_mode` and `point_cloud` describe the fused `points.ply` (world frame).

`frames` contains image path, timestamp, width, height, 3x3 intrinsics, 4x4 pose,
and tracking state. `region` records bottom-center world origin, cube size in
meters, and convention. `diagnostics` records successful frames, busy-writer skips,
tracking skips, and stop reason. The COLMAP exporter inverts the pose and changes
camera axes to right/down/forward. Validate alignment visually on real recordings
before relying on these poses for reconstruction.

## Validation boundary

Windows can check TypeScript, module discovery, JS bundling, and desktop tooling.
Swift compilation, CocoaPods resolution, camera calibration alignment, export
share-sheet behavior, and sustained device memory need the first cloud build and
physical iPhone test. A killed process may leave an incomplete folder without a
manifest; the validator/exporter intentionally do not treat it as a complete scan.
