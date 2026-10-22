# Food selection, tracking, and AR placement

Research reviewed 22 October 2026. This is a proposal, not an implemented food
recognition feature. The active reconstruction is unchanged. Its follow-up now
checks every ten minutes.

## Recommendation

Keep ARKit for camera poses, metric coordinates and support-plane detection.
Add image segmentation to select food pixels, temporal tracking to maintain the
selection, and a separate trained model to suggest food identities. The scan box
becomes an optional coarse crop and manual fallback rather than the foreground
definition.

Start in the existing Swift module with Apple Vision/Core ML for on-device
inference and tracking. Prototype OpenCV processing in the Python backend.
Consider an Objective-C++ wrapper for selected OpenCV operations on iOS only if
benchmarks justify it. Introducing a Rust/iOS/C++ bridge is not a prerequisite.
The linked Rust crate is bindings to OpenCV, not another recognition algorithm
or an automatic acceleration of the same C++ kernels.

## What exists in this branch

- `mobile/modules/tare-capture/ios/TareCaptureModule.swift` already receives
  ARFrames, uses horizontal raycasts, draws detected plane meshes, and records
  native-orientation RGB, intrinsics, timestamps and camera-to-world transforms.
- Selection currently stores a world-aligned cube with its origin at the bottom
  center. There is no food model, per-pixel food mask, or persistent food ID.
- The recorder samples at most two saved frames per second. Image tracking should
  use the live ARFrame stream independently of that export sampling rate.
- `server/app/pipeline/arkit_pipeline.py` uses recorded poses; camera relocalization
  does not need to be replaced by OpenCV SfM to add food selection.
- `server/app/pipeline/colmap_pipeline.py` reconstructs the images before applying
  the region crop and `background_removal.py` plane/largest-cluster heuristic.
  That heuristic has no food semantics and can discard disconnected food pieces
  or keep a plate instead of the selected food.
- `native_import.py` installs image files and `arkit.json`; it does not preserve
  an exported mask dataset today. Adding masks only to the phone ZIP is
  insufficient: validation, import, re-export and reconstruction must also change.

## Where OpenCV fits

| Capability | Proposed use | Boundary |
| --- | --- | --- |
| GrabCut (`imgproc`) | Interactive foreground baseline seeded by a coarse selection and foreground/background corrections. | Separates appearance, not food categories; similar-looking food and plate can merge. |
| Morphology, contours, connected components (`imgproc`) | Clean masks, extract outlines and reject isolated specks. | Preserve multiple user-selected components; do not replace the current largest-cluster assumption with another largest-component rule. |
| Pyramidal Lucas-Kanade flow (`video`) | Track selected image features between less frequent segmentation requests; use tracked points to propose mask motion. | Flow tracks image points, not a food identity or an entire reliable mask. Validate and resegment; features can drift onto a plate. |
| DNN (`dnn`) | Run a compatible trained detection, segmentation or classification network, including suitable ONNX exports. | Needs model weights and target-food evaluation; OpenCV's DNN API supplies inference, not a ready universal food recognizer. |
| Laplacian/gradient measures and image statistics (`imgproc`) | Candidate blur, exposure and redundancy scores for frame selection. | Thresholds depend on lighting, resolution and texture. They need calibration and cannot replace view-coverage checks. |
| ArUco detection and calibrated pose estimation (`objdetect`, `calib3d`) | Optional known-size marker beside the plate for geometric cross-checks and repeatable benchmarks. | Marker-derived scale/pose does not measure food height or identify the food. Keep markers optional. |

Apple's foreground instance masks are a useful first baseline, but may select an
entire plate or other prominent object. They are not guaranteed food segmentation.
If individual foods must be separated, evaluate a food-specific instance model
and allow user correction. Classification of a selected crop is a separate task;
present uncertain alternatives or Unknown rather than silently committing a label.

## Proposed phone flow

1. Find a table or plate support plane with ARKit as today.
2. Tap the desired food or plate. Generate candidate instance masks and choose the
   mask containing the tap; offer correction if it selects the plate or merges
   foods. Draw the selection as a 2D outline over the live camera image.
3. Assign an app-owned target ID. Track the selection, periodically resegment and
   associate the new mask with the existing target using appearance, overlap and
   plausible motion. Keep semantic-label confidence separate from tracking and
   mask quality. Stop trusting the selection after occlusion, large drift or
   loss, and offer reacquisition rather than jumping to another object.
4. Use raycasts on visible support-plane areas around the selection, not an
   assumption that a food-mask center returns the food's surface. Associate a
   tentative AR anchor and crop region with the confirmed support plane.
5. Save useful views with image-aligned masks, target IDs and confidence/quality
   metadata, alongside the existing calibrated RGB and camera poses. Food and
   plate remain stationary during the reconstruction capture.

For a bounded workload, begin testing segmentation at 1–3 Hz and cheaper tracking
at a higher rate. These are experiment settings, not demonstrated iPhone 16
throughput. Use a dedicated serial queue off the UI thread, one inference in
flight and a latest-frame policy. Drop stale work instead of accumulating frames.
Tag results with the source frame timestamp and selection generation so old
results cannot be applied after switching targets, resetting or resuming AR.

## Image coordinates and raycasting

The preview is portrait/aspect-filled while saved sensor images are unrotated.
OpenCV, Vision, the preview and ARKit use differing image/orientation conventions.
Define and test the transforms among model input, mask pixels, native image
pixels, normalized preview coordinates and raycast points. Include model resize,
crop/letterboxing, orientation and Vision origin conventions. Use the appropriate
ARFrame display transform for the supported OS/API; do not assume screen pixels
and sensor pixels coincide.

Keep RGB, mask, intrinsics and camera pose associated with the same source frame.
Store masks in a documented image coordinate system and preserve transforms when
resampling. Any actual reconstruction-image crop/resize also requires adjusting
the camera intrinsics and downstream image references.

The iPhone 16 has no LiDAR scene-depth buffer. A ray's intersection with the
confirmed support plane locates a point on that plane. Projecting a silhouette
onto it produces a provisional support-plane projection, not the true footprint
of elevated food, its top surface, height or volume. Treat it as guidance and a
coarse crop; use multiview reconstruction or an explicitly evaluated depth model
for shape. Never encode plane intersections as measured food-depth samples.

## Backend changes and performance

1. Add a versioned recording contract with optional image-aligned label masks,
   selection metadata, model version and confidences. Preserve support for old
   RGB/pose-only recordings. Validate mask paths, decoded dimensions, allowed
   labels and archive size/count limits; copy and rename masks consistently with
   frames during import. A label image per frame can represent several foods
   without multiplying ZIP entries for every instance.
2. Preserve useful RGB context for reconstruction. Do not simply paint the
   background black: sharp synthetic edges and loss of texture can harm matching.
   Evaluate foreground constraints and mask-aware point/fusion filtering with
   visibility and multi-view agreement. Retain uncertain geometry conservatively.
3. Replace largest-cluster-only food selection with mask/target evidence. Retain
   disconnected pieces belonging to the chosen target and separate food from
   plate/support geometry. Measure missing food as well as retained background.
4. Improve speed through useful-frame selection and bounded image resolution,
   tested against geometry quality. Use ARKit translation/view-angle diversity,
   overlap and mask coverage in addition to sharpness. Stationary frames with
   nearly identical viewpoints are candidates to omit.
5. Only claim ROI speedups once dense processing really uses smaller calibrated
   images or a supported restricted computation region. Masks applied after MVS
   or feature-extraction masks alone do not automatically reduce the full-image
   PatchMatch workload. COLMAP's documented image masks are for feature extraction.

The largest immediate opportunities are better target selection and better
capture inputs. OpenCV alone does not make dense MVS fast, supply LiDAR depth,
recover textureless/specular food reliably, or guarantee metric accuracy.

## Milestones and acceptance evidence

1. **Selection and tracking:** tap a food/plate, display and correct its mask,
   maintain target identity during a short orbit, expose loss/reacquisition and
   export/re-import aligned masks. Keep the existing manual box fallback.
2. **Food-aware reconstruction:** connect masks to foreground selection, preserve
   disconnected foods, and compare the same captures with the existing crop and
   plane-removal pipeline. A thin food item must not disappear as a support plane.
3. **Identification:** benchmark a chosen model on the target food categories,
   multiple foods, unseen foods and mixed dishes; add user-confirmed labels and
   reject uncertain predictions. Determine model and licensing choices here.
4. **Runtime tuning:** compare frame counts/resolutions and any dense ROI approach
   with the unmodified capture. Report stage timings and completeness together,
   not speed alone.

Use held-out real captures including white food on white plates, glossy food,
utensils/hands, occlusion, multiple disconnected items and target switching.
Record mask overlap against manually labeled frames, plate/background leakage,
identity switches, reacquisition success, inference latency, dropped recording
frames, thermal behavior, geometric completeness and dimensional error against
physical references. Set quantitative gates after establishing the baseline;
there are no measured gains claimed by this proposal.

## Primary sources

- [OpenCV library](https://opencv.org/)
- [Rust OpenCV crate](https://docs.rs/opencv/latest/opencv/) and [repository](https://github.com/twistedfall/opencv-rust)
- [GrabCut](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html)
- [Morphological operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) and [Rust Lucas-Kanade binding](https://docs.rs/opencv/latest/opencv/video/fn.calc_optical_flow_pyr_lk.html)
- [OpenCV DNN](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)
- [ArUco detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [Vision foreground instance masks](https://developer.apple.com/documentation/vision/vngenerateforegroundinstancemaskrequest)
- [Vision/Core ML classification](https://developer.apple.com/documentation/coreml/classifying-images-with-vision-and-core-ml)
- [Vision with ARKit](https://developer.apple.com/documentation/arkit/tracking-and-altering-images)
- [ARKit scene depth](https://developer.apple.com/documentation/arkit/ardepthdata)
- [COLMAP FAQ, including image masks](https://colmap.github.io/faq.html)
