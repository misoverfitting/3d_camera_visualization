# On-device scan performance diagnosis

Date: 2026-11-14. Static review; no iPhone timing or field capture available.

## Branch baseline

`codex/meal-scan-performance` starts at `origin/feat/meal-scan-core` (9a51f3f), merges `origin/main` (3297041, including on-device food and reconstruction speed work), then `origin/feat/food-id-diagnostics` (8cdc867). All four fetched refs, including `origin/feat/on-device-food`, are ancestors. Existing local build settings and untracked files are preserved. The remaining local `fix/IY-speed-optimization` commit adds Expo association settings and a design review, not an additional performance implementation; it was not merged over the local settings.

## Why coverage stalls near 45%

Coverage is viewpoint occupancy, not classifier confidence, food-mask area, rim support, or reconstruction completeness. `Orbit.swift` divides occupied cells by 24: 12 azimuth sectors times two elevation bands (15–45° and >45–90°). Eleven cells therefore display as `Int(11 / 24 * 100)` = 45%.

1. **Turning the camera does not change its position around the plate.** The selector uses the direction from the initial plate tap to the camera center. A high orbit can fill at most 12 cells (50%); filling the other half requires physically moving below 45° elevation while keeping the target centered. A tabletop against a wall makes a full orbit inconvenient. Near overhead, azimuth is poorly conditioned and equal sectors do not represent equal useful viewpoint diversity.
2. **Frames consume capacity without increasing coverage.** Any direction separated by at least 8° from all accepted directions is accepted, even inside an already filled cell. These frames consume the 32-frame budget. The app aims for 60% (at least 15 cells) and 18 built views, so local sweeps can exhaust the budget far earlier.
3. **Capacity handling depends on the next pose.** `.full` is checked after motion, framing, distance, elevation and redundancy. At capacity, a redundant pose still returns `.redundant`; the app only auto-finishes on `.full`. This can leave capture apparently stuck until a sufficiently novel valid pose arrives.
4. **Guidance hides the missing information.** Redundancy falls through to generic circling advice. There is no occupied-bin map, per-band progress, or rejection histogram. The distance hint says 20–60 cm while the actual range is 15–100 cm.
5. **Accepted coverage can outpace usable images.** The selector advances before asynchronous grayscale conversion succeeds. `PlateDemo.grayImage` accepts only 1920×1440 buffers; another capture resolution returns nil after the frame has consumed a direction and possibly a cell.

Evidence: the existing `OrbitTests.testOrbitSelectorSpreadsKeyframesAndTracksCoverage` explicitly expects 11/24. A Python translation of the selector's spherical separation, cell assignment and capacity logic reproduced a full 60°-elevation orbit at 50% (18 accepted frames), and repeated local sweeps exhausting 32 frames at 7/24 cells (29%). These examples demonstrate failure modes, not the exact cause of the reported phone run. Motion/projection guards were assumed valid in the translation. Field poses and rejection counts are needed to distinguish the causes.

## Current classification path

1. `PlateDemo.demoConsider` samples poses at most every 0.1 seconds, rejects fast motion, poor tracking, off-center targets, invalid distance/elevation and redundant directions. Accepted frames drive the edge-based rim fit; LiDAR supplements spot heights, not food recognition.
2. `FoodID.considerFoodPhoto` saves the most overhead **accepted** image with camera tilt below 45°. Selection uses tilt only, with no sharpness/exposure score. It encodes the camera buffer as JPEG on the serial demo queue.
3. After finishing and fitting the rim, identification decodes the JPEG, crops the projected plate, rotates clockwise and limits its longest side to 1092 pixels. Food search uses 90% of the fitted rim radius; the 78–90% annulus supplies the presumed bare-plate appearance.
4. `LocalFoodID.identify` downsamples to a 256-pixel analysis side. `PlateItems.segment` estimates median plate Lab color and texture, thresholds food, applies morphology, splits up to four color clusters and extracts connected components. Components below 1.5% of the plate interior are discarded. This is a hand-built segmenter, not a learned food detector.
5. Each region is padded 12.5% per side; everything outside its mask is dimmed to 35%. Vision scales it to 224×224 and runs the bundled FP16 MobileNetV3-Large Core ML classifier (`computeUnits = .all`), with ImageNet normalization baked into the export. There are 226 classes trained for six epochs on Food-101, FoodSeg103 and Nutrition5k pseudo-labels.
6. Every nonempty prediction is accepted as its top class. Regions with the same label are merged, keeping the maximum confidence and up to three spaced points. Points are lifted into world coordinates; an item with no liftable points is omitted from the displayed result.
7. The diagnostics branch adds one whole-plate inference, selects Food-101 dish labels from the same probability vector, renders an overlay, and saves plate/frame/region JPEGs plus predictions to `Documents/FoodScans`. Saving occurs before the success callback. This provides useful evidence but adds latency and does not record stage timings or orbit rejection counts.

## Modeling pain points, in priority order

* **Segmentation limits recall before the classifier runs.** Rice on a white plate, reflections, patterned plates, shadows and food in the reference annulus can defeat the bare-plate assumption. Similar-colored touching foods can merge; multicolored foods can fragment. Small sauces/garnishes disappear through area filtering and morphology. Better classifier weights cannot recover missing proposals.
* **Dish and ingredient classes compete in one softmax.** Whole-dish Food-101 photos and ingredient crops represent different tasks. The whole-plate dish guess only filters the resulting rankings; it is not a separately trained dish head or calibrated conditional probability. Separate task heads or explicit task conditioning deserve evaluation.
* **The relevant recorded accuracy is modest.** `ml/foodclass/metrics.json` epoch 6 reports FoodSeg103 top-1 69.2%, top-5 93.3%; Food-101 85.2%/97.0%; Nutrition5k 71.3%/92.1%. The aggregate 81.4% top-1 is dominated by whole-dish examples. These are dataset classification scores, not live-phone end-to-end item recall or point-placement accuracy. Nutrition5k validation uses teacher pseudo-labels, not independent human truth.
* **No abstention/calibration.** Every crop gets a food name even when it is plate, an unseen food, or an ambiguous mixed region. Maximum confidence when merging can amplify one mistaken fragment. Evaluate a reject option, probability calibration and top-two margin against a labeled phone holdout.
* **Single-image bottleneck.** Orbit images improve rim fitting, but food labels use one photo. The most overhead frame need not be sharp or unoccluded. Rank photo quality separately and evaluate a small number of complementary views before adding more inference cost.
* **Training/inference distribution mismatch.** FoodSeg103 uses annotated masks, while the app uses noisy heuristic masks. Nutrition5k partially addresses that but uses restricted ingredient lists and proxy classes for pseudo-labeling. Evaluate actual app-produced crops and masks by plate/session, including difficult plates and lighting.
* **Merging can conceal distinct items.** Equal top labels collapse regardless of spatial adjacency; confidence comes from the strongest region rather than a calibrated combined estimate. Preserve region identity for evaluation and use spatial evidence for merging.

## Runtime risks to measure

* Model initialization (`LocalFoodClassifier.shared`) happens before dispatching identification to the demo queue; model/bundle loading can block the caller. Cold and warm behavior need separate measurements.
* Rim fitting, grayscale generation, photo JPEG encoding, classification, overlay rendering and diagnostic writes share `tare.demo`. Repeated fits prepare every accumulated image again. Long fitting jobs can delay image building and food work. This is a scheduling risk, not a measured claim that neural inference dominates.
* Classification is serial per region plus one whole-plate pass, with no region-count budget. Four color clusters can still produce many connected components. Large padded crops and full-size masks are materialized before Vision reduces them to 224×224.
* Field logging encodes multiple JPEGs synchronously before success and retains region crops/overlay. Measure logging separately from segmentation and inference; consider an optional diagnostic mode or asynchronous bounded writes after measurements.

## Recommended first implementation and validation

1. Instrument accepted/built frame counts, occupied cells by elevation, rejection totals, capacity state and image-conversion failures. Expose actionable missing-view guidance. Add monotonic stage timings for cold model load, image preparation, segmentation, each inference, lifting, overlay and disk writes; collect p50/p95 on a physical phone.
2. Change selection to prioritize useful unfilled cells, handle capacity explicitly, and count only successfully built frames toward usable coverage. Test dense local sweeps, high-only orbits, crossing elevation boundaries, full/redundant states and failed image builds. Choose completion criteria from rim-fit quality and an achievable capture protocol rather than merely rescaling the percentage.
3. Use exported FoodScans to inspect the plate image, overlay and region inputs separately. Build a manually checked phone holdout grouped by meal/session; score proposal recall, mask quality, per-item classification, abstention, and lift success. Existing diagnostic logs alone cannot explain orbit rejection.
4. Fix the dominant measured errors first: segmentation/reference-ring failures and task confusion before increasing backbone size or training duration. Compare any learned segmenter or multi-view approach against latency and recall together.

No runtime behavior was changed by this diagnosis. Native Swift/ARKit/Core ML execution requires an Apple environment and a physical-device pass; Swift is unavailable in this Windows workspace.
