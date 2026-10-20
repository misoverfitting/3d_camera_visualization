# Testing horizontal surface detection

Yellow dots show visual feature points, not detected planes. `Surfaces` counts
horizontal plane anchors. A textured wall can produce many points without a
horizontal plane. We have not reproduced the reported zero-surface failure on a
physical device, so its underlying cause is still unconfirmed.

The updated capture view reconciles its cyan overlays with `ARFrame.anchors`
once per second, in addition to the anchor callbacks. This prevents the displayed
count from relying only on callbacks received since view initialization. It also
creates the SceneKit scene explicitly, raises surface opacity, and draws surface
patches without depth occlusion. It does not synthesize a detected plane from
feature points or claim that an estimated raycast is measured depth.

Apple's horizontal-plane coaching overlay now provides movement guidance while
there is no horizontal plane or placed box. Coaching is hidden while recording
or using Skip box. It permits taps through to the estimated-plane placement path.
The status reports tracking, horizontal surfaces, vertical walls, and the number
of current feature points. Restart surface detection resets world tracking and
anchors and clears the scan box; it is disabled while recording or saving.

## Physical-device acceptance check

1. Install the new native build through TestFlight. Reloading the app or updating
   the backend cannot install these Swift changes.
2. Use a well-lit matte table or floor with visible texture; a magazine on a table
   can help. Aim downward and translate the phone slowly sideways by 20–40 cm
   while keeping the surface in view, rather than only rotating in place.
3. Watch for normal tracking, `Surfaces` greater than zero, a cyan patch, and the
   coaching overlay disappearing. Tap a patch and verify a scan box appears.
4. Use Restart surface detection before recording. Verify the box disappears and
   surfaces are rediscovered. Verify the reset button is disabled during capture
   and saving, and that stopping and exporting still work.
5. Test Skip box: coaching should disappear and capture should work once tracking
   is normal, even if no plane is detected. No region should be saved.
6. If detection remains stuck after 20–30 seconds and one reset, record the build,
   exact tracking message, Surfaces, Walls, and Feature points. A screenshot of
   the camera view and status helps distinguish tracking/environment failures
   from a count or rendering failure. A phone test is required to establish the
   result; TypeScript checks and a cloud compile cannot establish AR behavior.

References: [ARFrame anchors](https://developer.apple.com/documentation/arkit/arframe/anchors)
and [horizontal-plane coaching](https://developer.apple.com/documentation/arkit/arcoachingoverlayview/goal-swift.enum/horizontalplane).
