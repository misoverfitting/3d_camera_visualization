# Testing horizontal surface detection

Yellow dots show visual feature points, not detected planes. `Surfaces` counts
horizontal plane anchors. A textured wall can produce many points without a
horizontal plane. Phone testing of build 4 confirmed that horizontal planes can
be detected while cyan patches remain invisible. The rendering failure has not
been reproduced locally on a physical device; its precise cause is unconfirmed.

The updated capture view reconciles its cyan overlays with `ARFrame.anchors`
once per second. This prevents the displayed count from relying only on callbacks
received since view initialization. Surface counts come from the live frame,
independently of successful overlay creation.

The renderer now uses `ARSCNPlaneGeometry` updated from the detected plane's mesh,
with an explicitly created constant cyan material and an opaque boundary outline.
It attaches this mesh under the anchor node supplied by `ARSCNViewDelegate`, so
SceneKit maintains the world transform. Mesh vertices are already in the anchor's
coordinates: no extra plane-center translation or 90-degree rotation is applied.
Overlays are raised 3 mm and drawn without depth testing. The scan box also uses
an explicitly assigned material. SceneKit rendering is explicitly continuous.
These changes address rendering weaknesses; a successful compile alone does not
prove the original failure is resolved.

The title displays the installed native build number. `Surface overlays` reports
created overlay nodes; `In view` counts those whose bounding boxes intersect the
SceneKit camera frustum. Frustum intersection is a diagnostic, not proof that
pixels were drawn or that a surface is unobstructed. Detection without overlays
points to anchor-node/Metal geometry creation; overlays with zero in view point
to placement outside the camera view. Nonzero in-view overlays with no cyan
would point to a remaining rendering/material problem. None of these values
represent measured depth, and no plane is synthesized from feature points.

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
3. Watch for normal tracking, `Surfaces` greater than zero, a cyan patch with an
   outline, and the coaching overlay disappearing. Aim at the detected surface
   and confirm `Surface overlays` and `In view` are nonzero. Tap a patch and verify
   a scan box appears. Move around it and verify the mesh stays on the surface.
4. Use Restart surface detection before recording. Verify the box disappears and
   surfaces are rediscovered. Verify the reset button is disabled during capture
   and saving, and that stopping and exporting still work.
5. Test Skip box: coaching should disappear and capture should work once tracking
   is normal, even if no plane is detected. No region should be saved.
6. If detection remains stuck after 20–30 seconds and one reset, record the build,
   exact tracking message, Surfaces, Surface overlays, In view, Walls, and Feature
   points. If detection works but no cyan appears, capture those same values
   while aimed at the detected table. A screenshot of
   the camera view and status helps distinguish tracking/environment failures
   from a count or rendering failure. A phone test is required to establish the
   result; TypeScript checks and a cloud compile cannot establish AR behavior.

References: [ARFrame anchors](https://developer.apple.com/documentation/arkit/arframe/anchors)
and [horizontal-plane coaching](https://developer.apple.com/documentation/arkit/arcoachingoverlayview/goal-swift.enum/horizontalplane).
Rendering follows Apple's [plane visualization workflow](https://developer.apple.com/documentation/arkit/tracking-and-visualizing-planes)
and [ARSCNPlaneGeometry](https://developer.apple.com/documentation/arkit/arscnplanegeometry).
