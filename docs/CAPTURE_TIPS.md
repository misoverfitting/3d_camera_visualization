# Getting a good capture

Summarized from *Phone-Based 3D Capture: The 2026 Landscape* (Aug 2026) and
built into the app's capture-screen guidance.

**Lighting.** Diffuse, even light is critical - overcast outdoor light is
ideal. Avoid harsh shadows and avoid letting the light change partway
through a capture session.

**Surfaces.** Matte, textured surfaces reconstruct beautifully. Shiny or
transparent objects confuse both photogrammetry and (to a lesser extent)
Gaussian splatting; for a mesh, coat the object with scanning spray (or dry
shampoo) first, or prefer splat mode, which handles glass/reflections much
better.

**Coverage.** Take more photos than feels necessary - 60 to 150+ for a
single object - with heavy overlap, orbiting at multiple heights. Keep the
object still and fill the frame. The app's shot counter turns green once
you've hit the recommended minimum, and periodically prompts you to change
height as you orbit.

**Background.** A little background texture (a tabletop, a textured
surface under the object) genuinely helps camera-pose estimation - an
object floating in front of a blank wall gives structure-from-motion
nothing to anchor distant parallax to. It doesn't need to be dramatic; a
consistent background is enough.

**Scale.** Photogrammetry has no inherent sense of scale. If true
dimensions matter, place a ruler or an object of known size in the scene,
and use the viewer's "Set real-world scale" tool afterward to calibrate the
mesh to metric units.

**Compelling vs. accurate.** If your goal is visual impact and
shareability, choose the splat ("compelling") mode. If your goal is
dimensional accuracy for printing or engineering, choose mesh ("accurate")
mode, and expect small objects with fine detail to need more care (or, for
true sub-millimeter accuracy, a dedicated structured-light scanner - phones
aren't the right tool for that scale).
