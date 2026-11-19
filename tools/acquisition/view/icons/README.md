# reachAQ icon, v2

The ReachX mouse inside a camera, in the ReachX mark's style: a solid ReachX-blue
(`#0b95d3`) body with a darker rim, viewfinder and shutter (`#076a99`), a white
lens ring, pale-blue glass (`#d6ecf8`), and the blue mouse with its details as
negative space. The mouse faces a pellet on the
spoon, and a short trajectory of per-frame samples runs from its forepaws to the
pellet.

The mouse is traced from the ReachX desktop logo
(`%LOCALAPPDATA%\ReachX\tools\reachx-logo.png`), with the hand removed. It is one
even-odd path, so the ear rings, eye, haunch line and forepaw gaps are true holes.

## Files

| File | Use |
|---|---|
| `reachaq.svg` | The icon at every size. Install as `hicolor/scalable/apps/reachaq.svg`. |
| `png/reachaq-<N>.png` | Transparent PNGs, 16-512 px, all from `reachaq.svg`. Install each as `hicolor/<N>x<N>/apps/reachaq.png`. |

No small-size variant is needed: the single-colour artwork still reads at 24 px,
and at 16 px the camera outline carries it.

## Construction

- 512 x 512 canvas, transparent outside the camera body.
- Flat fills only: no gradients, patterns, clip paths, masks, filters, text or
  embedded images. Verified in QtSvg and Blink.
- Lens centre (256, 292), ring radius 146 with a 14-unit white stroke.

Regenerate the PNGs from the SVG if it is edited.
