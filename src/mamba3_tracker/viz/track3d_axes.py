"""One shared camera convention for every 3D point-track figure.

All 3D track plots (paper Fig. 1(b) teaser, paper Fig. 13 / memo qualitative grid)
must show the world axes pointing in the *same* screen directions, and those
directions should read like the input image so a reader can map a 3D track back
onto the video frame without re-learning the orientation per figure:

    +X -> screen right          (image u)
    +Y -> screen straight down  (image v; TAPVid-3D camera frame has Y down)
    +Z -> 45 deg up-right       (depth, receding into the scene)

so the low corner of the box sits at the top left, exactly like an image origin.

Why X is 8 deg off horizontal instead of exact
----------------------------------------------
A true orthographic projection cannot give all three of those exactly. Writing
the projection's screen-right / screen-up rows as r1, r2 (orthonormal), the
projected axes are p_i = (r1_i, r2_i), and "X exactly horizontal" plus "Y exactly
vertical" forces r2_1 = r1_2 = 0, i.e. r1 = (a, 0, c) and r2 = (0, b, d); then
orthogonality r1 . r2 = c d = 0 kills c or d, so p_Z collapses onto p_X or p_Y
and depth disappears. Something has to bend. Technical drawing dodges this with
an oblique projection, which mplot3d has no equivalent of (`set_box_aspect`
rescales each axis but cannot rotate one, so projected directions are unchanged).

So we pin the two the eye judges hardest -- Y exactly vertical and Z exactly at
45 deg -- and spend the whole error on X's tilt. With Y vertical (r1_2 = 0) and
p_Z = m (cos45, sin45), the tilt follows from orthogonality alone:

    c = d = m cos45,  a = sqrt(1 - c^2),  v = -c d / a,  k = sqrt(1 - v^2 - d^2)
    r1 = (a, 0, c)         p_X = (a, v)  ->  tan(tilt) = (m^2 / 2) / (1 - m^2 / 2)
    r2 = (v, -k, d)        p_Y = (0, -k) ->  exactly down
                           p_Z = (c, d)  ->  exactly 45 deg, length m

`m` (how long a depth unit draws relative to a width unit) is the one free knob,
and it trades X's tilt against how much the depth spread of the data survives:
m = 0.4 flattens X to 5 deg but squashes the tracks into a line, m = 0.6 opens
the data up but slants X by 12 deg. m = 0.5 is the chosen balance -- an 8.1 deg
X tilt at a depth-to-width ratio of 0.53, which is also roughly the ratio a
cabinet projection uses. It lands on round camera angles:

    view direction = r1 x r2 = (0.3273, -0.3780, -0.8660)
        -> elev = asin(-0.8660) = -60 deg,  azim = atan2(-0.3780, 0.3273) = -49.107 deg
    roll = -(angle from matplotlib's unrolled right vector to r1) = -45 deg

`proj_type("ortho")` matters: under the default perspective projection the axis
directions drift across the box, so "Y is vertical" would only hold near the
centre.
"""

from __future__ import annotations

import math

from matplotlib.ticker import MultipleLocator

# Fixed for every 3D track figure -- do not expose as a CLI knob, or the figures
# drift apart again (paper Fig. 1(b) and Fig. 13 previously used different views,
# and Fig. 1(b) additionally permuted the axes).
CAMERA_ELEV = -60.0
CAMERA_AZIM = -49.106605350869
CAMERA_ROLL = -45.0


def apply_image_like_view(ax) -> None:
    """Point `ax` (a mplot3d Axes3D) at the shared image-like camera."""
    ax.set_proj_type("ortho")
    ax.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM, roll=CAMERA_ROLL)


def _shared_tick_step(span: float, min_ticks: int = 3) -> float:
    """Largest 1/2/2.5/5-decade step giving at least `min_ticks` labels on `span`.

    3, not 4: at 4 the teaser's ~16 m cube picks a 2.5 m step, and 6 labels per
    axis overlap once the panel is scaled down to its printed width. 5 m fits with
    room to spare, and the qualitative clips are unaffected (drivetrack still
    steps at 1 m) because their cubes are small enough that the extra tick was
    never what set their interval.
    """
    target = max(span, 1e-9) / min_ticks
    decade = 10.0 ** math.floor(math.log10(target))
    for mult in (5.0, 2.5, 2.0):
        if mult * decade <= target:
            return mult * decade
    return decade  # 1.0 * decade <= target always holds, by construction of decade


def apply_equal_cube(ax, lims, min_ticks: int = 3) -> float:
    """Frame the data in an equal-span cube, as tight as possible, and return the
    tick interval.

    `lims` is [(lo, hi)] * 3 of the data to frame. All three axes get the *same*
    span -- the largest of the three, centred on each axis's own midpoint -- with a
    cubic box aspect. Equal span is a hard requirement: it is what lets a reader
    compare an extent along X directly against one along Z, since a gridline step
    is the same number of metres and the same number of pixels on every axis.

    The cube is exactly the largest data span, with no additional padding, so the
    data touches the box faces on its longest axis. That is the whole zoom budget
    available: the cube must contain the longest axis, so a clip whose extents are
    anisotropic will always show its short axes only partly filled (drivetrack
    spans 1.2 m in X and Y against 5.4 m in Z, so X and Y fill ~22% however tight
    the cube is). Filling every axis would mean giving each its own span, which
    breaks equal span, or cropping real data.
    """
    mids = [0.5 * (lo + hi) for lo, hi in lims]
    half = max(hi - lo for lo, hi in lims) / 2.0
    for setter, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mids):
        setter(mid - half, mid + half)
    ax.set_box_aspect((1, 1, 1))
    step = _shared_tick_step(2.0 * half, min_ticks)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MultipleLocator(step))
    return step
