"""Synthetic checks of the stabilised view: the camera angles, the corridor path, the view mapping. Run: python -m pytest tests"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dronerange import stabilise as S  # noqa: E402
from dronerange.camera import Camera  # noqa: E402

LENS_2244 = dict(fx=4843.65, fy=4839.80, cx=1057.83, cy=756.90, dist=[0.0319, 0.2332, 0.0002, 0.0009, -2.383], width=2048, height=1500)


class _Sess:
    """a camera 24 deg below the horizon that shakes by about 0.5 deg from frame to frame"""

    def __init__(self, n=80, seed=0):
        rng = np.random.default_rng(seed)
        self.cam = Camera.from_calib(LENS_2244)
        self.n_frames = n
        self.ft = 1.0e9 + np.arange(n) * 0.2
        az = np.radians(30 + 0.2 * np.arange(n) + rng.normal(0, 0.5, n))        # a slow pan plus the shake
        el = np.radians(-24 + rng.normal(0, 0.5, n))
        roll = np.radians(3 + rng.normal(0, 0.2, n))
        self.R = S.R_from_angles(az, el, roll)

    def R_wc(self, t):
        return self.R[int(round((t - self.ft[0]) / 0.2))]


def test_angles_roundtrip():
    rng = np.random.default_rng(1)
    az, el, roll = rng.uniform(-3, 3, 50), rng.uniform(-1.2, 1.2, 50), rng.uniform(-0.3, 0.3, 50)
    R = S.R_from_angles(az, el, roll)
    assert np.allclose(np.einsum("nji,njk->nik", R, R), np.eye(3), atol=1e-12)          # rotations
    assert np.allclose(np.linalg.det(R), 1.0)
    a2, e2, r2 = S.angles_from_R(R)
    assert np.allclose(S.R_from_angles(a2, e2, r2), R, atol=1e-12)


def test_corridor_path_is_smooth_and_inside():
    rng = np.random.default_rng(2)
    a = np.cumsum(rng.normal(0, 0.02, 300)) + rng.normal(0, 0.01, 300)
    half = 0.015
    v = S.smooth_in_corridor(a, half)
    assert np.all(np.abs(v - a) <= half + 1e-9)
    assert np.sum(np.diff(v, 2) ** 2) < 0.15 * np.sum(np.diff(a, 2) ** 2)          # the corridor is narrower than one step


def test_stabiliser_maps_and_margins():
    sess = _Sess()
    st = S.Stabiliser(sess, zoom=1.25, turn_share=0.6, w1=1.0)
    real = np.array(S.angles_from_R(st.Rw)); virt = np.array(S.angles_from_R(st.Rv))
    assert np.sum(np.diff(virt[1], 2) ** 2) < 0.1 * np.sum(np.diff(real[1], 2) ** 2)   # the elevation shake is gone
    assert max(st.coverage(i) for i in range(sess.n_frames)) == 0.0                  # the frame always fills the view
    p = np.array([[100.0, 200.0], [1500.0, 1200.0], [1024.0, 750.0]])
    for i in (0, 40, 79):
        assert np.abs(st.to_camera(i, st.to_view(i, p)) - p).max() < 1e-6
    img = np.zeros((750, 1024, 3), np.uint8)
    assert st.warp(img, 10).shape == (938, 1280, 3)
    st.shift[:] = (5.0, -3.0)                                       # the residual shift moves the content by -shift
    q0 = st.to_view(20, p)
    st2 = S.Stabiliser(sess, zoom=1.25, turn_share=0.6, w1=1.0)
    assert np.allclose(st2.to_view(20, p) - q0, [5.0, -3.0])
