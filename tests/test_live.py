"""Synthetic checks of the causal heading from the fast-pair anchors. Run: python -m pytest tests"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dronerange import live_height  # noqa: E402


class _Sess:
    def __init__(self, n=2000, dt=0.05):
        self.ti = 1.0e9 + np.arange(n) * dt
        self.gyr = np.zeros((n, 3)); self.bias = np.zeros(3); self.gyro_scale = 1.0
        self.eul = np.zeros((n, 3)); self.eul[:, 2] = 40.0               # magnetic yaw 40 deg, still
        self.toff = 0.0


def test_heading_track_is_causal_and_robust():
    s = _Sess()
    t = s.ti
    A = pd.DataFrame(dict(t_imu=t[[400, 420, 440, 460, 480]], c=[-25.0, -24.0, -26.0, 60.0, -25.0]))   # one outlier
    H = live_height.heading_track(s, A, window_s=120.0, min_n=3)
    assert np.allclose(H.c.values[:440], 0.0)                         # nothing before the third anchor
    assert abs(H.c.values[445] + 25.0) < 1e-9                         # median of the first three
    assert abs(H.c.values[-1] + 25.0) < 0.6                           # the outlier at 60 deg is left out
    assert np.allclose(H.yaw_used.values, H.yaw_gyro.values + H.c.values)


def test_live_heading_known_after_three_anchors():
    import types
    est = types.SimpleNamespace(anchors=[], c=0.0, heading_known=False)   # the estimator's heading state only
    for n, (t, c) in enumerate([(100.0, -157.0), (101.0, -158.0), (102.0, -157.5)]):
        est.anchors.append(dict(t_imu=t, c=c))
        live_height.LiveHeight._update_heading(est)
        assert est.heading_known == (n == 2)                          # a dot on the map until the third anchor
    assert abs(est.c + 157.5) < 1e-9
