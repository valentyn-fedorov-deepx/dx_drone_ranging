"""Synthetic checks of the OpenCV lens model, the ground-plane tracker (ids, velocity, the moving test), the heading
correction and the map geometry. Run: python -m pytest tests"""
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dronerange import heading, worldtrack  # noqa: E402
from dronerange.camera import Camera  # noqa: E402
from dronerange.mapview import TileMosaic, latlon_to_global_px  # noqa: E402

LENS_2244 = dict(fx=4843.65, fy=4839.80, cx=1057.83, cy=756.90, dist=[0.0319, 0.2332, 0.0002, 0.0009, -2.383], width=2048, height=1500)


def test_opencv_lens_matches_cv2():
    cam = Camera.from_calib(LENS_2244)
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(-6, 6, 50), rng.uniform(-4, 4, 50), rng.uniform(25, 60, 50)]
    ref, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), cam.K, np.array(cam.dist))
    assert np.abs(cam.project(P) - ref.reshape(-1, 2)).max() < 1e-6
    px = np.c_[rng.uniform(0, 2047, 200), rng.uniform(0, 1499, 200)]
    assert np.abs(cam.project(cam.bearings(px)) - px).max() < 1e-4           # iterative inverse
    u = cam.undistort(px)
    b = cam.bearings(px)
    assert np.allclose((u[:, 0] - cam.cx) / cam.f, b[:, 0] / b[:, 2]) and np.allclose((u[:, 1] - cam.cy) / cam.fy, b[:, 1] / b[:, 2])


class _Sess:
    def __init__(self, n, dt=0.2):
        self.ft = 1.0e9 + np.arange(n) * dt


def _scene(n=150, dt=0.2, seed=0):
    """a person walking east at 1.4 m/s past a parked car; the drone hovers 30 m south; ranging noise 0.3 m"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        t = i * dt
        rows.append(dict(frame=i, cls="person", conf=0.8, track=7 if i < 80 else 31, x1=0, y1=0, x2=10, y2=30,
                         Er=-10 + 1.4 * t + rng.normal(0, 0.3), Nr=30 + rng.normal(0, 0.3), E_drone=0.0, N_drone=0.0, D=30.0))
        if i % 7 != 3:                                                      # the detector misses the car now and then
            rows.append(dict(frame=i, cls="car" if i % 5 else "truck", conf=0.9, track=12, x1=100, y1=0, x2=160, y2=40,
                             Er=8 + rng.normal(0, 0.4), Nr=42 + rng.normal(0, 0.4), E_drone=0.0, N_drone=0.0, D=43.0))
    R = pd.DataFrame(rows)
    fp = {i: [(-60, 5), (60, 5), (60, 90), (-60, 90)] for i in range(n)}
    return R, fp


def test_tracker_ids_velocity_moving():
    R, fp = _scene()
    T = worldtrack.run(_Sess(150), R, fp, log=lambda *a: None)
    seen = T[T.status == "seen"]
    assert seen.id.nunique() == 2                                          # one id each, despite the image tracker's id change
    p = seen[seen.grp == "person"]; c = seen[seen.grp == "vehicle"]
    late = p[p.t > 6]
    assert abs(late.speed.median() - 1.4) < 0.2
    assert abs(math.degrees(math.atan2(late.vE.median(), late.vN.median())) - 90) < 10
    assert late.moving.mean() > 0.95
    assert c[c.t > 3].moving.mean() < 0.05 and c[c.t > 3].speed.median() < 0.3
    assert c.cls.iloc[-1] == "car"                                          # majority class, not the occasional "truck"


def test_tracker_reports_every_detection_as_it_is():
    """the hits before a track is confirmed and a lone low-confidence detection come out without an id (status
    tentative): every detection is reported once and nothing is filled in from later frames"""
    R, fp = _scene(n=30)
    lone = dict(frame=12, cls="person", conf=0.1, track=-1, x1=300, y1=0, x2=310, y2=30, Er=-40.0, Nr=60.0, E_drone=0.0, N_drone=0.0, D=72.0)
    R = pd.concat([R, pd.DataFrame([lone])], ignore_index=True)
    T = worldtrack.run(_Sess(30), R, fp, log=lambda *a: None)
    det = T[T.det_row >= 0]
    assert det.det_row.nunique() == len(R) and not det.det_row.duplicated().any()
    p = T[(T.grp == "person") & (T.det_row >= 0) & (T.det_row != len(R) - 1)].sort_values("frame")
    assert (p.head(2).status == "tentative").all() and (p.head(2).id == -1).all()
    assert p.iloc[2].status == "seen" and p.iloc[2].id > 0
    row = T[T.det_row == len(R) - 1]
    assert len(row) == 1 and row.status.iloc[0] == "tentative" and row.id.iloc[0] == -1


def test_tracker_remembers_out_of_view():
    """a parked car leaves the footprint for 40 s and comes back: same id"""
    rng = np.random.default_rng(3)
    rows, fp = [], {}
    for i in range(400):
        visible = i < 60 or i > 260
        if visible:
            rows.append(dict(frame=i, cls="car", conf=0.9, track=1 if i < 60 else 99, x1=0, y1=0, x2=50, y2=30,
                             Er=20 + rng.normal(0, 0.4), Nr=40 + rng.normal(0, 0.4), E_drone=0.0, N_drone=0.0, D=45.0))
        fp[i] = [(-50, 5), (50, 5), (50, 90), (-50, 90)] if visible else [(-50, -5), (50, -5), (50, -90), (-50, -90)]
    T = worldtrack.run(_Sess(400), pd.DataFrame(rows), fp, log=lambda *a: None)
    assert T[T.status == "seen"].id.nunique() == 1
    assert (T[(T.frame > 100) & (T.frame < 250)].status == "memory").all()


def test_heading_correction_recovers_offset():
    class S:
        pass
    s = S()
    n = 9000
    s.ti = 1.0e9 + np.arange(n) * 0.011
    true_yaw = 10 + 2.0 * (s.ti - s.ti[0])                                  # deg, turning at 2 deg/s
    s.gyr = np.c_[np.zeros(n), np.zeros(n), np.full(n, math.radians(2.0))]
    s.bias = np.zeros(3); s.gyro_scale = 1.0; s.toff = 0.0
    xs = true_yaw + 40 + 5 * np.sin((s.ti - s.ti[0]) / 7)                  # magnetic yaw: off by 40 deg and wandering
    s.eul = np.c_[np.zeros(n), np.zeros(n), xs]
    s.ft = s.ti[::20] + 0.003
    k = np.arange(10, len(s.ft) - 5, 2)
    tk = s.ft[k]
    gdir = np.random.default_rng(0).uniform(-180, 180, len(k))
    err = np.interp(tk, s.ti, xs) - np.interp(tk, s.ti, true_yaw)           # ENU yaw error -> bearing error of the image direction
    Pd = pd.DataFrame(dict(i=k, t=tk - s.ft[0], dir_gnss=gdir, dir_img=heading.wrap(gdir - err), ang_n=1.0, rank1=0.05, inl=200, h=30.0, dC_h=3.0))
    good = dict(max_normal_deg=4, max_rank1=0.15, min_inliers=60, h_range=[2, 200], min_disp_m=2.0)
    model, A, T = heading.fit(s, Pd, good, log=lambda *a: None)
    m = (s.ti > s.ti[0] + 5) & (s.ti < s.ti[-1] - 5)
    assert np.abs(heading.wrap(model.yaws(s.ti[m]) - true_yaw[m])).max() < 0.5


def test_map_geometry():
    x, y = latlon_to_global_px(0.0, 0.0, 1)
    assert abs(x - 256) < 1e-9 and abs(y - 256) < 1e-9
    mo = TileMosaic(np.zeros((512, 512, 3), np.uint8), 0, 0, 19, 28.5536233, -81.5673233)
    x0, y0 = mo.en_to_px(0, 0)
    x1, y1 = mo.en_to_px(100, 50)
    assert abs((x1 - x0) * mo.mpp - 100) < 1e-6 and abs((y0 - y1) * mo.mpp - 50) < 1e-6
    # the mosaic's metres per pixel agree with the Mercator scale at that latitude
    lat1 = 28.5536233 + math.degrees(100 / 6371008.8)
    _, ya = latlon_to_global_px(28.5536233, -81.5673233, 19); _, yb = latlon_to_global_px(lat1, -81.5673233, 19)
    assert abs((ya - yb) * mo.mpp - 100) < 0.2
