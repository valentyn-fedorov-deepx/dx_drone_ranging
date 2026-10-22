"""Synthetic checks of the drone ranging geometry: camera model, ray-ground intersection, the calibration residual
model, height from a ground homography + horizontal displacement, vertical odometry sign, the height-track plan,
and the Xsens record layout. Run: python -m pytest tests"""
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dronerange.altitude import DOWN, build_track, plane_motion  # noqa: E402
from dronerange.camera import Camera, intersect_ground, intersect_terrain, world_rays  # noqa: E402
from dronerange.xsens import RECORD, read_bin  # noqa: E402

CAM = Camera(f=4926.3, cx=1023.5, cy=749.5, k1=0.058, width=2048, height=1500)


def look_down(depression_deg, heading_deg=0.0):
    """R_wc for a camera whose optical axis points along heading (from north, clockwise) and depression below the horizon"""
    z = np.array([math.sin(math.radians(heading_deg)) * math.cos(math.radians(depression_deg)),
                  math.cos(math.radians(heading_deg)) * math.cos(math.radians(depression_deg)),
                  -math.sin(math.radians(depression_deg))])
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)        # image right is horizontal
    y = np.cross(z, x)                                         # image down
    return np.c_[x, y, z]


def test_project_bearing_roundtrip():
    px = np.array([[0, 0], [2047, 0], [1023.5, 749.5], [1500, 1200], [0, 1499]], float)
    back = CAM.project(CAM.bearings(px))
    assert np.abs(back - px).max() < 0.1                       # first-order inverse of the radial term
    assert np.allclose(np.linalg.norm(CAM.bearings(px), axis=1), 1)


def test_undistort_matches_bearings():
    px = np.array([[100, 80], [1900, 1400]], float)
    u = CAM.undistort(px)
    b = CAM.bearings(px)
    assert np.allclose((u[:, 0] - CAM.cx) / CAM.f, b[:, 0] / b[:, 2])
    assert np.allclose((u[:, 1] - CAM.cy) / CAM.f, b[:, 1] / b[:, 2])


def test_ground_distance_centre_ray():
    for h, dep in ((8.0, 20.0), (25.0, 23.0), (35.0, 40.0)):
        R = look_down(dep, heading_deg=37.0)
        dE, dN, ok = intersect_ground(world_rays(CAM, R, [[CAM.cx, CAM.cy]]), h)
        assert ok[0]
        assert abs(math.hypot(dE[0], dN[0]) - h / math.tan(math.radians(dep))) < 1e-6
        assert abs(math.degrees(math.atan2(dE[0], dN[0])) - 37.0) < 1e-6


def test_ground_point_reprojects():
    R = look_down(25.0, heading_deg=-80.0)
    h = 30.0
    px = np.array([[300, 1300], [1700, 900], [1000, 200]], float)
    dE, dN, ok = intersect_ground(world_rays(CAM, R, px), h)
    P = np.c_[dE, dN, np.full(3, -h)]                         # camera at the origin
    assert np.abs(CAM.project(P @ R) - px).max() < 0.1


def test_ray_above_horizon_is_invalid():
    R = look_down(5.0)
    dE, dN, ok = intersect_ground(world_rays(CAM, R, [[CAM.cx, 0.0]]), 20.0)   # top of a frame tilted 5 deg down: above horizon
    assert not ok[0] and np.isnan(dE[0])


def test_terrain_flat_equals_plane():
    R = look_down(20.0, heading_deg=10.0)
    px = np.array([[500, 1000], [1500, 1100]], float)
    rays = world_rays(CAM, R, px)
    dE, dN, ok = intersect_ground(rays, 25.0)
    flat = lambda e, n: np.full(np.shape(e), 38.0)
    E, N, Z, ok2 = intersect_terrain(rays, 0.0, 0.0, 38.0 + 25.0, flat, step=0.5)
    assert ok2.all()
    assert np.allclose(E, dE, atol=1e-3) and np.allclose(N, dN, atol=1e-3) and np.allclose(Z, 38.0, atol=1e-3)


def test_terrain_slope_shortens_uphill():
    R = look_down(20.0)                                        # looking north
    rays = world_rays(CAM, R, [[CAM.cx, CAM.cy]])
    slope = lambda e, n: 0.05 * np.asarray(n)                  # ground rises 5 % to the north
    E, N, Z, ok = intersect_terrain(rays, 0.0, 0.0, 25.0, slope, step=0.5)
    d_flat = 25.0 / math.tan(math.radians(20.0))
    d_true = 25.0 / (math.tan(math.radians(20.0)) + 0.05)
    assert ok[0] and abs(N[0] - d_true) < 1e-3 and N[0] < d_flat


def test_calibration_rotation_model():
    """b2 = R_bc^T dR^T R_bc b1 for a fixed world direction seen before and after a body rotation dR"""
    rng = np.random.default_rng(1)
    R_ref1 = Rot.random(random_state=2).as_matrix()
    dR = Rot.from_rotvec(np.radians([2.0, -1.0, 3.0])).as_matrix()
    R_bc = Rot.random(random_state=3).as_matrix()
    v = rng.normal(size=(20, 3))
    R_ref2 = R_ref1 @ dR
    c1 = v @ R_ref1 @ R_bc                                     # camera-frame coordinates at t1: R_bc^T R_ref1^T v
    c2 = v @ R_ref2 @ R_bc
    assert np.allclose(c2, c1 @ (R_bc.T @ dR.T @ R_bc).T)


def _homography_pair(h=30.0, dC=(3.0, 1.5, 0.4), dep=25.0, turn_deg=(0.5, -0.8, 1.2)):
    """camera 1 at height h over the plane z = 0; camera 2 displaced by dC (world) and rotated by turn_deg"""
    Rwc1 = look_down(dep, heading_deg=50.0)
    Rwc2 = Rot.from_rotvec(np.radians(turn_deg)).as_matrix() @ Rwc1
    C1 = np.array([0, 0, h]); C2 = C1 + np.array(dC)
    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < 200:                                     # ground points seen by both cameras
        px = rng.uniform([0, 0], [2047, 1499])
        b = world_rays(CAM, Rwc1, [px])[0]
        X = C1 + b * (h / -b[2])
        p2 = CAM.project(((X - C2) @ Rwc2)[None])[0]
        if 0 <= p2[0] < 2048 and 0 <= p2[1] < 1500:
            pts.append((px, p2))
    p1 = CAM.undistort(np.array([a for a, _ in pts])); p2 = CAM.undistort(np.array([b for _, b in pts]))
    return Rwc1, Rwc2, C1, C2, p1, p2


def test_height_from_homography_and_displacement():
    h = 30.0
    Rwc1, Rwc2, C1, C2, p1, p2 = _homography_pair(h)
    Hm, _ = cv2.findHomography(p1, p2, 0)
    R21 = Rwc2.T @ Rwc1                                       # camera-1 -> camera-2 directions (from the gyro in the pipeline)
    n = Rwc1.T @ DOWN
    K = CAM.K
    u, S, Vt = plane_motion(Hm, K, np.linalg.inv(K), R21, n)
    u_w = Rwc1 @ u
    dC_h = math.hypot(*(C2 - C1)[:2])
    assert abs(dC_h / math.hypot(u_w[0], u_w[1]) - h) < 0.05   # height from the horizontal GNSS displacement
    assert abs(-u_w[2] - (C2 - C1)[2] / h) < 1e-3               # vertical odometry: dz / h, camera moved up -> positive
    assert math.degrees(math.acos(min(1.0, abs(Vt[0] @ n)))) < 1.0   # plane normal agrees with the IMU normal
    assert S[1] / S[0] < 1e-3                                   # rank one


def test_height_track_plan():
    class S:
        tt = np.arange(0, 100, 0.5)
        ft = tt + 1e9
    vo_a = pd.DataFrame(dict(t=np.arange(10, 40, 0.4), h_open=np.linspace(8, 20, 75)))
    vo_b = pd.DataFrame(dict(t=np.arange(60, 100, 0.4), h_open=np.linspace(10, 9, 100)))
    gs = pd.Series(np.full(20, 30.0), index=np.linspace(45, 60, 20))
    plan = [dict(until=10, kind="const", value=8.0), dict(until=40, kind="vo", run="a"), dict(until=45, kind="ramp"),
            dict(until=60, kind="pairs"), dict(until=None, kind="vo_scaled", run="b")]
    T = build_track(S, plan, gs, {"a": vo_a, "b": vo_b})
    h = dict(zip(T.t, T.h))
    assert h[5.0] == 8.0 and abs(h[10.0] - 8.0) < 1e-9
    assert abs(h[42.5] - 0.5 * (np.interp(40, vo_a.t, vo_a.h_open) + 30.0)) < 1e-9   # ramp midpoint
    assert h[50.0] == 30.0 and abs(h[60.0] - 30.0) < 1e-9                          # odometry continues the pairs height
    assert abs(h[99.5] / h[60.0] - np.interp(99.5, vo_b.t, vo_b.h_open) / np.interp(60, vo_b.t, vo_b.h_open)) < 1e-9
    assert not T.h.isna().any()


def test_xsens_record_roundtrip(tmp_path):
    a = np.zeros(3, dtype=RECORD)
    a["t_ns"] = [1792512803922715822, 1792512803931884065, 1792512803940000000]
    a["gx"] = [0.1, 0.2, 0.3]; a["lat"] = 28.5536233; a["lon"] = -81.5673233; a["yaw"] = [-25.8, -25.7, -25.6]
    p = tmp_path / "xsens_1.bin"
    p.write_bytes(a.tobytes() + b"\x00" * 7)                   # a torn last record is ignored
    d = read_bin(p)
    assert RECORD.itemsize == 115 and len(d) == 3
    assert d.t_ns.iloc[1] == 1792512803931884065 and abs(d.gx.iloc[2] - 0.3) < 1e-6 and abs(d.lat.iloc[0] - 28.5536233) < 1e-12
