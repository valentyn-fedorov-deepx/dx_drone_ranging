"""Absolute tilt check against the satellite imagery: the asphalt band of a road crossing the view.

Along the image's centre column the asphalt rows are found (low saturation, mid grey, wide); their ground distances
are computed with the IMU attitude and with fixed tilts. Along the same bearing from the drone's position (DJI GNSS) the
asphalt band is found in the satellite mosaic. The tilt whose band lands on the imagery's band is the right one.

Usage: python tools/road_check.py <config> <frame> <image png or -> [depression_deg ...]
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronerange import config as C  # noqa: E402
from dronerange.camera import intersect_ground  # noqa: E402
from dronerange.heading import HeadingModel  # noqa: E402
from dronerange.mapview import TileMosaic  # noqa: E402
from dronerange.session import Session  # noqa: E402
from tilt_test import look  # noqa: E402


def asphalt(rgb):
    """grey, not bright, not dark: asphalt"""
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return (hsv[:, 1] < 45) & (hsv[:, 2] > 70) & (hsv[:, 2] < 190)


def band(mask, xs, min_len):
    """longest run of True in mask -> (start, end) values of xs"""
    best, cur = None, None
    for k, m in enumerate(mask):
        if m and cur is None:
            cur = k
        if (not m or k == len(mask) - 1) and cur is not None:
            e = k if m else k - 1
            if best is None or e - cur > best[1] - best[0]:
                best = (cur, e)
            cur = None
    if best is None or best[1] - best[0] < min_len:
        return None
    return xs[best[0]], xs[best[1]]


def main():
    cfg = C.load(sys.argv[1]); fr = int(sys.argv[2])
    fixed = [float(a) for a in sys.argv[4:]] or [30.0]
    s = Session(cfg); s.attach_calib(C.work(cfg, "calib.json"))
    s.heading = HeadingModel.load(C.work(cfg, "heading.csv"))
    h = float(pd.read_csv(C.work(cfg, "altitude_track.csv")).h.values[fr])
    t = s.ft[fr]; E0, N0 = s.drone_EN(t, smooth=True)
    im = cv2.imread(sys.argv[3]) if sys.argv[3] != "-" else s.video().read(fr)
    Rimu = s.R_wc(t); ax = Rimu @ np.array([0, 0, 1.0])
    hd = math.degrees(math.atan2(ax[0], ax[1])); dep = math.degrees(math.asin(-ax[2]))
    cols = np.arange(int(s.cam.cx) - 40, int(s.cam.cx) + 41, 8)
    rows = np.arange(0, im.shape[0])
    m = np.mean([asphalt(im[:, c]) for c in cols], axis=0) > 0.6
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 1), np.uint8)).ravel() > 0
    rb = band(m, rows, 60)
    print(f"frame {fr}: height {h:.1f} m, camera heading {hd:.1f}, IMU depression {dep:.1f} deg; asphalt rows in the image {rb}")
    lat_a, lon_a = s.en_to_latlon(E0 - 200, N0 - 200); lat_b, lon_b = s.en_to_latlon(E0 + 200, N0 + 200)
    mo = TileMosaic.build(lat_a, lat_b, lon_a, lon_b, 19, C.work(cfg, "tiles"), s.lat0, s.lon0)
    for name, Rwc in [(f"IMU {dep:.1f} deg", Rimu)] + [(f"fixed {d:g} deg", look(hd, d)) for d in fixed]:
        px = np.c_[np.full(2, s.cam.cx), [rb[1], rb[0]]]
        dE, dN, ok = intersect_ground(s.cam.bearings(px) @ Rwc.T, h)
        D = np.hypot(dE, dN)
        print(f"  {name:14s}: road band in the image is {D[0]:6.1f} .. {D[1]:6.1f} m from the drone")
    dists = np.arange(10, 160, 0.25)
    b = math.radians(hd)
    x, y = mo.en_to_px(E0 + dists * math.sin(b), N0 + dists * math.cos(b))
    rgb = np.array([mo.img[int(round(yy)), int(round(xx))] for xx, yy in zip(x, y)])
    sm = asphalt(rgb)
    sm = cv2.morphologyEx(sm.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8)).ravel() > 0
    sb = band(sm, dists, 20)
    print(f"  satellite     : asphalt band along bearing {hd:.1f} deg from the drone: {sb}")


if __name__ == "__main__":
    main()
