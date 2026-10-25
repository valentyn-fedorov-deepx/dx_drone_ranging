"""Warp one camera frame onto the satellite map (ground plane, the drone's position and height) for several tilt models
and save them side by side: the model whose roads and kerbs land on the imagery's is the right one.

Usage: python tools/overlay_check.py <config> <frame> [depression_deg ...]
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronerange import config as C  # noqa: E402
from dronerange.heading import HeadingModel  # noqa: E402
from dronerange.mapview import TileMosaic  # noqa: E402
from dronerange.session import Session  # noqa: E402
from tilt_test import look  # noqa: E402


def main():
    cfg = C.load(sys.argv[1])
    fr = int(sys.argv[2])
    fixed = [float(a) for a in sys.argv[3:]] or [30.0]
    s = Session(cfg); s.attach_calib(C.work(cfg, "calib.json"))
    s.heading = HeadingModel.load(C.work(cfg, "heading.csv"))
    trk = pd.read_csv(C.work(cfg, "altitude_track.csv"))
    t = s.ft[fr]; h = float(trk.h.values[fr])
    E0, N0 = s.drone_EN(t, smooth=True)
    Rimu = s.R_wc(t)
    ax = Rimu @ np.array([0, 0, 1.0])
    hd = math.degrees(math.atan2(ax[0], ax[1])); dep = math.degrees(math.asin(-ax[2]))
    im = s.video().read(fr)
    lat_a, lon_a = s.en_to_latlon(E0 - 160, N0 - 160); lat_b, lon_b = s.en_to_latlon(E0 + 160, N0 + 160)
    mo = TileMosaic.build(lat_a, lat_b, lon_a, lon_b, 19, C.work(cfg, "tiles"), s.lat0, s.lon0)
    H, W = im.shape[:2]
    panels = []
    for name, Rwc in [(f"IMU tilt {dep:.1f} deg", Rimu)] + [(f"fixed {d:g} deg", look(hd, d)) for d in fixed]:
        src, dst = [], []
        for u in np.linspace(0, W - 1, 9):
            for v in np.linspace(0, H - 1, 7):
                b = s.cam.bearings(np.array([[u, v]])) @ Rwc.T
                if b[0, 2] >= -1e-3:
                    continue
                lam = h / -b[0, 2]
                x, y = mo.en_to_px(E0 + b[0, 0] * lam, N0 + b[0, 1] * lam)
                src.append((u, v)); dst.append((x, y))
        Hm, _ = cv2.findHomography(np.float32(src), np.float32(dst))
        warp = cv2.warpPerspective(im, Hm, (mo.img.shape[1], mo.img.shape[0]))
        mask = cv2.warpPerspective(np.full(im.shape[:2], 255, np.uint8), Hm, (mo.img.shape[1], mo.img.shape[0])) > 0
        out = mo.img.copy()
        out[mask] = (0.6 * warp[mask] + 0.4 * mo.img[mask]).astype(np.uint8)
        x0, y0 = mo.en_to_px(E0, N0)
        cv2.circle(out, (int(x0), int(y0)), 7, (255, 255, 255), -1)
        cv2.putText(out, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        panels.append(out)
    path = C.work(cfg, f"overlay_{fr}.jpg")
    cv2.imwrite(path, cv2.resize(np.hstack(panels), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(path, f"(drone at E {E0:.1f} N {N0:.1f}, height {h:.1f} m, camera heading {hd:.1f}, IMU depression {dep:.1f})")


if __name__ == "__main__":
    main()
