"""Which camera tilt makes the ranging consistent? The drone held its place over the drain and climbed 13 -> 30 m
(barometer), so every parked car must keep its ground distance from the drone at every height. For each tilt model
the ground distance to each car is computed at each height hold; the spread across heights shows the tilt error
(a wrong tilt makes the distance change with the height: D = h / tan(depression)).

Models: the IMU attitude (fused roll / pitch, the pipeline's), and fixed depressions of the optical axis with a level
horizon (what a system without the IMU would assume).

Usage: python tools/tilt_test.py configs/drone2244_20261023.yaml [depression_deg ...]
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronerange import config as C  # noqa: E402
from dronerange.camera import intersect_ground  # noqa: E402
from dronerange.heading import HeadingModel  # noqa: E402
from dronerange.session import Session  # noqa: E402


def look(heading_deg, depression_deg):
    h, d = math.radians(heading_deg), math.radians(depression_deg)
    z = np.array([math.sin(h) * math.cos(d), math.cos(h) * math.cos(d), -math.sin(d)])
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


def main():
    cfg = C.load(sys.argv[1])
    fixed = [float(a) for a in sys.argv[2:]] or [30.0, 24.0]
    s = Session(cfg); s.attach_calib(C.work(cfg, "calib.json"))
    s.heading = HeadingModel.load(C.work(cfg, "heading.csv"))
    T = pd.read_csv(C.work(cfg, "tracks.csv")); R = pd.read_csv(C.work(cfg, "ranges.csv"))
    H = pd.read_csv(Path(cfg["data_root"]) / "anchors/heights.csv")
    trk = pd.read_csv(C.work(cfg, "altitude_track.csv"))
    S = T[(T.status == "seen") & (T.grp == "vehicle") & (T.det_row >= 0)]
    D = R.loc[S.det_row.astype(int)].reset_index(drop=True)
    D["wid"] = S.id.values
    models = ["IMU"] + [f"fixed {d:g} deg" for d in fixed]
    rows = []
    for r in H.itertuples():
        a, b = int(r.settled_frame), int(r.held_until_frame)
        W = D[D.frame.between(a, b)]
        for i, g in W.groupby("frame"):
            t = s.ft[int(i)]; h = float(trk.h.values[int(i)])
            Rimu = s.R_wc(t)
            ax = Rimu @ np.array([0, 0, 1.0])
            hd = math.degrees(math.atan2(ax[0], ax[1]))
            px = np.c_[0.5 * (g.x1 + g.x2), g.y2]
            for name, Rwc in zip(models, [Rimu] + [look(hd, d) for d in fixed]):
                dE, dN, ok = intersect_ground(s.cam.bearings(px) @ Rwc.T, h)
                for wid, dist in zip(g.wid.values, np.hypot(dE, dN)):
                    rows.append(dict(model=name, lev=r.height_m, wid=wid, D=dist))
    X = pd.DataFrame(rows)
    X = X.groupby(["model", "wid", "lev"]).D.median().reset_index()
    X = X[X.groupby(["model", "wid"]).lev.transform("nunique") >= 2]
    X["ratio"] = X.D / X.groupby(["model", "wid"]).D.transform("median")
    print(f"{X.wid.nunique()} parked vehicles seen at two or more of the heights 13.1 / 16.2 / 20.0 / 25.4 / 30.2 m")
    print("model            median |distance change across heights|   distance at 13 m vs 30 m (per car, median)   median distance")
    for m in models:
        x = X[X.model == m]
        lo = x[x.lev == 13.5].set_index("wid").D; hi = x[x.lev == 30.0].set_index("wid").D
        both = lo.index.intersection(hi.index)
        r = (lo[both] / hi[both]).median() if len(both) else float("nan")
        print(f"{m:15s}  {100 * (x.ratio - 1).abs().median():6.1f} %                               "
              f"{(r - 1) * 100:+6.1f} %                                {x.D.median():6.1f} m")
    print("\nper height (median distance ratio to the car's own median):")
    print(X.pivot_table(index="lev", columns="model", values="ratio", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
