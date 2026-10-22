"""Checks of the ranging without any ground truth survey.

A) pilot triangle: the pilot stands at the take-off point (= the ENU origin, first GNSS fix). While the camera sees
   the pilot, the ranged ground distance to that person must equal the GNSS distance drone -> take-off point.
   The pilot is identified by frame windows (config `validate.pilot_windows`; in each window the person track with
   the most detections). All low-hover person tracks are listed as well, with a crop sheet for a visual check.
B) static-vehicle trilateration: a parked car seen from a moving drone (baseline >= 15 m) must sit at one point; the
   per-frame scatter of the ranged distances about the least-squares point measures the ranging noise. The implied
   heading error (absolute bearing vs the trilaterated direction) measures the IMU yaw error.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def person_tracks(R, max_frame=None, min_n=10):
    p = R[(R.cls == "person") & (R.track >= 0)].copy()
    if max_frame is not None:
        p = p[p.frame <= max_frame]
    p["d_takeoff"] = np.hypot(p.E_drone, p.N_drone)
    p["pos_err"] = np.hypot(p.E, p.N)                  # distance of the estimated position from the take-off point (heading-dependent)
    g = p.groupby("track").agg(n=("frame", "size"), f0=("frame", "min"), f1=("frame", "max"), D=("D", "median"), d_to=("d_takeoff", "median"),
                               pos_err=("pos_err", "median"), h=("h", "median"))
    g["D_vs_dto_pct"] = 100 * (g.D / g.d_to - 1)
    return g[g.n >= min_n].sort_values("f0"), p


def pilot_check(R, windows, max_frame=None):
    tracks, p = person_tracks(R, max_frame)
    rows = []
    for f0, f1 in windows:
        w = p[p.frame.between(f0, f1)]
        if not len(w):
            rows.append(dict(window=f"{f0}-{f1}", track=np.nan, n=0))
            continue
        tid = w.track.value_counts().index[0]
        s = w[w.track == tid]
        rows.append(dict(window=f"{f0}-{f1}", track=int(tid), n=len(s), h=s.h.median(), D=s.D.median(), d_to=s.d_takeoff.median(),
                         err_pct=100 * (s.D.median() / s.d_takeoff.median() - 1)))
    return pd.DataFrame(rows), tracks


def trilateration(R, min_n=15, min_span=15.0):
    V = R[R.cls.isin(["car", "truck"]) & (R.track >= 0)]
    res = []
    for tid, g in V.groupby("track"):
        if len(g) < min_n:
            continue
        span = np.hypot(g.E_drone.max() - g.E_drone.min(), g.N_drone.max() - g.N_drone.min())
        if span < min_span:                            # the drone must move: the baseline of the trilateration
            continue
        x0 = np.r_[g.E.median(), g.N.median()]
        f = lambda q: np.hypot(q[0] - g.E_drone.values, q[1] - g.N_drone.values) - g.D.values
        s = least_squares(f, x0, loss="soft_l1", f_scale=1.0)
        r = f(s.x)
        rel = r / g.D.values
        ang_tri = np.degrees(np.arctan2(s.x[0] - g.E_drone.values, s.x[1] - g.N_drone.values))
        yaw_err = ((g.brg.values - ang_tri + 180) % 360) - 180
        res.append(dict(track=tid, n=len(g), t0=g.t.min(), t1=g.t.max(), span=span, Dmed=g.D.median(), rms_m=np.sqrt(np.mean(r ** 2)),
                        mad_rel=100 * 1.4826 * np.median(np.abs(rel - np.median(rel))), bias_rel=100 * np.median(rel), yaw_err=np.median(yaw_err)))
    return pd.DataFrame(res)


def crop_sheet(path, sess, R, tracks, size=160):
    """one crop per person track (middle detection), labelled with the track id, D and the drone -> take-off distance"""
    import cv2
    if not len(tracks):
        return
    vid = sess.video()
    tiles = []
    for tid, row in tracks.iterrows():
        g = R[(R.track == tid) & (R.cls == "person")].sort_values("frame")
        o = g.iloc[len(g) // 2]
        im = vid.read(int(o.frame))
        if im is None:
            continue
        cx, cy = 0.5 * (o.x1 + o.x2), 0.5 * (o.y1 + o.y2)
        half = max(o.y2 - o.y1, o.x2 - o.x1, 40) * 1.5
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, im.shape[1])), int(min(cy + half, im.shape[0]))
        c = cv2.resize(im[y0:y1, x0:x1], (size, size))
        cv2.rectangle(c, (0, size - 34), (size, size), (0, 0, 0), -1)
        cv2.putText(c, f"#{tid} f{int(o.frame)}", (4, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(c, f"D {row.D:.1f} / {row.d_to:.1f} m", (4, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 230, 60), 1)
        tiles.append(c)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))
