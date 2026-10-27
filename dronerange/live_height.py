"""Height above the ground from what the unit itself has, frame by frame and causally (a live run can do the same).

- Vertical odometry: the change of height relative to a key frame, from the ground homography between the key frame
  and the current frame, the gyro rotation between them and the IMU tilt (the ground normal): ln h = ln h_key +
  ln(1 + dz/h_key). Measuring against a key frame (renewed every few seconds or when the overlap gets thin) instead of
  the previous frame keeps the small bias of every pair from adding up frame after frame.
- Absolute fixes: whenever the unit's GNSS says it flew a few metres horizontally at 3 m/s or more AND the images
  show that motion too, the height from that pair: h = |GNSS displacement| / |horizontal image motion per metre of
  height| (the pair model of altitude.pairs). Only fast travel gives a usable fix: on 10-20 the fixes agree with the
  height plan within 2 % at >= 3 m/s, +15 % at 1.5-3 m/s and are useless below (the receiver's drift, metres per
  minute, is then as large as the travel). The windows of successive fixes do not overlap, so their GNSS errors are
  independent. A flight without fast horizontal travel gets no absolute height from this.
- A Kalman filter on ln h joins the two: it starts from the median of the first fixes, takes later fixes within 3
  sigma and never more than 30 % away; three fixes in a row that agree with each other but not with the filter
  restart it.
Nothing is set per session: the thresholds are physical (metres of baseline, image motion relative to the height,
inliers, the plane check). Not used: the unit's GNSS height (it drifts by metres), the aircraft's log, the heading
(neither the odometry nor the fixes need it).
"""
import math
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd

from .altitude import DOWN, plane_motion, ratio_matches

DEFAULTS = dict(
    features=2500, ratio=0.75, ransac_px=1.5, min_inliers=60, max_dz=0.25,
    key_max_s=3.0, key_min_inliers=150, key_max_dz=0.08,          # renew the key frame after 3 s, thin overlap or 8 % of height change
    fix_min_disp_m=5.0, fix_min_speed=3.0, fix_max_gap_s=4.0, fix_every=3,   # a fix: >= 5 m of GNSS travel at >= 3 m/s, tried every 3rd frame
    fix_min_image_motion=0.12,                                      # ... and the images moved >= 12 % of the height sideways
    fix_max_normal_deg=4.0, fix_max_rank1=0.15, h_range=(1.5, 300.0),
    vo_sigma=0.006, vo_fail_sigma=0.04, pause_sigma=0.3,           # ln-height noise per frame (odometry ok / failed / recording pause)
    fix_sigma_m=0.5, fix_sigma_rel=0.04,                           # fix noise: GNSS displacement error (m) and model error (relative)
    gate=3.0, max_jump=math.log(1.3), init_fixes=3, restart_fixes=3,
)


class _Frames:
    """features of the recent frames"""

    def __init__(self, sess, n_features, keep=90):
        self.vid = sess.video()
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.cache = OrderedDict()
        self.keep = keep

    def get(self, i):
        if i not in self.cache:
            g = self.vid.gray(i)
            kp, de = self.sift.detectAndCompute(g, None) if g is not None else ([], None)
            self.cache[i] = (np.float32([k.pt for k in kp]), de)
            while len(self.cache) > self.keep:
                self.cache.popitem(last=False)
        return self.cache[i]


def run(sess, params=None, frames=None, log=print):
    """-> (per-frame DataFrame: frame, t, h, sigma (of ln h), dlnh, vo_ok, key, fixes so far; DataFrame of the fixes)"""
    P = {**DEFAULTS, **(params or {})}
    cam, Rbc, ft, ti, toff = sess.cam, sess.R_bc, sess.ft, sess.ti, sess.toff
    K = cam.K; Kinv = np.linalg.inv(K)
    E_, N_ = sess.E_vel, sess.N_vel                                 # the unit's receiver velocity, integrated
    gE = lambda t: float(np.interp(t + toff, ti, E_)); gN = lambda t: float(np.interp(t + toff, ti, N_))
    F = _Frames(sess, P["features"])
    bf = cv2.BFMatcher(cv2.NORM_L2)
    frames = list(range(sess.n_frames)) if frames is None else list(frames)

    def plane(i, j):
        p1, d1 = F.get(i); p2, d2 = F.get(j)
        if d1 is None or d2 is None or len(p1) < 100 or len(p2) < 100:
            return None
        good = ratio_matches(bf, d1, d2, P["ratio"])
        if len(good) < P["min_inliers"]:
            return None
        q1 = cam.undistort(p1[[g.queryIdx for g in good]].astype(float)); q2 = cam.undistort(p2[[g.trainIdx for g in good]].astype(float))
        Hm, inl = cv2.findHomography(q1, q2, cv2.RANSAC, P["ransac_px"], maxIters=4000)
        if Hm is None or inl.sum() < P["min_inliers"]:
            return None
        t1, t2 = ft[i], ft[j]
        R21 = Rbc.T @ sess.dR_gyro(t1, t2).T @ Rbc
        Rwc1 = sess.R_wb(t1) @ Rbc                                  # only its tilt matters here
        n = Rwc1.T @ DOWN
        u, S, Vt = plane_motion(Hm, K, Kinv, R21, n)
        ang = np.degrees(np.arccos(min(1.0, abs(Vt[0] @ n)))) if S[0] > 1e-5 else 0.0
        return Rwc1 @ u, ang, S[1] / max(S[0], 1e-12), int(inl.sum())

    x = Pv = None                                                   # filter: ln h and its variance
    L = {}                                                          # odometry: ln h relative to the start
    init, outliers = [], []                                         # fixes before the start / gated in a row
    rows, fixes = [], []
    prev = key = None
    last_fix = -1                                                   # frame of the last fix: windows do not overlap
    for n_done, k in enumerate(frames):
        vo_ok, var = False, P["vo_fail_sigma"] ** 2
        if prev is None:
            L[k] = 0.0; key = k
        elif ft[k] - ft[prev] > 0.5:                                 # recording pause: the height may have changed
            L[k] = L[prev]; key = k; var = P["pause_sigma"] ** 2
        else:
            r = plane(key, k) if key != prev else None
            if r is not None and r[3] >= P["min_inliers"] and abs(-r[0][2]) < P["max_dz"]:
                L[k] = L[key] + math.log1p(-r[0][2]); vo_ok = True
            else:                                                    # from the previous frame, which becomes the key frame
                r = plane(prev, k)
                if r is not None and abs(-r[0][2]) < P["max_dz"]:
                    L[k] = L[prev] + math.log1p(-r[0][2]); vo_ok = True
                else:
                    L[k] = L[prev]
                key = prev
            if vo_ok:
                var = P["vo_sigma"] ** 2
                if ft[k] - ft[key] > P["key_max_s"] or r[3] < P["key_min_inliers"] or abs(L[k] - L[key]) > P["key_max_dz"]:
                    key = k
        d = L[k] - (L[prev] if prev is not None else 0.0)
        if x is not None:
            x += d; Pv += var
        if k % P["fix_every"] == 0:                                  # an absolute fix from a few metres of travel
            e1, n1, i = gE(ft[k]), gN(ft[k]), None
            for j in range(k - 1, max(last_fix, 0) - 1, -1):
                if j not in L or ft[k] - ft[j] > P["fix_max_gap_s"] or ft[j + 1] - ft[j] > 0.5:
                    break
                if math.hypot(e1 - gE(ft[j]), n1 - gN(ft[j])) >= P["fix_min_disp_m"]:
                    i = j
                    break
            if i is not None:
                r = plane(i, k)
                if r is not None:
                    u_w, ang, r1, inl = r
                    dC = math.hypot(e1 - gE(ft[i]), n1 - gN(ft[i]))
                    uh = math.hypot(u_w[0], u_w[1])
                    h_i = dC / uh if uh > 1e-6 else float("nan")
                    lo, hi = P["h_range"]
                    status = "bad pair"
                    if dC / max(ft[k] - ft[i], 1e-6) < P["fix_min_speed"]:
                        status = "slow"                              # slow travel: the receiver's drift is as large as the travel
                    elif uh < P["fix_min_image_motion"]:
                        status = "no image motion"                   # the receiver drifted, the drone did not move
                    elif ang < P["fix_max_normal_deg"] and r1 < P["fix_max_rank1"] and lo <= h_i <= hi:
                        last_fix = k
                        z = math.log(h_i) + (L[k] - L[i])            # the fix refers to frame i: carry it to frame k
                        rv = (P["fix_sigma_m"] / dC) ** 2 + P["fix_sigma_rel"] ** 2
                        if x is None:
                            init.append((k, z, rv)); status = "start"
                            if len(init) >= P["init_fixes"]:
                                zs = [zz + (L[k] - L[kk]) for kk, zz, _ in init]
                                x = float(np.median(zs)); Pv = float(np.median([v for _, _, v in init])) / len(init)
                        elif abs(z - x) <= min(P["gate"] * math.sqrt(Pv + rv), P["max_jump"]):
                            G = Pv / (Pv + rv); x += G * (z - x); Pv *= 1 - G; status = "used"; outliers = []
                        else:
                            status = "gated"; outliers.append((k, z, rv))
                            if len(outliers) >= P["restart_fixes"]:  # they agree with each other: the filter was wrong
                                zs = np.array([zz + (L[k] - L[kk]) for kk, zz, _ in outliers[-P["restart_fixes"]:]])
                                if zs.max() - zs.min() < math.log(1.15):
                                    x = float(np.median(zs)); Pv = float(np.median([v for _, _, v in outliers])) / len(zs)
                                    status = "restart"; outliers = []
                    fixes.append(dict(frame=k, i=i, t=ft[k] - ft[0], dC=dC, speed=dC / max(ft[k] - ft[i], 1e-6), uh=uh, h_i=h_i,
                                      h_fix=h_i * math.exp(L[k] - L[i]), ang_n=ang, rank1=r1, inl=inl, status=status))
        rows.append(dict(frame=k, t=ft[k] - ft[0], h=math.exp(x) if x is not None else np.nan,
                         sigma=math.sqrt(Pv) if x is not None else np.nan, dlnh=d, vo_ok=vo_ok, key=key,
                         n_fix=sum(1 for f in fixes if f["status"] in ("used", "start", "restart"))))
        prev = k
        if n_done % 300 == 0:
            log(f"live height: frame {k}, h {rows[-1]['h']:.1f} m, fixes used {rows[-1]['n_fix']}")
    return pd.DataFrame(rows), pd.DataFrame(fixes)
