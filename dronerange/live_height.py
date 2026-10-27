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

The same fast pairs anchor the heading, causally: the direction the images say the camera moved against the GNSS
course gives the error of the yaw used in the pair (the unit's magnetic yaw, off by tens of degrees near the ground);
the live heading is the gyro yaw plus the median correction of the anchors of the last two minutes (3-MAD outliers
out), held between anchors (heading_track).
"""
import math
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd

from .altitude import DOWN, plane_motion, ratio_matches
from .heading import gyro_yaw, wrap

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
    """features of the recent frames (detected on a scaled copy, returned in full-resolution pixels)"""

    def __init__(self, sess, n_features, scale=1.0, keep=90):
        self.sess, self.vid = sess, None
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.cache = OrderedDict()
        self.keep, self.scale = keep, scale

    def put(self, i, gray):
        if i in self.cache:
            return
        g = gray if self.scale == 1.0 else cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        kp, de = self.sift.detectAndCompute(g, None) if g is not None else ([], None)
        self.cache[i] = (np.float32([k.pt for k in kp]) / self.scale, de)
        while len(self.cache) > self.keep:
            self.cache.popitem(last=False)

    def get(self, i):
        if i not in self.cache:
            if self.vid is None:
                self.vid = self.sess.video()
            self.put(i, self.vid.gray(i))
        return self.cache[i]


class LiveHeight:
    """the estimator frame by frame: step(k[, gray image of frame k]) -> the state after frame k; frames in time
    order, gaps allowed (a live run drops frames it cannot keep up with)"""

    def __init__(self, sess, params=None):
        P = self.P = {**DEFAULTS, **(params or {})}
        self.sess = sess
        self.K = sess.cam.K; self.Kinv = np.linalg.inv(self.K)
        self.F = _Frames(sess, P["features"], P.get("feature_scale", 1.0))
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.x = self.Pv = None                                     # filter: ln h and its variance
        self.L = {}                                                 # odometry: ln h relative to the start
        self.init, self.outliers = [], []                           # fixes before the start / gated in a row
        self.fixes, self.anchors = [], []
        self.prev = self.key = None
        self.last_fix = -1                                          # frame of the last fix: windows do not overlap
        self.yg = gyro_yaw(sess)                                    # heading anchors: the yaw used in the pairs vs the gyro yaw
        self.c = 0.0                                                # live heading correction (deg) on top of the gyro yaw
        self.processed = []

    def gE(self, t):
        return float(np.interp(t + self.sess.toff, self.sess.ti, self.sess.E_vel))

    def gN(self, t):
        return float(np.interp(t + self.sess.toff, self.sess.ti, self.sess.N_vel))

    @property
    def h(self):
        return math.exp(self.x) if self.x is not None else None

    def yaw(self, t_imu):
        """the live heading at an IMU time: gyro yaw + the current correction"""
        return float(np.interp(t_imu, self.sess.ti, self.yg)) + self.c

    def _plane(self, i, j):
        P, sess, cam = self.P, self.sess, self.sess.cam
        p1, d1 = self.F.get(i); p2, d2 = self.F.get(j)
        if d1 is None or d2 is None or len(p1) < 100 or len(p2) < 100:
            return None
        good = ratio_matches(self.bf, d1, d2, P["ratio"])
        if len(good) < P["min_inliers"]:
            return None
        q1 = cam.undistort(p1[[g.queryIdx for g in good]].astype(float)); q2 = cam.undistort(p2[[g.trainIdx for g in good]].astype(float))
        Hm, inl = cv2.findHomography(q1, q2, cv2.RANSAC, P["ransac_px"], maxIters=4000)
        if Hm is None or inl.sum() < P["min_inliers"]:
            return None
        t1, t2 = sess.ft[i], sess.ft[j]
        R21 = sess.R_bc.T @ sess.dR_gyro(t1, t2).T @ sess.R_bc
        Rwc1 = sess.R_wb(t1) @ sess.R_bc                            # only its tilt matters here
        n = Rwc1.T @ DOWN
        u, S, Vt = plane_motion(Hm, self.K, self.Kinv, R21, n)
        ang = np.degrees(np.arccos(min(1.0, abs(Vt[0] @ n)))) if S[0] > 1e-5 else 0.0
        return Rwc1 @ u, ang, S[1] / max(S[0], 1e-12), int(inl.sum())

    def _update_heading(self, window_s=120.0, min_n=3):
        tv = np.array([a["t_imu"] for a in self.anchors]); cv = np.array([a["c"] for a in self.anchors])
        m = tv >= tv[-1] - window_s
        if m.sum() < min_n:
            return
        w = cv[m]; c0 = float(np.median(wrap(w))); w = c0 + wrap(w - c0)
        med = np.median(w); mad = 1.4826 * np.median(np.abs(w - med))
        self.c = float(np.median(w[np.abs(w - med) <= max(3 * mad, 6.0)]))

    def step(self, k, gray=None):
        P, sess, ft, L = self.P, self.sess, self.sess.ft, self.L
        if gray is not None:
            self.F.put(k, gray)
        prev, key = self.prev, self.key
        vo_ok, var = False, P["vo_fail_sigma"] ** 2
        if prev is None:
            L[k] = 0.0; key = k
        elif ft[k] - ft[prev] > 0.5 + 0.25 * (k - prev - 1):          # recording pause (dropped frames are not a pause)
            L[k] = L[prev]; key = k; var = P["pause_sigma"] ** 2
        else:
            r = self._plane(key, k) if key != prev else None
            if r is not None and r[3] >= P["min_inliers"] and abs(-r[0][2]) < P["max_dz"]:
                L[k] = L[key] + math.log1p(-r[0][2]); vo_ok = True
            else:                                                    # from the previous frame, which becomes the key frame
                r = self._plane(prev, k)
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
        if self.x is not None:
            self.x += d; self.Pv += var
        if k % P["fix_every"] == 0:
            self._fix(k)
        self.prev, self.key = k, key
        self.processed.append(k)
        return dict(frame=k, t=ft[k] - ft[0], h=self.h if self.h is not None else np.nan,
                    sigma=math.sqrt(self.Pv) if self.x is not None else np.nan, dlnh=d, vo_ok=vo_ok, key=key,
                    n_fix=sum(1 for f in self.fixes if f["status"] in ("used", "start", "restart")), heading_c=self.c)

    def _fix(self, k):
        """an absolute fix from a few metres of fast travel ending at frame k"""
        P, sess, ft, L = self.P, self.sess, self.sess.ft, self.L
        e1, n1, i = self.gE(ft[k]), self.gN(ft[k]), None
        for pos in range(len(self.processed) - 1, -1, -1):           # the frames before k that were processed
            j = self.processed[pos]
            if j <= self.last_fix or ft[k] - ft[j] > P["fix_max_gap_s"]:
                break
            nxt = self.processed[pos + 1] if pos + 1 < len(self.processed) else k
            if ft[nxt] - ft[j] > 0.5 + 0.25 * (nxt - j - 1):
                break
            if math.hypot(e1 - self.gE(ft[j]), n1 - self.gN(ft[j])) >= P["fix_min_disp_m"]:
                i = j
                break
        if i is None:
            return
        r = self._plane(i, k)
        if r is None:
            return
        u_w, ang, r1, inl = r
        dC = math.hypot(e1 - self.gE(ft[i]), n1 - self.gN(ft[i]))
        uh = math.hypot(u_w[0], u_w[1])
        h_i = dC / uh if uh > 1e-6 else float("nan")
        lo, hi = P["h_range"]
        status = "bad pair"
        if dC / max(ft[k] - ft[i], 1e-6) < P["fix_min_speed"]:
            status = "slow"                                          # slow travel: the receiver's drift is as large as the travel
        elif uh < P["fix_min_image_motion"]:
            status = "no image motion"                               # the receiver drifted, the drone did not move
        elif ang < P["fix_max_normal_deg"] and r1 < P["fix_max_rank1"] and lo <= h_i <= hi:
            self.last_fix = k
            t_imu = ft[i] + sess.toff                                # heading anchor: image motion direction vs GNSS course
            e = float(wrap(math.degrees(math.atan2(-u_w[0], -u_w[1])) - math.degrees(math.atan2(e1 - self.gE(ft[i]), n1 - self.gN(ft[i])))))
            y_used = sess.heading.yaw(t_imu) if sess.heading is not None else float(np.interp(t_imu, sess.ti, sess.eul[:, 2]))
            self.anchors.append(dict(frame=k, t_imu=t_imu, e=e, c=y_used + e - float(np.interp(t_imu, sess.ti, self.yg))))
            self._update_heading()
            z = math.log(h_i) + (L[k] - L[i])                        # the fix refers to frame i: carry it to frame k
            rv = (P["fix_sigma_m"] / dC) ** 2 + P["fix_sigma_rel"] ** 2
            if self.x is None:
                self.init.append((k, z, rv)); status = "start"
                if len(self.init) >= P["init_fixes"]:
                    zs = [zz + (L[k] - L[kk]) for kk, zz, _ in self.init]
                    self.x = float(np.median(zs)); self.Pv = float(np.median([v for _, _, v in self.init])) / len(self.init)
            elif abs(z - self.x) <= min(P["gate"] * math.sqrt(self.Pv + rv), P["max_jump"]):
                G = self.Pv / (self.Pv + rv); self.x += G * (z - self.x); self.Pv *= 1 - G; status = "used"; self.outliers = []
            else:
                status = "gated"; self.outliers.append((k, z, rv))
                if len(self.outliers) >= P["restart_fixes"]:          # they agree with each other: the filter was wrong
                    zs = np.array([zz + (L[k] - L[kk]) for kk, zz, _ in self.outliers[-P["restart_fixes"]:]])
                    if zs.max() - zs.min() < math.log(1.15):
                        self.x = float(np.median(zs)); self.Pv = float(np.median([v for _, _, v in self.outliers])) / len(zs)
                        status = "restart"; self.outliers = []
        self.fixes.append(dict(frame=k, i=i, t=ft[k] - ft[0], dC=dC, speed=dC / max(ft[k] - ft[i], 1e-6), uh=uh, h_i=h_i,
                               h_fix=h_i * math.exp(L[k] - L[i]), ang_n=ang, rank1=r1, inl=inl, status=status))


def run(sess, params=None, frames=None, log=print):
    """the whole session -> (per-frame DataFrame: frame, t, h, sigma (of ln h), dlnh, vo_ok, key, fixes so far, heading
    correction; DataFrame of the fixes; DataFrame of the heading anchors)"""
    est = LiveHeight(sess, params)
    frames = list(range(sess.n_frames)) if frames is None else list(frames)
    rows = []
    for n_done, k in enumerate(frames):
        rows.append(est.step(k))
        if n_done % 300 == 0:
            log(f"live height: frame {k}, h {rows[-1]['h']:.1f} m, fixes used {rows[-1]['n_fix']}")
    return pd.DataFrame(rows), pd.DataFrame(est.fixes), pd.DataFrame(est.anchors)


def heading_track(sess, anchors, window_s=120.0, min_n=3):
    """causal heading on the IMU grid from the anchors: the gyro yaw plus, after each anchor, the median correction of
    the anchors of the last window_s (3-MAD outliers, at least 6 deg, left out); before min_n anchors the gyro yaw as
    it started (from the first magnetic yaw)"""
    yg = gyro_yaw(sess)
    c = np.zeros(len(yg))
    if len(anchors):
        A = anchors.sort_values("t_imu")
        tv, cv = A.t_imu.values, A.c.values
        c0 = float(np.median(wrap(cv))); cv = c0 + wrap(cv - c0)
        steps = []
        for k in range(len(tv)):
            m = (tv >= tv[k] - window_s) & (tv <= tv[k])
            if m.sum() < min_n:
                continue
            w = cv[m]; med = np.median(w); mad = 1.4826 * np.median(np.abs(w - med))
            w = w[np.abs(w - med) <= max(3 * mad, 6.0)]
            steps.append((tv[k], float(np.median(w))))
        for tk, ck in steps:
            c[sess.ti >= tk] = ck
    return pd.DataFrame(dict(t_imu=sess.ti, yaw_xsens=sess.eul[:, 2], yaw_gyro=yg, c=c, yaw_used=yg + c))
