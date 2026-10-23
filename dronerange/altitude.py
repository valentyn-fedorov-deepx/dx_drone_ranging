"""Height above the ground for every frame, from the images (the GNSS height of this unit is not usable).

1) pairs: GNSS horizontal displacement + ground homography (compass-independent).
   For frames i, j: ground-plane homography H between them; with the camera rotation c1->c2 from the gyro (R21) and
   the ground normal in camera-1 coordinates from the IMU attitude (n, pointing down), the plane-induced model is
       R21^T K^-1 H K / s = I + u n^T,  u = -C2 / d
   (C2 = camera-2 centre in camera-1 coordinates, d = height of camera 1 above the plane), u = E n with E = M - I.
   The GNSS gives |dC_h| (receiver velocity integrated) -> d = |dC_h| / |u_h|. Per-pair checks: dominant right
   singular vector of E vs the IMU normal (the plane really is the ground), rank-1 residual, inlier count.
   Needs a few metres of horizontal travel; hover pairs (about 1 m) are garbage, do not use them.
   Do not filter on the motion-direction agreement: near the ground the IMU heading is off by tens of degrees,
   the height is unaffected.
2) vertical odometry (hover): relative height change dz/h from the vertical part of u between nearby frames,
   integrated in ln h from an anchor height.
3) track: a per-session piecewise plan (config) that stitches constants, odometry runs and smoothed pair heights;
   or the drone's own height-above-ground telemetry when it is available (preferred for live use).
4) cars check: parked-car widths on the ground give an independent height scale (h = W_car / ground width at h = 1).
"""
import math

import cv2
import numpy as np
import pandas as pd

UP = np.array([0, 0, 1.0])
DOWN = np.array([0, 0, -1.0])


def ratio_matches(bf, d1, d2, ratio):
    m = bf.knnMatch(d1, d2, k=2)
    return [a for a, b in (x for x in m if len(x) == 2) if a.distance < ratio * b.distance]


def plane_motion(Hm, K, Kinv, R21, n):
    """ground homography + gyro rotation + ground normal -> (u = E n, singular values, right singular vectors of E)"""
    M = R21.T @ Kinv @ Hm @ K
    M = M / np.linalg.svd(M, compute_uv=False)[1]
    if np.linalg.det(M) < 0:
        M = -M
    Em = M - np.eye(3)
    U, S, Vt = np.linalg.svd(Em)
    return Em @ n, S, Vt


def pairs(sess, P, log=print):
    cam, Rbc, toff, ft, ti = sess.cam, sess.R_bc, sess.toff, sess.ft, sess.ti
    K = cam.K; Kinv = np.linalg.inv(K)
    E_, N_ = sess.E_vel, sess.N_vel
    vid = sess.video()
    sift = cv2.SIFT_create(nfeatures=P["sift_features"])
    bf = cv2.BFMatcher(cv2.NORM_L2)
    feat = {}

    def features(i):
        if i not in feat:
            kp, de = sift.detectAndCompute(vid.gray(i), None)
            feat[i] = (np.float32([k.pt for k in kp]), de)
            if len(feat) > 120:
                feat.pop(next(iter(feat)))
        return feat[i]

    STEP, MIN_DH, MAX_GAP = P["step"], P["min_disp_m"], P["max_gap"]
    F0 = P.get("start_frame", 75)
    F1 = P.get("end_frame") or len(ft) - 1
    rows = []
    for i in range(F0, F1, STEP):
        t1 = ft[i]
        e1, n1 = np.interp(t1 + toff, ti, E_), np.interp(t1 + toff, ti, N_)
        j = None
        for jj in range(i + 1, min(i + MAX_GAP, len(ft))):
            if ft[jj] - ft[jj - 1] > 0.3:
                break
            dh = np.hypot(np.interp(ft[jj] + toff, ti, E_) - e1, np.interp(ft[jj] + toff, ti, N_) - n1)
            if dh >= MIN_DH:
                j = jj
                break
        if j is None:
            continue
        t2 = ft[j]
        dC_h = np.hypot(np.interp(t2 + toff, ti, E_) - e1, np.interp(t2 + toff, ti, N_) - n1)
        p1, d1 = features(i); p2, d2 = features(j)
        if d1 is None or d2 is None or len(p1) < 100 or len(p2) < 100:
            continue
        good = ratio_matches(bf, d1, d2, P["ratio"])
        if len(good) < 60:
            continue
        q1 = cam.undistort(p1[[g.queryIdx for g in good]].astype(float)); q2 = cam.undistort(p2[[g.trainIdx for g in good]].astype(float))
        Hm, inl = cv2.findHomography(q1, q2, cv2.RANSAC, P["ransac_px"], maxIters=4000)
        if Hm is None or inl.sum() < 40:
            continue
        R21 = Rbc.T @ sess.dR_gyro(t1, t2).T @ Rbc                 # bearings camera-1 -> camera-2 for far points
        Rwc1 = sess.R_wb(t1) @ Rbc
        n = Rwc1.T @ DOWN                                            # ground normal (down) in camera-1 coordinates
        u, S, Vt = plane_motion(Hm, K, Kinv, R21, n)
        ang_n = np.degrees(np.arccos(min(1.0, abs(Vt[0] @ n))))
        rank1 = S[1] / max(S[0], 1e-12)
        u_w = Rwc1 @ u                                               # in the world: u = -C2/d
        uh = np.hypot(u_w[0], u_w[1])
        h = dC_h / uh if uh > 1e-6 else np.nan
        gdir = np.degrees(np.arctan2(np.interp(t2 + toff, ti, E_) - e1, np.interp(t2 + toff, ti, N_) - n1))
        idir = np.degrees(np.arctan2(-u_w[0], -u_w[1]))              # the image says the camera moved along -u_w
        rows.append(dict(i=i, j=j, t=t1 - ft[0], dt=t2 - t1, dC_h=dC_h, h=h, dz_rel=-u_w[2], ang_n=ang_n, rank1=rank1, inl=int(inl.sum()),
                         dir_gnss=gdir, dir_img=idir, depr_axis=np.degrees(np.arcsin(-(Rwc1 @ UP)[2]))))
        if len(rows) % 50 == 0:
            log(len(rows), "pairs, frame", i)
    Pd = pd.DataFrame(rows)
    if len(Pd):
        Pd["dir_err"] = ((Pd.dir_img - Pd.dir_gnss + 180) % 360) - 180
    return Pd


def good_pairs(Pd, G):
    lo, hi = G["h_range"]
    m = (Pd.ang_n < G["max_normal_deg"]) & (Pd.rank1 < G["max_rank1"]) & (Pd.inl > G["min_inliers"]) & Pd.h.between(lo, hi)
    if G.get("min_disp_m"):
        m &= Pd.dC_h >= G["min_disp_m"]
    return Pd[m].sort_values("t")


def smoothed_pairs(Pd, G, window):
    g = good_pairs(Pd, G)
    return pd.Series(g.h.values, index=g.t.values).rolling(window, center=True, min_periods=3).median()


def vertical_odometry(sess, f0, f1, h0, h1, step=2, log=print):
    cam, Rbc, ft = sess.cam, sess.R_bc, sess.ft
    K = cam.K; Kinv = np.linalg.inv(K)
    vid = sess.video()
    sift = cv2.SIFT_create(nfeatures=4000); bf = cv2.BFMatcher(cv2.NORM_L2)
    prev, rows = None, []
    for i in range(f0, f1 + 1):
        im = vid.read(i)
        if im is None:
            break
        if (i - f0) % step:
            continue
        kp, de = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
        cur = (i, np.float32([k.pt for k in kp]), de)
        if prev is not None and de is not None and prev[2] is not None:
            j0, p1, d1 = prev; j1, p2, d2 = cur
            t1, t2 = ft[j0], ft[j1]
            rec = dict(i=j0, j=j1, t=t1 - ft[0], ok=False, dz_rel=np.nan, dxy_rel=np.nan, ang_n=np.nan, inl=np.nan, smax=np.nan)
            if t2 - t1 < 0.25 * step + 0.2:
                good = ratio_matches(bf, d1, d2, 0.75)
                if len(good) >= 60:
                    q1 = cam.undistort(p1[[x.queryIdx for x in good]].astype(float)); q2 = cam.undistort(p2[[x.trainIdx for x in good]].astype(float))
                    Hm, inl = cv2.findHomography(q1, q2, cv2.RANSAC, 1.5, maxIters=4000)
                    if Hm is not None and inl.sum() >= 50:
                        R21 = Rbc.T @ sess.dR_gyro(t1, t2).T @ Rbc
                        Rwc1 = sess.R_wb(t1) @ Rbc
                        n = Rwc1.T @ DOWN
                        u, S, Vt = plane_motion(Hm, K, Kinv, R21, n)
                        ang_n = np.degrees(np.arccos(min(1.0, abs(Vt[0] @ n)))) if S[0] > 1e-5 else 0.0
                        u_w = Rwc1 @ u
                        rec.update(ok=True, dz_rel=-u_w[2], dxy_rel=np.hypot(u_w[0], u_w[1]), ang_n=ang_n, inl=int(inl.sum()), smax=S[0])
            rows.append(rec)
        prev = cur
    V = pd.DataFrame(rows)
    V["use"] = V.ok & (V.inl >= 60) & ((V.ang_n < 8) | (V.smax < 0.003))
    dz = np.where(V.use, V.dz_rel, 0.0)
    lnh = np.log(h0) + np.r_[0, np.cumsum(np.log1p(dz[:-1]))]
    lnh_end = np.log(h0) + np.sum(np.log1p(dz))
    closure = lnh_end - np.log(h1)
    log(f"VO {f0}-{f1}: pairs {len(V)}, used {int(V.use.sum())}; open-loop end height {np.exp(lnh_end):.1f} m vs anchor {h1:.1f} m "
        f"(closure {100 * (np.exp(closure) - 1):+.1f} %)")
    w = (V.t - V.t.iloc[0]) / max(V.t.iloc[-1] - V.t.iloc[0], 1e-6)  # closed variant: log-error spread linearly in time
    V["h_open"] = np.exp(lnh); V["h"] = np.exp(lnh - closure * w.values)
    return V


def build_track(sess, plan, gs, vo, telemetry=None):
    """Piecewise height plan -> DataFrame(frame, t, h, src). Segment kinds:
    const (value), vo (open-loop odometry run), pairs (smoothed GNSS+homography heights), ramp (linear between the
    neighbouring segments' boundary values), vo_scaled (odometry run scaled to continue the previous segment)."""
    tt = sess.tt
    if telemetry is not None:
        T = pd.read_csv(telemetry["csv"])
        h = np.interp(sess.ft, T[telemetry.get("t_col", "t_unix")].values, T[telemetry.get("h_col", "h_agl")].values)
        return pd.DataFrame(dict(frame=np.arange(len(tt)), t=tt, h=h, src="telemetry"))
    bounds, t0 = [], -np.inf
    for s in plan:
        t1 = np.inf if s.get("until") is None else float(s["until"])
        bounds.append((t0, t1)); t0 = t1

    def value(k, t):
        s = plan[k]
        kind = s["kind"]
        t = np.atleast_1d(np.asarray(t, float))
        if kind == "const":
            return np.full_like(t, float(s["value"]))
        if kind == "vo":
            V = vo[s["run"]]
            return np.interp(t, V.t, V.h if s.get("closed") else V.h_open)
        if kind == "pairs":
            return np.interp(t, gs.index.values, gs.values)
        if kind == "vo_scaled":
            V = vo[s["run"]]; ts = bounds[k][0]
            return np.interp(t, V.t, V.h_open) * (value(k - 1, ts)[0] / np.interp(ts, V.t, V.h_open))
        if kind == "ramp":
            a, b = bounds[k]
            ha, hb = value(k - 1, a)[0], value(k + 1, b)[0]
            return ha + (t - a) / (b - a) * (hb - ha)
        raise ValueError(kind)

    h = np.full(len(tt), np.nan); src = np.array([""] * len(tt), dtype=object)
    for k, s in enumerate(plan):
        a, b = bounds[k]
        m = (tt >= a) & (tt < b)
        if m.any():
            v = value(k, tt[m])
            if s.get("smooth_s"):                                    # rolling median inside the segment (odometry noise)
                n = max(1, int(round(s["smooth_s"] / np.median(np.diff(tt)))))
                v = pd.Series(v).rolling(n, center=True, min_periods=1).median().values
            h[m] = v; src[m] = s.get("src", s["kind"])
    return pd.DataFrame(dict(frame=np.arange(len(tt)), t=tt, h=h, src=src))


def cars_check(sess, P, weights, track=None, log=print):
    """Independent height scale from parked cars: rays through a car box's bottom corners hit the ground at h = 1 m;
    the lower tail (10th percentile) of the corner-to-corner widths over many cars is the car width."""
    from ultralytics import YOLO
    cam, W, H = sess.cam, sess.cam.width, sess.cam.height
    m = YOLO(weights)
    vid = sess.video()
    rows = []
    for i in range(P["start"], P["end"], P["step"]):
        im = vid.read(i)
        r = m.predict(im, imgsz=P["imgsz"], conf=P["conf"], classes=[2, 7], verbose=False, half=True)[0]
        if r.boxes is None or len(r.boxes) < 3:
            continue
        B = r.boxes.xyxy.cpu().numpy()
        B = B[(B[:, 0] > 5) & (B[:, 2] < W - 5) & (B[:, 3] < H - 5)]   # not clipped by the image border
        if len(B) < 3:
            continue
        Rwc = sess.frame_R_wc(i)
        bl = cam.bearings(np.c_[B[:, 0], B[:, 3]]) @ Rwc.T
        br = cam.bearings(np.c_[B[:, 2], B[:, 3]]) @ Rwc.T
        ok = (bl[:, 2] < -1e-3) & (br[:, 2] < -1e-3)
        L = bl[:, :2] / np.maximum(-bl[:, 2:3], 1e-6); Rr = br[:, :2] / np.maximum(-br[:, 2:3], 1e-6)
        w = np.linalg.norm(L - Rr, axis=1)[ok]                        # ground width at h = 1 m
        if len(w) < 3:
            continue
        rows.append(dict(i=i, t=sess.tt[i], n=len(w), w10=np.percentile(w, 10), w25=np.percentile(w, 25), wmed=np.median(w),
                         h_est=P["width_m"] / np.percentile(w, 10)))
    D = pd.DataFrame(rows)
    if track is not None and len(D):
        D["h_track"] = track.h.values[D.i.values]
        D["ratio"] = D.h_est / D.h_track
        log("car-width height / track height: median %.3f (IQR %.3f-%.3f, n=%d)" % (D.ratio.median(), D.ratio.quantile(.25), D.ratio.quantile(.75), len(D)))
    return D


def plot(path, sess, Pd, G, track, vo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    if len(Pd):
        g = good_pairs(Pd, G)
        ax[0].plot(Pd.t, Pd.h, ".", color="0.75", ms=3, label="all GNSS+homography pairs")
        ax[0].plot(g.t, g.h, ".", ms=4, label="good pairs")
    for k, V in vo.items():
        ax[0].plot(V.t, V.h_open, "-", lw=0.8, alpha=0.7, label=f"vertical odometry ({k}, open loop)")
    ax[0].plot(track.t, track.h, "r-", lw=1.6, label="height used")
    ax[0].set_ylim(0, 80); ax[0].set_ylabel("height above ground, m"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    if len(Pd):
        ax[1].plot(Pd.t, Pd.ang_n, ".", ms=3)
    ax[1].set_ylim(0, 20); ax[1].set_ylabel("plane normal vs IMU, deg"); ax[1].grid(alpha=.3); ax[1].set_xlabel("s from the first frame")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)
