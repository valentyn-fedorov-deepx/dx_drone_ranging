"""Camera intrinsics (f, k1) and camera-IMU rotation from the flight itself: image rotation between consecutive frames
during hover at height vs the bias-corrected gyro over the same interval (hand-eye rotation calibration).

Model (pure rotation; pairs at height with ground speed below max_speed so translation parallax stays small):
    dR = R_ref_b(t1)^T R_ref_b(t2) from the gyro (body = IMU; continuous Slerp of the integrated orientation),
    b2 = R_bc^T dR^T R_bc b1 for a fixed world direction, b = undistort(K^-1 [u v 1]) with one radial term k1.
Unknowns: f, k1, R_bc (rotation vector), IMU time offset (ms). Start: best of the 24 axis-aligned R_bc, then a grid
over the time offset and f, then robust least squares. The principal point stays at the image centre.
Output: calib.json (same keys as consumed by the other steps).
"""
import json
from itertools import permutations, product

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from .camera import Camera


def axis_aligned_rotations():
    mats = []
    for perm in permutations(range(3)):
        for sg in product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, sg)):
                M[c, r] = s
            if np.linalg.det(M) > 0:
                mats.append(M)
    return mats


def candidate_pairs(sess, P):
    ti, spd, ft = sess.ti, sess.speed, sess.ft
    cand = []
    for i in range(P["start_frame"], len(ft) - 1):
        t1, t2 = ft[i], ft[i + 1]
        if t2 - t1 > P["max_dt"]:
            continue
        i0, i1 = np.searchsorted(ti, t1 - P["margin_s"]), np.searchsorted(ti, t2 + P["margin_s"])
        if np.max(np.diff(ti[i0:i1 + 1])) > P["max_imu_gap_s"] or spd[i0:i1].max() > P["max_speed"]:
            continue
        ang = np.degrees(np.linalg.norm(Rot.from_matrix(sess.dR_gyro(t1, t2, 0.0)).as_rotvec()))
        if P["min_deg"] <= ang <= P["max_deg"]:
            cand.append((i, ang))
    return cand


def run(sess, P, out_path, log=print):
    W, H = P["width"], P["height"]
    cx, cy = W / 2 - 0.5, H / 2 - 0.5
    bg = sess.gyro_bias(P["bias_seconds"])
    sess.set_bias(bg)
    log("gyro bias (deg/s):", np.degrees(bg).round(3))
    ft = sess.ft
    cand = candidate_pairs(sess, P)
    log("candidate pairs", len(cand))
    rng = np.random.default_rng(P["seed"])
    pairs = [cand[k] for k in sorted(rng.choice(len(cand), min(P["n_pairs"], len(cand)), replace=False))]
    sift = cv2.SIFT_create(nfeatures=P["sift_features"])
    bf = cv2.BFMatcher(cv2.NORM_L2)
    vid = sess.video()
    data = []
    for (i, ang) in pairs:
        g1, g2 = vid.gray(i), vid.gray(i + 1)
        k1_, d1 = sift.detectAndCompute(g1, None); k2_, d2 = sift.detectAndCompute(g2, None)
        if d1 is None or d2 is None:
            continue
        m = bf.knnMatch(d1, d2, k=2)
        good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < P["ratio"] * b.distance]
        if len(good) < 40:
            continue
        p1 = np.float32([k1_[g.queryIdx].pt for g in good]); p2 = np.float32([k2_[g.trainIdx].pt for g in good])
        Hm, inl = cv2.findHomography(p1, p2, cv2.RANSAC, P["ransac_px"])
        if Hm is None or inl.sum() < 30:
            continue
        inl = inl.ravel() > 0
        p1, p2 = p1[inl], p2[inl]
        if len(p1) > P["max_points"]:
            s = rng.choice(len(p1), P["max_points"], replace=False); p1, p2 = p1[s], p2[s]
        data.append((i, ft[i], ft[i + 1], p1.astype(float), p2.astype(float), ang))
    log("pairs with matches", len(data), "points", sum(len(x[3]) for x in data))

    def residuals(p, per_pair=False):
        cam = Camera(p[0], cx, cy, p[1])
        Rbc = Rot.from_rotvec(p[2:5]).as_matrix()
        toff = p[5] * 1e-3
        out = []
        for (i, t1, t2, p1, p2, ang) in data:
            Rc = Rbc.T @ sess.dR_gyro(t1, t2, toff).T @ Rbc
            out.append(cam.project(cam.bearings(p1) @ Rc.T) - p2)
        return out if per_pair else np.concatenate([o.ravel() for o in out])

    best = None
    for M in axis_aligned_rotations():
        for f0 in P["f0_axis_search"]:
            p0 = np.r_[f0, 0.0, Rot.from_matrix(M).as_rotvec(), 0.0]
            c = np.median(np.abs(residuals(p0)))
            if best is None or c < best[0]:
                best = (c, p0)
    log("axis-aligned start: median |res| %.1f px, R_bc =\n%s" % (best[0], Rot.from_rotvec(best[1][2:5]).as_matrix().round(2)))
    grid = []
    lo, hi, st = P["toff_grid_ms"]
    for to in np.arange(lo, hi + 1, st):
        for f0 in P["f0_grid"]:
            p0 = best[1].copy(); p0[0] = f0; p0[5] = to
            grid.append((np.median(np.abs(residuals(p0))), to, f0))
    gm = min(grid)
    log("time-offset grid: median |res| %.2f px at %+.0f ms (f0 %.0f)" % gm)
    p0 = best[1].copy(); p0[0] = gm[2]; p0[5] = gm[1]
    sol = least_squares(residuals, p0, loss="soft_l1", f_scale=2.0, x_scale=np.r_[100, 0.05, 0.02, 0.02, 0.02, 5.0],
                        diff_step=np.r_[1e-4, 1e-3, 1e-4, 1e-4, 1e-4, 1e-3],
                        bounds=([1000, -0.6, -np.pi, -np.pi, -np.pi, -300], [12000, 0.6, np.pi, np.pi, np.pi, 300]), max_nfev=300)
    p = sol.x
    pp = residuals(p, per_pair=True)
    e = np.linalg.norm(np.concatenate(pp), axis=1)
    Rbc = Rot.from_rotvec(p[2:5]).as_matrix()
    hfov = 2 * np.degrees(np.arctan(W / 2 / p[0])); vfov = 2 * np.degrees(np.arctan(H / 2 / p[0]))
    log(f"f = {p[0]:.1f} px, k1 = {p[1]:+.4f}, IMU time offset {p[5]:+.1f} ms; residual median {np.median(e):.2f} px, p90 {np.percentile(e, 90):.2f}")
    log(f"HFOV {hfov:.1f} deg, VFOV {vfov:.1f} deg; R_bc =\n{Rbc.round(4)}")
    pm = np.array([np.median(np.linalg.norm(o, axis=1)) for o in pp])
    log("per-pair median residual px: p50 %.2f p90 %.2f max %.2f" % (np.median(pm), np.percentile(pm, 90), pm.max()))
    C = dict(f_px=float(p[0]), k1=float(p[1]), cx=cx, cy=cy, width=W, height=H, R_bc=Rbc.tolist(), imu_time_offset_s=float(p[5]) * 1e-3,
             gyro_bias_rad_s=bg.tolist(), residual_median_px=float(np.median(e)), residual_p90_px=float(np.percentile(e, 90)),
             n_pairs=len(data), n_points=int(len(e)), hfov_deg=hfov, vfov_deg=vfov,
             note="b2 = R_bc^T dR^T R_bc b1; R_bc columns = camera x (right), y (down), z (optical axis) in IMU axes; image = the video's upright frame")
    json.dump(C, open(out_path, "w"), indent=1)
    sess.attach_calib(C)
    return C
