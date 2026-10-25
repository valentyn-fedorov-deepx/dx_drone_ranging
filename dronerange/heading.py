"""Heading (yaw) for map positions and motion directions.

The Xsens magnetic yaw is off by tens of degrees near the ground and by about 8 deg at height (the static-car
trilateration shows it). Ranging distances and the drone-relative radar do not need the heading; map positions and
the direction of motion do.

Here the yaw comes from the gyro: the Euler yaw rate psi_dot = (w_y sin(roll) + w_z cos(roll)) / cos(pitch) with the
bias- and scale-corrected body rates and the Xsens roll and pitch (gravity-referenced, reliable), integrated. It is
smooth and drifts slowly. Its absolute offset comes from the visual motion direction: for every GNSS + homography pair
(altitude step) the image gives the direction of the camera's horizontal displacement, the GNSS gives it in the
world; their difference is the heading error at that moment. With the Xsens attitude the pairs give
e = bearing_img - bearing_gnss, so yaw_true = yaw_xsens + e (ENU yaw is counter-clockwise, bearings clockwise), and
the correction of the gyro yaw is c = yaw_xsens + e - yaw_gyro. The anchors are smoothed with a rolling median and
interpolated in time (held constant before the first and after the last anchor):
    yaw_used(t) = yaw_gyro(t) + c(t)
"""
import numpy as np
import pandas as pd


def wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


class HeadingModel:
    """yaw_used on the IMU time grid (ENU degrees, unwrapped); yaw(t) takes IMU times"""

    def __init__(self, t, yaw):
        self.t = np.asarray(t, float)
        self.y = np.asarray(yaw, float)

    def yaw(self, t):
        return float(np.interp(t, self.t, self.y))

    def yaws(self, t):
        return np.interp(t, self.t, self.y)

    @classmethod
    def load(cls, path):
        d = pd.read_csv(path)
        return cls(d.t_imu.values, d.yaw_used.values)


def gyro_yaw(sess):
    """integrated Euler yaw rate (deg, unwrapped) on the IMU grid, starting at the first Xsens yaw"""
    w = (sess.gyr - sess.bias) * sess.gyro_scale
    r = np.radians(sess.eul[:, 0]); p = np.radians(sess.eul[:, 1])
    ydot = (w[:, 1] * np.sin(r) + w[:, 2] * np.cos(r)) / np.cos(p)
    dt = np.diff(sess.ti)
    return sess.eul[0, 2] + np.degrees(np.r_[0, np.cumsum(ydot[:-1] * dt)])


def anchors_from_pairs(sess, Pd, good):
    """heading anchors from the altitude pairs: time (IMU clock), the Xsens-based heading error e, the correction c"""
    from .altitude import good_pairs
    g = good_pairs(Pd, good).copy()
    t1 = sess.ft[g.i.astype(int).values] + sess.toff
    g["t_imu"] = t1
    g["e_xsens"] = wrap(g.dir_img - g.dir_gnss)
    g["yaw_xsens"] = np.interp(t1, sess.ti, sess.eul[:, 2])
    return g


def fit(sess, Pd, good, window_s=30.0, min_n=5, gap_s=20.0, log=print):
    """anchors are grouped into clusters (time gaps over gap_s); a cluster needs min_n anchors; outliers (beyond
    3 MAD, at least 6 deg, from the cluster median) are dropped; inside a cluster a rolling median over window_s
    follows the slow gyro drift; between clusters the correction is interpolated linearly"""
    yg = gyro_yaw(sess)
    A = anchors_from_pairs(sess, Pd, good)
    A["yaw_gyro"] = np.interp(A.t_imu, sess.ti, yg)
    c = A.yaw_xsens + A.e_xsens - A.yaw_gyro
    c0 = np.median(wrap(c))
    A["c_raw"] = c0 + wrap(c - c0)                                   # unwrapped around the median
    A = A.sort_values("t_imu").reset_index(drop=True)
    A["cluster"] = np.r_[0, np.cumsum(np.diff(A.t_imu.values) > gap_s)]
    A["c"] = np.nan
    A["outlier"] = True
    for cl, g in A.groupby("cluster"):
        if len(g) < min_n:
            continue
        med = np.median(g.c_raw); mad = 1.4826 * np.median(np.abs(g.c_raw - med))
        inl = np.abs(g.c_raw - med) <= max(3 * mad, 6.0)
        gi = g[inl]
        if len(gi) < min_n:
            continue
        tv, cv = gi.t_imu.values, gi.c_raw.values
        A.loc[gi.index, "c"] = [np.median(cv[np.abs(tv - t) <= window_s / 2]) for t in tv]
        A.loc[gi.index, "outlier"] = False
    ok = A.c.notna() & ~A.outlier
    if ok.sum() == 0:
        log("heading: no usable anchors, keeping the Xsens yaw")
        return HeadingModel(sess.ti, sess.eul[:, 2]), A, pd.DataFrame(dict(t_imu=sess.ti, yaw_xsens=sess.eul[:, 2], yaw_gyro=yg,
                                                                              c=np.zeros(len(yg)), yaw_used=sess.eul[:, 2]))
    ct = np.interp(sess.ti, A.t_imu[ok].values, A.c[ok].values)
    yaw_used = yg + ct
    A["resid_after"] = wrap(A.yaw_xsens + A.e_xsens - (A.yaw_gyro + np.interp(A.t_imu, sess.ti, ct)))
    T = pd.DataFrame(dict(t_imu=sess.ti, yaw_xsens=sess.eul[:, 2], yaw_gyro=yg, c=ct, yaw_used=yaw_used))
    err_x = wrap(A.e_xsens[ok])
    log(f"heading anchors {int(ok.sum())} (of {len(A)}): Xsens heading error median {np.median(err_x):+.1f} deg, "
        f"p10..p90 {np.percentile(err_x, 10):+.1f}..{np.percentile(err_x, 90):+.1f}; after the correction residual "
        f"median {np.median(A.resid_after[ok]):+.1f}, MAD {1.4826 * np.median(np.abs(A.resid_after[ok] - np.median(A.resid_after[ok]))):.1f} deg")
    return HeadingModel(sess.ti, yaw_used), A, T


def plot(path, sess, A, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = sess.ft[0]
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(T.t_imu - t0, wrap(T.yaw_xsens - T.yaw_used), lw=0.8, label="Xsens yaw - yaw used")
    ax[0].plot(A.t_imu - t0, wrap(A.yaw_xsens + A.e_xsens - np.interp(A.t_imu, T.t_imu, T.yaw_used)), ".", ms=4,
               label="anchor: visual-GNSS heading - yaw used")
    ax[0].set_ylabel("deg"); ax[0].set_ylim(-120, 120); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].plot(T.t_imu - t0, T.c, lw=1.2, label="gyro yaw correction c(t)")
    ax[1].plot(A.t_imu - t0, A.c_raw, ".", ms=3, color="0.6", label="anchors")
    ax[1].set_ylabel("deg"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8); ax[1].set_xlabel("s from the first frame")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)


def fit_dji(sess, Pd, good, dji, yaw_col="yaw_deg", log=print):
    """Heading from the aircraft's own log: the camera unit is fixed to the airframe, so its ENU yaw is
    c0 - (DJI yaw, a compass bearing) with one constant c0 (the mount). c0 comes from the visual-GNSS anchors (moving
    stretches); the gyro yaw is compared with the DJI yaw as a check of the sign and of the drift between them."""
    from .telemetry import yaw_at
    yd = yaw_at(dji, sess.ti, yaw_col)
    yg = gyro_yaw(sess)
    A = anchors_from_pairs(sess, Pd, good) if len(Pd) else pd.DataFrame(columns=["t_imu", "e_xsens", "yaw_xsens"])
    if len(A):
        A["yaw_dji"] = np.interp(A.t_imu, sess.ti, yd)
        c = A.yaw_xsens + A.e_xsens + A.yaw_dji                     # yaw_true + bearing = c0
        c0 = np.median(wrap(c))
        A["c_raw"] = c0 + wrap(c - c0)
        med = np.median(A.c_raw); mad = 1.4826 * np.median(np.abs(A.c_raw - med))
        A["outlier"] = np.abs(A.c_raw - med) > max(3 * mad, 6.0)
        c0 = float(np.median(A.c_raw[~A.outlier]))
        A["c"] = c0
        A["resid_after"] = wrap(A.yaw_xsens + A.e_xsens - (c0 - A.yaw_dji))
        ok = ~A.outlier
        log(f"heading from the DJI log: mount constant c0 {c0:.1f} deg from {int(ok.sum())} anchors (of {len(A)}), residual median "
            f"{np.median(A.resid_after[ok]):+.1f}, MAD {1.4826 * np.median(np.abs(A.resid_after[ok] - np.median(A.resid_after[ok]))):.1f} deg")
    else:
        c0 = float(np.median(wrap(sess.eul[:, 2] + yd)))
        log(f"heading from the DJI log: no visual anchors, mount constant from the Xsens yaw (unreliable): c0 {c0:.1f} deg")
    yaw_used = c0 - yd
    d = (yg - yg[0]) - (yaw_used - yaw_used[0])
    tt = sess.ti - sess.ti[0]
    b = np.polyfit(tt, d, 1)[0]
    log(f"gyro yaw vs DJI yaw: drift {b * 60:+.2f} deg/min, scatter about the drift {np.std(d - np.polyval(np.polyfit(tt, d, 1), tt)):.2f} deg")
    T = pd.DataFrame(dict(t_imu=sess.ti, yaw_xsens=sess.eul[:, 2], yaw_gyro=yg, c=np.full(len(yg), c0), yaw_used=yaw_used, yaw_dji=yd))
    return HeadingModel(sess.ti, yaw_used), A, T
