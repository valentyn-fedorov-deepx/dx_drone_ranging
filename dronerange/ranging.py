"""Ground-ray ranging for every detection: contact pixel (bottom centre of the box) -> camera ray (calibrated lens)
-> world (IMU attitude, camera-IMU rotation, IMU time offset) -> intersection with the ground at the frame's height.

Outputs per detection: ground distance D, slant distance S, bearing relative to the camera's ground heading (radar,
compass-independent), absolute bearing, ENU position of the contact point (E, N) and of the tracked reference point
(Er, Nr: the feet for people, the middle of the body 0.75 m above the ground for vehicles), drone position. With a
heading model attached the absolute bearings and positions use the corrected heading; the drone position is the
smoothed one (GNSS position low-passed + the receiver velocity track). With a terrain model the ray is intersected with
the terrain instead of the flat plane through the ground under the drone.
"""
import math

import numpy as np
import pandas as pd

from .camera import intersect_ground, intersect_height, intersect_terrain
from .worldtrack import GROUPS, REF_HEIGHT

OPTICAL_AXIS = np.array([0, 0, 1.0])


def run(sess, dets, track, terrain=None, log=print):
    rows = []
    for i, g in dets.groupby("frame"):
        rows += range_frame(sess, int(i), g, float(track.h.values[int(i)]), terrain)
    R = pd.DataFrame(rows)
    log(len(R), "ranged detections;", round(float(R.D.notna().mean()), 3), "with a ground hit")
    log(R.groupby(R.cls.where(R.cls == "person", "vehicle")).D.describe().round(1).to_string())
    return R


def range_frame(sess, i, g, h, terrain=None):
    """the detections g of frame i (x1..y2, cls, conf, track) at height h -> one row per detection"""
    cam, ft = sess.cam, sess.ft
    rows = []
    t = ft[i]
    Rwc = sess.R_wc(t)
    E0, N0 = sess.drone_EN(t, smooth=True)
    ax = Rwc @ OPTICAL_AXIS
    cam_brg = math.degrees(math.atan2(ax[0], ax[1]))
    px = np.c_[0.5 * (g.x1.values + g.x2.values), g.y2.values]
    b = cam.bearings(px) @ Rwc.T
    if terrain is None:
        dE, dN, ok = intersect_ground(b, h)
        dz = np.zeros(len(b))
    else:
        z_cam = terrain.ground_ref(E0, N0) + h
        Eh, Nh, Zh, ok = intersect_terrain(b, E0, N0, z_cam, terrain.height_at)
        dE, dN, dz = Eh - E0, Nh - N0, Zh - (z_cam - h)
    bc = cam.bearings(np.c_[0.5 * (g.x1.values + g.x2.values), 0.5 * (g.y1.values + g.y2.values)]) @ Rwc.T
    Dh = np.hypot(dE, dN)
    brg = np.degrees(np.arctan2(dE, dN))
    rel = ((brg - cam_brg + 180) % 360) - 180
    for k, rr in enumerate(g.itertuples()):
        fin = bool(ok[k] and np.isfinite(Dh[k]))
        grp = GROUPS.get(rr.cls, "vehicle")
        if grp == "person" or not fin:
            er, nr = dE[k], dN[k]
        else:
            ce, cn, okc = intersect_height(bc[k:k + 1], h - dz[k], REF_HEIGHT[grp])
            er, nr = (ce[0], cn[0]) if okc[0] else (dE[k], dN[k])
        rows.append(dict(frame=i, t=t - ft[0], track=rr.track, cls=rr.cls, conf=rr.conf, x1=rr.x1, y1=rr.y1, x2=rr.x2, y2=rr.y2,
                         h=h, D=Dh[k] if fin else np.nan, S=math.hypot(Dh[k], h - dz[k]) if fin else np.nan,
                         rel_brg=rel[k], brg=brg[k], E=E0 + dE[k], N=N0 + dN[k], Er=E0 + er if fin else np.nan, Nr=N0 + nr if fin else np.nan,
                         E_drone=E0, N_drone=N0, cam_brg=cam_brg,
                         depr=math.degrees(math.asin(min(1.0, -b[k, 2]))) if b[k, 2] < 0 else np.nan))
    return rows


def footprint(cam, Rwc, h, rmax):
    """camera footprint on the ground: image corners and edge midpoints -> ENU offsets from the nadir, clipped at
    rmax; rays above the horizon go to the rim"""
    W, H = cam.width, cam.height
    edge = np.array([[0, 0], [W // 2, 0], [W - 1, 0], [W - 1, H // 2], [W - 1, H - 1], [W // 2, H - 1], [0, H - 1], [0, H // 2]], float)
    b = cam.bearings(edge) @ Rwc.T
    out = []
    for v in b:
        if v[2] < -1e-4:
            lam = h / -v[2]; e, n = v[0] * lam, v[1] * lam
            r = math.hypot(e, n)
            if r > rmax:
                e, n = e / r * rmax, n / r * rmax
        else:
            hz = math.hypot(v[0], v[1]); e, n = v[0] / hz * rmax, v[1] / hz * rmax
        out.append((e, n))
    return np.array(out)


def footprints(sess, track, rmax=300.0):
    """frame -> footprint polygon in absolute ENU (smoothed drone position)"""
    out = {}
    for i in range(sess.n_frames):
        t = sess.ft[i]
        E0, N0 = sess.drone_EN(t, smooth=True)
        G = footprint(sess.cam, sess.R_wc(t), float(track.h.values[i]), rmax)
        out[i] = [(E0 + e, N0 + n) for e, n in G]
    return out


def annotate_tracks(sess, T, track):
    """per-frame track rows -> drone-relative geometry for the labels: ground distance, slant range, camera-frame
    coordinates Xo (right), Yo (down), Zo (along the optical axis) of the tracked point, lat / lon, speed and direction"""
    T = T.copy()
    cols = {k: np.full(len(T), np.nan) for k in ("Xo", "Yo", "Zo", "range", "D", "lat", "lon", "cam_brg", "dir", "rel_dir")}
    for i, g in T.groupby("frame"):
        t = sess.ft[int(i)]
        Rwc = sess.R_wc(t)
        E0, N0 = sess.drone_EN(t, smooth=True)
        h = float(track.h.values[int(i)])
        ax = Rwc @ OPTICAL_AXIS
        cb = math.degrees(math.atan2(ax[0], ax[1]))
        zr = np.array([REF_HEIGHT.get(x, 0.0) for x in g.grp.values]) - h
        Pw = np.c_[g.E.values - E0, g.N.values - N0, zr]
        Pc = Pw @ Rwc                                                # = (R_wc^T Pw^T)^T
        idx = g.index.values
        cols["Xo"][T.index.get_indexer(idx)] = Pc[:, 0]
        cols["Yo"][T.index.get_indexer(idx)] = Pc[:, 1]
        cols["Zo"][T.index.get_indexer(idx)] = Pc[:, 2]
        cols["range"][T.index.get_indexer(idx)] = np.linalg.norm(Pc, axis=1)
        cols["D"][T.index.get_indexer(idx)] = np.hypot(Pw[:, 0], Pw[:, 1])
        cols["cam_brg"][T.index.get_indexer(idx)] = cb
    lat, lon = sess.en_to_latlon(T.E.values, T.N.values)
    cols["lat"], cols["lon"] = lat, lon
    cols["dir"] = np.degrees(np.arctan2(T.vE.values, T.vN.values))
    cols["rel_dir"] = ((cols["dir"] - cols["cam_brg"] + 180) % 360) - 180
    for k, v in cols.items():
        T[k] = v
    T["speed_kmh"] = 3.6 * T.speed
    return T
