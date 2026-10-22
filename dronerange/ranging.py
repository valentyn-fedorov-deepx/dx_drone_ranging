"""Ground-ray ranging for every detection: contact pixel (bottom centre of the box) -> camera ray (calibrated f, k1)
-> world (IMU attitude, camera-IMU rotation, IMU time offset) -> intersection with the ground at the frame's height.

Outputs per detection: ground distance D, slant distance S, bearing relative to the camera's ground heading (radar,
compass-independent), absolute bearing (IMU yaw: unreliable on this unit), ENU position of the object (drone GNSS +
offset). With a terrain model the ray is intersected with the terrain instead of the flat plane through the ground
under the drone.
"""
import math

import numpy as np
import pandas as pd

from .camera import intersect_ground, intersect_terrain

OPTICAL_AXIS = np.array([0, 0, 1.0])


def run(sess, dets, track, terrain=None, log=print):
    cam, ft = sess.cam, sess.ft
    rows = []
    for i, g in dets.groupby("frame"):
        i = int(i)
        t = ft[i]
        Rwc = sess.R_wc(t)
        E0, N0 = sess.drone_EN(t)
        h = float(track.h.values[i])
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
        Dh = np.hypot(dE, dN)
        brg = np.degrees(np.arctan2(dE, dN))
        rel = ((brg - cam_brg + 180) % 360) - 180
        for k, rr in enumerate(g.itertuples()):
            fin = bool(ok[k] and np.isfinite(Dh[k]))
            rows.append(dict(frame=i, t=t - ft[0], track=rr.track, cls=rr.cls, conf=rr.conf, x1=rr.x1, y1=rr.y1, x2=rr.x2, y2=rr.y2,
                             h=h, D=Dh[k] if fin else np.nan, S=math.hypot(Dh[k], h - dz[k]) if fin else np.nan,
                             rel_brg=rel[k], brg=brg[k], E=E0 + dE[k], N=N0 + dN[k], E_drone=E0, N_drone=N0, cam_brg=cam_brg,
                             depr=math.degrees(math.asin(min(1.0, -b[k, 2]))) if b[k, 2] < 0 else np.nan))
    R = pd.DataFrame(rows)
    log(len(R), "ranged detections;", round(float(R.D.notna().mean()), 3), "with a ground hit")
    log(R.groupby(R.cls.where(R.cls == "person", "vehicle")).D.describe().round(1).to_string())
    return R
