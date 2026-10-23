"""One recording session: frame timestamps, the IMU/GNSS log, attitude, gyro orientation and drone positions.

Clocks: frame file names carry UTC wall-clock time with microseconds, the IMU log carries Unix nanoseconds on the
same clock. The calibrated IMU time offset `toff` is added to frame times before looking up IMU quantities.
"""
import datetime
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot, Slerp

from .camera import Camera
from .xsens import read_imu

R_EARTH = 6371008.8
_TS = re.compile(r"(\d{4})_(\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d{6})")


def frame_time(name):
    y, mo, d, H, M, S, us = map(int, _TS.match(name).groups())
    return datetime.datetime(y, mo, d, H, M, S, us, tzinfo=datetime.timezone.utc).timestamp()


def load_frame_times(frames_tsv):
    """FRAMES.tsv (one row per recorded frame) -> DataFrame(frame, t) sorted by time; row k = video frame k."""
    fr = pd.read_csv(frames_tsv, sep="\t")
    fr["t"] = fr.frame.map(frame_time)
    return fr.sort_values("t").reset_index(drop=True)[["frame", "t"]]


def integrate_gyro(ti, gyr, bias, scale=1.0):
    """Body orientation relative to the first sample from the bias-corrected (and scale-corrected) gyro, as a
    continuous Slerp."""
    qs = [Rot.identity()]
    for k in range(1, len(ti)):
        qs.append(qs[-1] * Rot.from_rotvec((gyr[k - 1] - bias) * scale * (ti[k] - ti[k - 1])))
    return Slerp(ti, Rot.concatenate(qs))


def lowpass(t, x, tau):
    """causal first-order low-pass with time constant tau (s) on an irregular time grid"""
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for k in range(1, len(x)):
        a = (t[k] - t[k - 1]) / (tau + t[k] - t[k - 1])
        y[k] = y[k - 1] + a * (x[k] - y[k - 1])
    return y


def lowpass_zero_phase(t, x, tau):
    """the same filter forward and backward: no lag (offline)"""
    y = lowpass(t, x, tau)
    return lowpass(-t[::-1], y[::-1], tau)[::-1]


class Video:
    """Random access (exact seeking was checked for these H.264 files) plus sequential reading."""

    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(self.path)
        self.next = 0

    def read(self, i):
        if i != self.next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, im = self.cap.read()
        self.next = i + 1
        return im if ok else None

    def gray(self, i):
        im = self.read(i)
        return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


class Session:
    def __init__(self, cfg, calib=None):
        self.cfg = cfg
        data = cfg["data"]
        self.video_path = data["video"]
        d = read_imu(data["imu"])
        self.imu = d
        self.ti = d.t_ns.values * 1e-9
        fr = load_frame_times(data["frames_tsv"])
        self.frame_names = fr.frame.values
        self.ft = fr.t.values
        self.tt = self.ft - self.ft[0]
        self.n_frames = len(self.ft)
        self.gyr = d[["gx", "gy", "gz"]].values
        eul = d[["roll", "pitch", "yaw"]].values.copy()
        eul[:, 2] = np.degrees(np.unwrap(np.radians(eul[:, 2])))      # yaw interpolates across +-180 without a jump
        self.eul = eul
        self.eul_raw = eul.copy()
        self.speed = d.ground_speed.values
        self.nsat = d.num_sats.values
        if cfg.get("enu_origin"):
            lat0, lon0 = cfg["enu_origin"]
        else:
            ok = (d.gnss_valid.values > 0) & (d.fix_type.values >= 3)
            k = int(np.argmax(ok))
            lat0, lon0 = float(d.lat.values[k]), float(d.lon.values[k])
        self.lat0, self.lon0 = lat0, lon0
        self.E = np.radians(d.lon.values - lon0) * R_EARTH * math.cos(math.radians(lat0))
        self.N = np.radians(d.lat.values - lat0) * R_EARTH
        ve, vn = d.velE.values, d.velN.values                        # receiver velocity integrated: smooth displacement
        self.E_vel = np.r_[0, np.cumsum(0.5 * (ve[1:] + ve[:-1]) * np.diff(self.ti))]
        self.N_vel = np.r_[0, np.cumsum(0.5 * (vn[1:] + vn[:-1]) * np.diff(self.ti))]
        tau = cfg.get("position_smoothing_s", 5.0)                   # complementary filter: GNSS position (low) + velocity track (high)
        lp = lowpass if cfg.get("position_smoothing_causal", False) else lowpass_zero_phase
        self.E_s = self.E_vel + lp(self.ti, self.E - self.E_vel, tau)
        self.N_s = self.N_vel + lp(self.ti, self.N - self.N_vel, tau)
        self.heading = None                                          # HeadingModel: corrected yaw (see heading.py)
        self.gyro_scale = 1.0
        self.toff = 0.0
        self.cam = None
        self.R_bc = None
        self.bias = None
        self._slerp = None
        self.calib = None
        if calib is not None:
            self.attach_calib(calib)

    # ---- calibration
    def gyro_bias(self, seconds):
        return self.gyr[self.ti < self.ti[0] + seconds].mean(0)

    def set_bias(self, bias):
        self.bias = np.asarray(bias, float)
        self._slerp = None

    def attach_calib(self, c):
        if isinstance(c, (str, Path)):
            c = json.load(open(c))
        self.calib = c
        self.cam = Camera.from_calib(c)
        self.R_bc = np.array(c["R_bc"])
        self.toff = float(c["imu_time_offset_s"])
        self.gyro_scale = float(c.get("gyro_scale", 1.0))
        self.set_bias(c["gyro_bias_rad_s"])
        w = self.cfg.get("attitude", {}).get("tilt_window_s", 10.0)
        if w:
            self.fuse_tilt(w)
        return self

    def fuse_tilt(self, window_s=10.0):
        """Complementary roll and pitch: the gyro orientation for everything faster than window_s, the Xsens roll and
        pitch (gravity-referenced, but off by about 1 deg over a few seconds in hover) for the slow part. The world up
        vector seen in the body (from the Xsens) is carried into the gyro's reference frame, where it should stand
        still, averaged over the window (centred), and brought back to the body. The yaw is left alone."""
        Rg = self.slerp(self.ti).as_matrix()
        r = np.radians(self.eul_raw[:, 0]); p = np.radians(self.eul_raw[:, 1])
        u = np.c_[-np.sin(p), np.sin(r) * np.cos(p), np.cos(r) * np.cos(p)]
        w = np.einsum("nij,nj->ni", Rg, u)
        cs = np.vstack([np.zeros(3), np.cumsum(w, axis=0)])
        lo = np.searchsorted(self.ti, self.ti - window_s / 2)
        hi = np.searchsorted(self.ti, self.ti + window_s / 2, side="right")
        wf = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
        wf /= np.linalg.norm(wf, axis=1, keepdims=True)
        uf = np.einsum("nji,nj->ni", Rg, wf)
        self.eul = self.eul_raw.copy()
        self.eul[:, 1] = -np.degrees(np.arcsin(np.clip(uf[:, 0], -1, 1)))
        self.eul[:, 0] = np.degrees(np.arctan2(uf[:, 1], uf[:, 2]))

    @property
    def slerp(self):
        if self._slerp is None:
            self._slerp = integrate_gyro(self.ti, self.gyr, self.bias, self.gyro_scale)
        return self._slerp

    # ---- IMU lookups (frame / unix times; the IMU offset is applied here)
    def dR_gyro(self, t1, t2, toff=None):
        """body rotation between t1 and t2 (R_ref_b(t1)^T R_ref_b(t2))"""
        o = self.toff if toff is None else toff
        r = self.slerp([t1 + o, t2 + o])
        return (r[0].inv() * r[1]).as_matrix()

    def at(self, t, arr):
        return np.interp(t + self.toff, self.ti, arr)

    def R_wb_xsens(self, t):
        """roll and pitch as fused (see fuse_tilt), the Xsens magnetic yaw"""
        r, p, y = [np.interp(t + self.toff, self.ti, self.eul[:, k]) for k in range(3)]
        return Rot.from_euler("ZYX", [y, p, r], degrees=True).as_matrix()

    def R_wb(self, t):
        """attitude used for ranging: fused roll and pitch; the corrected heading when a heading model is attached"""
        if self.heading is None:
            return self.R_wb_xsens(t)
        r, p = [np.interp(t + self.toff, self.ti, self.eul[:, k]) for k in range(2)]
        return Rot.from_euler("ZYX", [self.heading.yaw(t + self.toff), p, r], degrees=True).as_matrix()

    def R_wc(self, t):
        return self.R_wb(t) @ self.R_bc

    def frame_R_wc(self, i):
        return self.R_wc(self.ft[i])

    def drone_EN(self, t, smooth=False):
        if smooth:
            return float(self.at(t, self.E_s)), float(self.at(t, self.N_s))
        return float(self.at(t, self.E)), float(self.at(t, self.N))

    def en_to_latlon(self, e, n):
        lat = self.lat0 + np.degrees(np.asarray(n) / R_EARTH)
        return lat, self.lon0 + np.degrees(np.asarray(e) / (R_EARTH * math.cos(math.radians(self.lat0))))

    def latlon_to_en(self, lat, lon):
        return (np.radians(np.asarray(lon) - self.lon0) * R_EARTH * math.cos(math.radians(self.lat0)),
                np.radians(np.asarray(lat) - self.lat0) * R_EARTH)

    def vel_EN(self, t):
        return float(self.at(t, self.E_vel)), float(self.at(t, self.N_vel))

    def video(self):
        return Video(self.video_path)

    def write_frame_times(self, path):
        pd.DataFrame(dict(frame=self.frame_names, t=self.ft)).to_csv(path, index=False)
