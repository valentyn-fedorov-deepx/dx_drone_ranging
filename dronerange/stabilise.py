"""Video stabilisation for the HUD: every frame is turned to a smooth virtual camera.

The real camera orientation per frame is the one used for ranging (gyro, tilt fusion, heading), written as azimuth,
elevation and roll of the optical axis. The virtual camera follows the smoothest path (least squared second and first
differences) that stays within a corridor around the real one: the slight zoom of the view (zoom) leaves a margin,
and the turn may use 80 % of it, so the turned frames always fill the view. Each frame is resampled through the lens
model as seen from the virtual camera (a pure rotation; the output is a pinhole view with the camera's principal
point). What the rotation leaves (the drone's sway: a few pixels) is measured on the turned frames (phase correlation
of consecutive frames) and smoothed inside the rest of the margin the same way. The path breaks at recording pauses.
Labels, boxes and clicks go through the same mapping (`to_view`, `to_camera`); the measurements are not touched.

`python -m dronerange --config <session>.yaml stabilise` computes the path for every frame (stabilise.npz); render
and gui use it with `hud.stabilise: true`.
"""
import math

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize


def angles_from_R(R):
    """camera-to-world rotations (n, 3, 3) -> azimuth (from north, clockwise), elevation, roll of the camera (rad)"""
    z, x = R[:, :, 2], R[:, :, 0]
    az = np.unwrap(np.arctan2(z[:, 0], z[:, 1]))
    el = np.arcsin(np.clip(z[:, 2], -1, 1))
    x0 = np.c_[np.cos(az), -np.sin(az), np.zeros_like(az)]
    y0 = np.cross(z, x0)
    roll = np.arctan2((x * y0).sum(1), (x * x0).sum(1))
    return az, el, roll


def R_from_angles(az, el, roll):
    """inverse of angles_from_R: camera x right, y down, z along the optical axis; world E, N, Up"""
    z = np.c_[np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)]
    x0 = np.c_[np.cos(az), -np.sin(az), np.zeros_like(az)]
    y0 = np.cross(z, x0)
    c, s = np.cos(roll)[:, None], np.sin(roll)[:, None]
    x, y = c * x0 + s * y0, -s * x0 + c * y0
    return np.stack([x, y, z], axis=2)


def smooth_in_corridor(a, half, w1=0.05, w2=1.0):
    """the sequence v closest to straight and still (w2 sum (second differences)^2 + w1 sum (first differences)^2)
    with |v - a| <= half"""
    a = np.asarray(a, float)
    n = len(a)
    if n < 4:
        return a.copy()
    half = np.broadcast_to(np.asarray(half, float), a.shape)

    def dt1(u):                                                     # transpose of np.diff
        return np.r_[0.0, u] - np.r_[u, 0.0]

    def f(v):
        d1 = np.diff(v); d2 = np.diff(v, 2)
        g = 2 * w2 * dt1(dt1(d2)) + 2 * w1 * dt1(d1)
        return w2 * (d2 @ d2) + w1 * (d1 @ d1), g

    x0 = np.clip(ndimage.gaussian_filter1d(a, 3.0, mode="nearest"), a - half, a + half)
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(a - half, a + half)),
                 options=dict(maxiter=20000, maxfun=40000, ftol=1e-15, gtol=1e-12))
    return np.clip(r.x, a - half, a + half)


def segments(t, gap_s=0.5):
    """index ranges between recording pauses"""
    cut = np.r_[0, np.nonzero(np.diff(t) > gap_s)[0] + 1, len(t)]
    return [slice(cut[k], cut[k + 1]) for k in range(len(cut) - 1)]


class Stabiliser:
    def __init__(self, sess, zoom=1.2, out_size=(1280, 938), turn_share=0.8, roll_deg=0.4, w1=0.05, grid_step=8, path=None):
        self.sess, self.cam = sess, sess.cam
        self.W, self.H = self.cam.width, self.cam.height
        self.ow, self.oh = out_size
        self.zoom = zoom
        self.frames = np.arange(sess.n_frames)
        self.t = sess.ft[self.frames]
        self.margin = np.array([(self.ow / 2) * (1 - 1 / zoom), (self.oh / 2) * (1 - 1 / zoom)])   # output px, x / y
        self.shift = np.zeros((len(self.frames), 2))                 # residual shift of the view (output pixels)
        if path is not None:
            self.Rw, self.Rv, self.shift = path["Rw"], path["Rv"], path["shift"]
        else:
            self.Rw = np.array([sess.R_wc(t) for t in self.t])
            az, el, roll = angles_from_R(self.Rw)
            mh = turn_share * (self.W / 2) * (1 - 1 / zoom) / self.cam.f
            mv = turn_share * (self.H / 2) * (1 - 1 / zoom) / self.cam.fy
            v = [np.empty_like(az) for _ in range(3)]
            for sl in segments(self.t):
                v[0][sl] = smooth_in_corridor(az[sl], mh / np.maximum(np.cos(el[sl]), 0.2), w1)
                v[1][sl] = smooth_in_corridor(el[sl], mv, w1)
                v[2][sl] = smooth_in_corridor(roll[sl], math.radians(roll_deg), w1)
            self.Rv = R_from_angles(*v)
        self.D = np.einsum("nji,njk->nik", self.Rw, self.Rv)          # real -> virtual axes: Rv = Rw D
        self.step = grid_step                                        # coarse grid of the output view (pixels)
        gx = np.arange(0, self.ow + grid_step, grid_step, dtype=np.float64)
        gy = np.arange(0, self.oh + grid_step, grid_step, dtype=np.float64)
        self.gshape = (len(gy), len(gx))
        self.GX, self.GY = (a.ravel() for a in np.meshgrid(gx, gy))

    def save(self, path):
        np.savez_compressed(path, Rw=self.Rw, Rv=self.Rv, shift=self.shift, zoom=self.zoom, out=np.array([self.ow, self.oh]))

    @classmethod
    def load(cls, sess, path, **kw):
        z = np.load(path)
        return cls(sess, zoom=float(z["zoom"]), out_size=tuple(int(x) for x in z["out"]), path=dict(Rw=z["Rw"], Rv=z["Rv"], shift=z["shift"]), **kw)

    # ---- per frame
    def _D(self, i):
        return self.D[int(i)], self.shift[int(i)]

    def maps(self, i, src_size):
        """remap maps from the output view to a source picture of src_size (w, h) of the camera's view"""
        D, sh = self._D(i)
        u = (self.GX + sh[0]) * self.W / self.ow                    # output x samples the turned view at x + shift
        v = (self.GY + sh[1]) * self.H / self.oh
        bv = np.c_[(u - self.cam.cx) / (self.cam.f * self.zoom), (v - self.cam.cy) / (self.cam.fy * self.zoom), np.ones_like(u)]
        px = self.cam.project(bv @ D.T)
        sx, sy = src_size[0] / self.W, src_size[1] / self.H
        gh, gw = self.gshape
        z = (((gh - 1) * self.step + 1) / gh, ((gw - 1) * self.step + 1) / gw)     # grid point k = output pixel k * step
        mx = ndimage.zoom((px[:, 0] * sx).reshape(self.gshape), z, order=1, grid_mode=False, mode="nearest")
        my = ndimage.zoom((px[:, 1] * sy).reshape(self.gshape), z, order=1, grid_mode=False, mode="nearest")
        return np.ascontiguousarray(mx[:self.oh, :self.ow], np.float32), np.ascontiguousarray(my[:self.oh, :self.ow], np.float32)

    def coverage(self, i):
        """fraction of the view whose source pixel lies outside the camera frame (0 = the frame fills the view)"""
        mx, my = self.maps(i, (self.W, self.H))
        return float(np.mean((mx < 0) | (my < 0) | (mx > self.W - 1) | (my > self.H - 1)))

    def warp(self, picture, i):
        """camera picture (any size of the camera's view) -> stabilised view (out_size)"""
        mx, my = self.maps(i, (picture.shape[1], picture.shape[0]))
        return cv2.remap(picture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(12, 14, 12))

    def to_view(self, i, pts):
        """camera pixels (N x 2, full resolution) -> output view pixels"""
        D, sh = self._D(i)
        b = self.cam.bearings(np.atleast_2d(np.asarray(pts, float))) @ D      # b_v = D^T b_c
        u = self.cam.cx + self.cam.f * self.zoom * b[:, 0] / b[:, 2]
        v = self.cam.cy + self.cam.fy * self.zoom * b[:, 1] / b[:, 2]
        return np.c_[u * self.ow / self.W - sh[0], v * self.oh / self.H - sh[1]]

    def to_camera(self, i, pts):
        """output view pixels -> camera pixels (full resolution)"""
        D, sh = self._D(i)
        p = np.atleast_2d(np.asarray(pts, float)) + sh
        u, v = p[:, 0] * self.W / self.ow, p[:, 1] * self.H / self.oh
        bv = np.c_[(u - self.cam.cx) / (self.cam.f * self.zoom), (v - self.cam.cy) / (self.cam.fy * self.zoom), np.ones(len(u))]
        return self.cam.project(bv @ D.T)

    def residual(self, read, share=0.2, w1=0.05, size=(640, 469), log=print):
        """take out the shift the rotation leaves: phase correlation of consecutive turned frames (read(i) -> BGR
        picture of frame i), accumulated, smoothed inside share of the margin"""
        win = cv2.createHanningWindow(size, cv2.CV_32F)
        acc = np.zeros((len(self.frames), 2)); prev = None
        for k, f in enumerate(self.frames):
            g = cv2.cvtColor(self.warp(read(int(f)), int(f)), cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev is not None and self.t[k] - self.t[k - 1] <= 0.5:
                (dx, dy), _ = cv2.phaseCorrelate(prev, g, win)
                acc[k] = acc[k - 1] + np.array([dx, dy]) * self.ow / size[0]
            elif prev is not None:
                acc[k] = acc[k - 1]
            prev = g
            if k % 500 == 0:
                log("stabilise: residual shift", int(f))
        sm = np.empty_like(acc)
        for sl in segments(self.t):
            for j in (0, 1):
                sm[sl, j] = smooth_in_corridor(acc[sl, j], share * self.margin[j], w1)
        self.shift = acc - sm
        return self.shift
