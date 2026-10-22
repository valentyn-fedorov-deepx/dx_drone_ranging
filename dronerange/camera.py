"""Pinhole camera with one radial distortion term, and camera rays meeting the ground.

Conventions
    image: pixel (u right, v down), principal point at the image centre unless calibrated otherwise
    camera frame: x right, y down, z = optical axis (OpenCV)
    world frame: ENU (x east, y north, z up); R_wc maps camera vectors to world vectors
    distortion: x_d = x_u (1 + k1 r_u^2) in normalised coordinates; its first-order inverse is used for undistortion
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    f: float
    cx: float
    cy: float
    k1: float = 0.0
    width: int = 2048
    height: int = 1500

    @classmethod
    def from_calib(cls, c):
        return cls(f=c["f_px"], cx=c["cx"], cy=c["cy"], k1=c.get("k1", 0.0), width=c.get("width", 2048), height=c.get("height", 1500))

    @property
    def K(self):
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    def normalised(self, px):
        px = np.atleast_2d(np.asarray(px, float))
        x = (px[:, 0] - self.cx) / self.f
        y = (px[:, 1] - self.cy) / self.f
        s = 1.0 / (1.0 + self.k1 * (x * x + y * y))
        return x * s, y * s

    def undistort(self, px):
        """distorted pixels -> ideal pinhole pixels"""
        px = np.atleast_2d(np.asarray(px, float))
        x = (px[:, 0] - self.cx) / self.f
        y = (px[:, 1] - self.cy) / self.f
        s = 1.0 / (1.0 + self.k1 * (x * x + y * y))
        return np.c_[self.cx + self.f * x * s, self.cy + self.f * y * s]

    def bearings(self, px):
        """pixels -> unit rays in the camera frame"""
        x, y = self.normalised(px)
        b = np.c_[x, y, np.ones(len(x))]
        return b / np.linalg.norm(b, axis=1, keepdims=True)

    def project(self, b):
        """camera-frame directions -> distorted pixels"""
        b = np.atleast_2d(np.asarray(b, float))
        x = b[:, 0] / b[:, 2]
        y = b[:, 1] / b[:, 2]
        r2 = x * x + y * y
        return np.c_[self.cx + self.f * x * (1 + self.k1 * r2), self.cy + self.f * y * (1 + self.k1 * r2)]


def world_rays(cam, R_wc, px):
    return cam.bearings(px) @ np.asarray(R_wc).T


def intersect_ground(rays, h):
    """Rays from a camera h metres above a horizontal ground plane -> (dE, dN, valid); dE, dN relative to the nadir."""
    rays = np.atleast_2d(rays)
    down = -rays[:, 2]
    valid = down > 1e-6
    lam = np.where(valid, h / np.maximum(down, 1e-12), np.nan)
    return rays[:, 0] * lam, rays[:, 1] * lam, valid


def intersect_terrain(rays, cam_e, cam_n, cam_z, height_at, max_range=800.0, step=1.0):
    """Rays from the camera at absolute height cam_z (same datum as the terrain) -> first terrain hit by ray marching
    (step metres of horizontal travel) refined by bisection. height_at(e, n) returns the terrain height (vectorised).
    Returns (E, N, Z, valid) in absolute coordinates."""
    rays = np.atleast_2d(rays)
    out = np.full((len(rays), 3), np.nan)
    valid = np.zeros(len(rays), bool)
    for k, r in enumerate(rays):
        hz = np.hypot(r[0], r[1])
        if hz < 1e-9 or r[2] > 0.3 * hz:                       # pointing up: no hit
            continue
        ts = np.arange(step, max_range, step) / hz              # ray parameter for each horizontal step
        pe, pn, pz = cam_e + r[0] * ts, cam_n + r[1] * ts, cam_z + r[2] * ts
        below = pz <= height_at(pe, pn)
        if not below.any():
            continue
        j = int(np.argmax(below))
        lo, hi = (ts[j - 1] if j > 0 else 0.0), ts[j]
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            pm = np.array([cam_e + r[0] * mid]), np.array([cam_n + r[1] * mid])
            if cam_z + r[2] * mid <= height_at(*pm)[0]:
                hi = mid
            else:
                lo = mid
        t = 0.5 * (lo + hi)
        out[k] = (cam_e + r[0] * t, cam_n + r[1] * t, cam_z + r[2] * t)
        valid[k] = True
    return out[:, 0], out[:, 1], out[:, 2], valid
