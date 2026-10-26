"""Polarisation views from the raw frames, computed the way the VyzAI application computes them.

The sensor (Sony IMX253MYR, 4096 x 3000, 12 bit) has a 4 x 4 cell of colour x polariser angle, stored upside down
(the unit is mounted inverted):

    R 90   R 45   G 90   G 45
    R 135  R 0    G 135  G 0
    G 90   G 45   B 90   B 45
    G 135  G 0    B 135  B 0

The application's way, found by fitting its own decodes of frame 2522 (`frame_2026_10_23_18_24_19/`; `polar --verify`
reproduces the check):
- no black level is subtracted;
- each polariser angle is a 2048 x 1500 Bayer picture (one value per 2 x 2 polariser cell; upright, its top-left cell
  is blue), demosaiced edge-aware; the intensity is the mean of R, G and B;
- S0 = (I0 + I45 + I90 + I135) / 2, S1 = I0 - I90, S2 = I45 - I135; rho = |(S1, S2)| / S0 (degree, 0..1) and
  phi = atan2(S2, S1) / 2 (angle in the image, -90..90 deg);
- Nz: grey = rho / 0.7. Nxy: the jet colour map over phi from -90 to 90 deg. Nxyz: (|sin t cos(phi + 45)|,
  |sin t sin(phi + 45)|, cos t) with t = atan(19 rho);
- the physical models: the zenith from rho by the Fresnel equations with n = 1.5, diffuse (PxDiffuse, VyzLut) or
  specular below the Brewster angle (PxSpecularV2); the azimuth phi taken with y <= 0 (PxDiffuse, PxSpecularV2) or
  phi + 90 deg taken with x <= 0 (VyzLut); colour (v + 1) / 2;
- S0 as a colour picture: the cell sums demosaiced, a colour matrix (fitted to the application's S0 picture of the
  unit) and the sRGB curve.
Against the application on frame 2522 (levels of 255, rms): Nz 1.3, Nxy 13 (one step of the colour map is 4 levels),
Nxyz 5, PxSpecularV2 1.8, PxDiffuse and VyzLut 7 (the diffuse zenith is steep at small rho), S0 1.2.

The viewer and the videos use the Stokes values averaged over 2 x 2 polariser cells (1024 x 750): the application's
live views are smaller still (512 x 375).
"""
import glob
import json
import os
import tarfile

import cv2
import numpy as np

SENSOR_W, SENSOR_H = 4096, 3000
CELL_W, CELL_H = 2048, 1500
BAYER = cv2.COLOR_BayerRG2RGB_EA            # the upright cell picture starts with a blue cell: OpenCV calls that RG

VIEWS = {"s0": "S0", "nz": "Nz", "nxy": "Nxy", "nxyz": "Nxyz",
         "pn_vyzlut": "PN VyzLut", "pn_pxdiffuse": "PN PxDiffuse", "pn_pxspecularv2": "PN PxSpecularV2"}
NZ_FULL = 0.7                               # rho shown as white in Nz
NXYZ_K = 19.0                               # Nxyz zenith: atan(k rho)
N_REFR = 1.5


# ---- raw pixels

def unpack_raw12(buf):
    """MIPI RAW12 as the unit writes it (3 bytes = 2 pixels, then a zero tail) -> (3000, 4096) uint16, as stored"""
    b = np.frombuffer(buf, np.uint8)[: SENSOR_W * SENSOR_H * 3 // 2].reshape(-1, 3).astype(np.uint16)
    px = np.empty(SENSOR_W * SENSOR_H, np.uint16)
    px[0::2] = (b[:, 0] << 4) | (b[:, 2] & 0x0F)
    px[1::2] = (b[:, 1] << 4) | (b[:, 2] >> 4)
    return px.reshape(SENSOR_H, SENSOR_W)


def untile(tiles):
    """the archive's JPEG-LS picture: tile (tr, tc) holds the pixels at rows r % 4 == tr, columns c % 4 == tc"""
    h, w = tiles.shape
    return tiles.reshape(4, h // 4, 4, w // 4).transpose(1, 0, 3, 2).reshape(h, w)


def decode_jls(buf):
    import imagecodecs
    return untile(np.squeeze(np.asarray(imagecodecs.jpegls_decode(buf))).astype(np.uint16))


def angles(px):
    """stored pixels -> upright 2048 x 1500 pictures I0, I45, I90, I135 (one value per polariser cell)"""
    turn = lambda a: np.ascontiguousarray(a[::-1, ::-1])
    return turn(px[1::2, 1::2]), turn(px[0::2, 1::2]), turn(px[0::2, 0::2]), turn(px[1::2, 0::2])


# ---- Stokes and the application's views

def stokes(ang):
    """the application's Stokes values at 2048 x 1500 (float32)"""
    I0, I45, I90, I135 = [cv2.cvtColor(a, BAYER).astype(np.float32).mean(axis=2) for a in ang]
    return (I0 + I45 + I90 + I135) / 2, I0 - I90, I45 - I135


def half(a):
    """mean over 2 x 2 polariser cells"""
    h, w = a.shape[:2]
    return a.reshape(h // 2, 2, w // 2, 2, *a.shape[2:]).mean(axis=(1, 3))


def degree_angle(S0, S1, S2):
    S0, S1, S2 = (np.asarray(x, np.float32) for x in (S0, S1, S2))
    return np.hypot(S1, S2) / np.maximum(S0, 1e-3), 0.5 * np.arctan2(S2, S1)


def s0_colour(ang, matrix):
    """the application's S0 picture: cell sums, demosaiced, colour matrix (4 x 3 with the offset row), sRGB"""
    cell = (ang[0].astype(np.uint32) + ang[1] + ang[2] + ang[3]).astype(np.uint16)
    lin = cv2.cvtColor(cell, BAYER).astype(np.float32) / 16380.0
    m = np.asarray(matrix, np.float32)
    v = np.clip(lin.reshape(-1, 3) @ m[:3] + m[3], 0, 1).reshape(lin.shape)
    v = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    return np.clip(v * 255 + 0.5, 0, 255).astype(np.uint8)


def _fresnel_luts(n=N_REFR):
    t = np.radians(np.linspace(0, 90, 9001))
    s2 = np.sin(t) ** 2
    rd = ((n - 1 / n) ** 2 * s2) / (2 + 2 * n * n - (n + 1 / n) ** 2 * s2 + 4 * np.cos(t) * np.sqrt(n * n - s2))
    rs = (2 * s2 * np.cos(t) * np.sqrt(n * n - s2)) / (n * n - s2 - n * n * s2 + 2 * s2 * s2)
    lo = t <= np.arctan(n)
    return (rd, t), (rs[lo], t[lo])


_DIFFUSE, _SPECULAR = _fresnel_luts()


def zenith(rho, model):
    """zenith angle (rad) from the degree: the diffuse curve saturates at 90 deg (rho 0.38 for n = 1.5), the
    specular one is taken below the Brewster angle"""
    r, t = _SPECULAR if model == "specular" else _DIFFUSE
    return np.interp(rho, r, t).astype(np.float32)


def model_vectors(rho, phi, view):
    """the physical models' unit vectors (x, y, z), z towards the camera"""
    th = zenith(rho, "specular" if view == "pn_pxspecularv2" else "diffuse")
    if view == "pn_vyzlut":
        az = np.where(phi >= 0, phi + np.pi / 2, phi - np.pi / 2)
    else:
        az = np.where(phi <= 0, phi, phi - np.pi)
    s = np.sin(th)
    return np.stack([s * np.cos(az), s * np.sin(az), np.cos(th)], axis=-1)


def nxyz_vectors(rho, phi):
    """the Nxyz view's components before the colour (absolute x and y, as the application shows them)"""
    th = np.arctan(NXYZ_K * rho)
    s = np.sin(th)
    return np.stack([s * np.cos(phi + np.pi / 4), s * np.sin(phi + np.pi / 4), np.cos(th)], axis=-1)


def view(name, P):
    """one of VIEWS from the products P (S0, S1, S2 and, for "s0", rgb) -> uint8 RGB picture"""
    if name == "s0":
        return P["rgb"]
    rho, phi = degree_angle(P["S0"], P["S1"], P["S2"])
    if name == "nz":
        g = np.clip(rho / NZ_FULL * 255 + 0.5, 0, 255).astype(np.uint8)
        return np.dstack([g, g, g])
    if name == "nxy":
        u = np.clip((np.degrees(phi) + 90) / 180 * 255 + 0.5, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(u, cv2.COLORMAP_JET)[..., ::-1]
    if name == "nxyz":
        v = nxyz_vectors(rho, phi)
        v[..., :2] = np.abs(v[..., :2])
        return np.clip(v * 255 + 0.5, 0, 255).astype(np.uint8)
    if name.startswith("pn_"):
        return np.clip((model_vectors(rho, phi, name) + 1) / 2 * 255 + 0.5, 0, 255).astype(np.uint8)
    raise KeyError(name)


def pixel_values(P, x, y):
    """the values behind every view at one picture pixel (x, y) of the products' grid"""
    S0, S1, S2 = (float(P[k][y, x]) for k in ("S0", "S1", "S2"))
    rho, phi = (float(v) for v in degree_angle(S0, S1, S2))
    out = dict(S0=S0, rho=rho, phi_deg=float(np.degrees(phi)), nxyz=nxyz_vectors(np.float32(rho), np.float32(phi)).tolist())
    for v in ("pn_vyzlut", "pn_pxdiffuse", "pn_pxspecularv2"):
        out[v] = model_vectors(np.float32(rho), np.float32(phi), v).tolist()
    return out


def products(px, matrix=None, full=False):
    """stored raw pixels -> {S0, S1, S2 (float16), rgb (uint8, when the colour matrix is given)} at 1024 x 750 (or
    2048 x 1500 with full=True)"""
    ang = angles(px)
    S = stokes(ang)
    P = {k: (v if full else half(v)).astype(np.float16) for k, v in zip(("S0", "S1", "S2"), S)}
    if matrix is not None:
        rgb = s0_colour(ang, matrix)
        P["rgb"] = rgb if full else cv2.resize(rgb, (CELL_W // 2, CELL_H // 2), interpolation=cv2.INTER_AREA)
    return P


# ---- the raw frames of a session

class RawStore:
    """The session's raw frames by video frame number (FRAMES.tsv order): the layer archives (one .jls per frame, read
    in place) and loose .raw files. An archive still downloading is indexed up to its last complete frame."""

    def __init__(self, raw_dirs, frames_tsv):
        from .session import load_frame_times
        names = load_frame_times(frames_tsv).frame.values
        self.index_of = {os.path.splitext(n)[0]: k for k, n in enumerate(names)}
        self.n_frames = len(names)
        self.loc = {}
        self.dirs = [raw_dirs] if isinstance(raw_dirs, str) else list(raw_dirs)
        self.scan()

    def scan(self):
        self.loc = {}
        for d in self.dirs:
            for p in sorted(glob.glob(os.path.join(d, "*.tar")) + glob.glob(os.path.join(d, "*.tar.*.partial"))):
                self._index_tar(p)
            for p in sorted(glob.glob(os.path.join(d, "*.raw"))):
                k = self.index_of.get(os.path.splitext(os.path.basename(p))[0])
                if k is not None:
                    self.loc[k] = (p, 0, os.path.getsize(p), "raw")
        return self

    def _index_tar(self, path):
        size = os.path.getsize(path)
        try:
            with open(path, "rb") as fh, tarfile.open(fileobj=fh, mode="r:") as tf:
                for m in tf:
                    if m.isfile() and m.name.endswith(".jls") and m.offset_data + m.size <= size:
                        k = self.index_of.get(os.path.basename(m.name)[:-4])
                        if k is not None:
                            self.loc[k] = (path, m.offset_data, m.size, "jls")
        except (tarfile.TarError, EOFError, OSError):
            pass                                                     # a partial archive ends inside a member

    @property
    def frames(self):
        return sorted(self.loc)

    def __contains__(self, k):
        return k in self.loc

    def __len__(self):
        return len(self.loc)

    def job(self, k):
        return self.loc[k]

    def pixels(self, k):
        return read_pixels(self.loc[k])


def read_pixels(loc):
    path, off, size, kind = loc
    with open(path, "rb") as fh:
        fh.seek(off)
        buf = fh.read(size)
    return unpack_raw12(buf) if kind == "raw" else decode_jls(buf)


def frame_products(job):
    """worker: (location, colour matrix[, full resolution]) -> products (for a process pool)"""
    loc, matrix = job[:2]
    return products(read_pixels(loc), matrix, full=len(job) > 2 and job[2])


def load_matrix(path):
    return None if not path or not os.path.exists(path) else json.load(open(path))["matrix"]


class Prefetcher:
    """products for frames requested in a known order, decoded ahead by worker processes (the video export)"""

    def __init__(self, store, matrix, frames, workers=4, ahead=16, full=False):
        from concurrent.futures import ProcessPoolExecutor
        self.store, self.matrix, self.frames, self.ahead = store, matrix, list(frames), ahead
        self.full = full
        self.pos = {k: n for n, k in enumerate(self.frames)}
        self.ex = ProcessPoolExecutor(workers)
        self.fut, self.next = {}, 0

    def get(self, k):
        upto = self.pos[k] + self.ahead + 1
        while self.next < min(upto, len(self.frames)):
            j = self.frames[self.next]
            if j not in self.fut:
                self.fut[j] = self.ex.submit(frame_products, (self.store.job(j), self.matrix, self.full))
            self.next += 1
        if k not in self.fut:
            self.fut[k] = self.ex.submit(frame_products, (self.store.job(k), self.matrix, self.full))
        return self.fut.pop(k).result()

    def close(self):
        self.ex.shutdown(cancel_futures=True)


# ---- the check against the application's own decodes

def _ref(frame_dir, name):
    f = glob.glob(os.path.join(frame_dir, f"*_{name}_2048x1500.png"))
    return None if not f else cv2.imread(f[0], cv2.IMREAD_COLOR)[..., ::-1]


def fit_s0_colour(frame_dir):
    """colour matrix (4 x 3, offset row last) from the frame's raw to the application's S0 picture, fitted in linear
    light on the pixels that clip in neither"""
    px = unpack_raw12(open(glob.glob(os.path.join(frame_dir, "*.raw"))[0], "rb").read())
    ang = angles(px)
    cell = (ang[0].astype(np.uint32) + ang[1] + ang[2] + ang[3]).astype(np.uint16)
    lin = cv2.cvtColor(cell, BAYER).astype(np.float32) / 16380.0
    T = _ref(frame_dir, "s0_app").astype(np.float32) / 255
    T_lin = np.where(T <= 0.04045, T / 12.92, ((T + 0.055) / 1.055) ** 2.4)
    ok = (T.max(axis=2) < 0.97) & (T.min(axis=2) > 0.02) & (lin.max(axis=2) < 0.95)
    idx = np.nonzero(ok.ravel())[0][::5]
    X = np.c_[lin.reshape(-1, 3)[idx], np.ones(len(idx))]
    M, *_ = np.linalg.lstsq(X, T_lin.reshape(-1, 3)[idx], rcond=None)
    return M


def verify(frame_dir, matrix):
    """our views of the reference frame at full resolution against the application's -> {view: rms levels}, and a
    contact sheet (ours above, the application's below)"""
    px = unpack_raw12(open(glob.glob(os.path.join(frame_dir, "*.raw"))[0], "rb").read())
    P = products(px, matrix, full=True)
    clip = np.maximum.reduce(angles(px)) >= 4000                     # clipped polariser pixels: no valid Stokes values
    res, tiles = {}, []
    names = [("s0", "s0_app")] + [(v, v) for v in VIEWS if v != "s0"]
    for v, refname in names:
        ref = _ref(frame_dir, refname)
        if ref is None or (v == "s0" and matrix is None):
            continue
        ours = view(v, P)
        e = ours.astype(np.float32) - ref
        res[VIEWS[v]] = dict(rms=round(float(np.sqrt(np.mean(e ** 2))), 2),
                             rms_unclipped=round(float(np.sqrt(np.mean(e[~clip] ** 2))), 2),
                             median_abs=float(np.median(np.abs(e[~clip]))))
        small = lambda a: cv2.resize(a, (512, 375), interpolation=cv2.INTER_AREA)
        col = np.vstack([small(ours), small(ref)])
        cv2.putText(col, VIEWS[v], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(col)
    sheet = np.hstack(tiles)[..., ::-1] if tiles else None
    return res, sheet
