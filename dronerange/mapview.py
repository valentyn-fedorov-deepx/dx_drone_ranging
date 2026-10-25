"""Map panel: satellite tiles (Web Mercator, cached), north-up, with the drone, its track, the camera footprint and the
tracked objects by id.

Web Mercator is conformal, so over a few hundred metres the tile pixels are a uniform grid of m_per_px =
156543.034 cos(lat) / 2^zoom metres in both axes, and ENU metres map to mosaic pixels by a scale and an offset.
"""
import math
import time
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def latlon_to_global_px(lat, lon, z):
    n = 256.0 * 2 ** z
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * n
    return x, y


class TileMosaic:
    def __init__(self, img, x0, y0, z, lat0, lon0):
        self.img, self.x0, self.y0, self.z = img, x0, y0, z      # global pixel of img[0, 0]
        gx, gy = latlon_to_global_px(lat0, lon0, z)
        self.ox, self.oy = gx - x0, gy - y0                      # mosaic pixel of the ENU origin
        self.mpp = 156543.03392 * math.cos(math.radians(lat0)) / 2 ** z

    def en_to_px(self, e, n):
        return self.ox + np.asarray(e) / self.mpp, self.oy - np.asarray(n) / self.mpp

    @classmethod
    def build(cls, lat_min, lat_max, lon_min, lon_max, z, cache_dir, lat0, lon0, url=ESRI, log=print):
        x0g, y1g = latlon_to_global_px(lat_min, lon_min, z)
        x1g, y0g = latlon_to_global_px(lat_max, lon_max, z)
        tx0, tx1 = int(x0g // 256), int(x1g // 256)
        ty0, ty1 = int(y0g // 256), int(y1g // 256)
        cache = Path(cache_dir) / str(z)
        cache.mkdir(parents=True, exist_ok=True)
        jobs = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

        def get(job):
            tx, ty = job
            p = cache / f"{tx}_{ty}.jpg"
            im = cv2.imread(str(p)) if p.exists() else None
            if im is None:                                          # missing or broken: fetch, write atomically, retry
                req = urllib.request.Request(url.format(z=z, x=tx, y=ty), headers={"User-Agent": "dx_ranging/0.2 (drone demo map)"})
                for k in range(4):
                    try:
                        tmp = p.with_name(f"{p.stem}.{os.getpid()}.part")
                        tmp.write_bytes(urllib.request.urlopen(req, timeout=20).read())
                        os.replace(tmp, p)
                        im = cv2.imread(str(p))
                        if im is not None:
                            break
                    except Exception:
                        time.sleep(1.0 + k)
            return job, im

        n_new = sum(not (cache / f"{tx}_{ty}.jpg").exists() for tx, ty in jobs)
        with ThreadPoolExecutor(6) as ex:
            tiles = dict(ex.map(get, jobs))
        img = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), 40, np.uint8)
        miss = 0
        for (tx, ty), im in tiles.items():
            if im is None or im.shape[:2] != (256, 256):
                miss += 1
                continue
            img[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = im
        log(f"map tiles z{z}: {len(jobs)} ({n_new} downloaded, {miss} missing), mosaic {img.shape[1]} x {img.shape[0]} px")
        return cls(img, tx0 * 256, ty0 * 256, z, lat0, lon0)


def mosaic_for_session(sess, extra_en, cfg_map, cache_dir, log=print):
    """tiles covering the drone track and the given ENU points, plus a margin"""
    E = np.r_[sess.E_s, extra_en[0]]; N = np.r_[sess.N_s, extra_en[1]]
    m = cfg_map.get("margin_m", 150.0)
    lat_a, lon_a = sess.en_to_latlon(np.nanmin(E) - m, np.nanmin(N) - m)
    lat_b, lon_b = sess.en_to_latlon(np.nanmax(E) + m, np.nanmax(N) + m)
    return TileMosaic.build(lat_a, lat_b, lon_a, lon_b, cfg_map.get("zoom", 19), cache_dir, sess.lat0, sess.lon0,
                            cfg_map.get("tiles_url", ESRI), log)


class MapView:
    """north-up panel of size (W, H) at m_per_px, centred on an ENU point"""

    def __init__(self, mosaic, W, H, m_per_px, dim=0.82):
        self.mo, self.W, self.H, self.mpp = mosaic, W, H, m_per_px
        s = mosaic.mpp / m_per_px
        self.base = cv2.resize(mosaic.img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self.base = (self.base.astype(np.float32) * dim).astype(np.uint8)
        self.s = s

    def render(self, ce, cn):
        """background crop for the centre (ce, cn); returns BGR image and the ENU -> panel transform"""
        mx, my = self.mo.en_to_px(ce, cn)
        bx, by = mx * self.s, my * self.s
        M = np.float32([[1, 0, self.W / 2 - bx], [0, 1, self.H / 2 - by]])
        img = cv2.warpAffine(self.base, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=(30, 30, 30))

        def to_panel(e, n):
            return self.W / 2 + (np.asarray(e) - ce) / self.mpp, self.H / 2 - (np.asarray(n) - cn) / self.mpp
        return img, to_panel
