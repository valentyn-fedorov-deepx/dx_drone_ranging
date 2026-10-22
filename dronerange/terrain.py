"""Terrain over the flight area from the USGS 3DEP elevation point service (EPQS, US only, about 1 m lidar where
available), the flat-ground error of the ranging, and a terrain model for terrain-aware ranging.

Flat-ground error: the ray meets the plane through the ground under the drone; an object standing dz higher than
that ground is really at D (1 - dz / h) (to first order).
"""
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import RegularGridInterpolator

EPQS = "https://epqs.nationalmap.gov/v1/json?x={lon:.7f}&y={lat:.7f}&units=Meters&wkid=4326&includeDate=false"
R_EARTH = 6371008.8


def query(lat, lon, tries=3):
    for _ in range(tries):
        try:
            return float(json.loads(urllib.request.urlopen(EPQS.format(lat=lat, lon=lon), timeout=20).read())["value"])
        except Exception:
            pass
    return np.nan


def en_to_latlon(e, n, lat0, lon0):
    lat = lat0 + math.degrees(n / R_EARTH)
    return lat, lon0 + math.degrees(e / (R_EARTH * math.cos(math.radians(lat0))))


def fetch_grid(lat0, lon0, e0, e1, n0, n1, step=10.0, workers=8):
    ee = np.arange(e0, e1 + step, step); nn = np.arange(n0, n1 + step, step)
    pts = [en_to_latlon(e, n, lat0, lon0) for n in nn for e in ee]
    with ThreadPoolExecutor(workers) as ex:
        z = list(ex.map(lambda p: query(*p), pts))
    return ee, nn, np.array(z).reshape(len(nn), len(ee))


class Terrain:
    """z_ref: the ground height the drone's height is measured from (the take-off point for barometric heights);
    None = the ground straight under the drone (laser altimeter)."""

    def __init__(self, E, N, Z, z_ref=None):
        self.E, self.N, self.Z, self.z_ref = E, N, Z, z_ref
        ok = np.isfinite(Z)
        self._I = RegularGridInterpolator((N, E), np.where(ok, Z, np.nanmedian(Z)), bounds_error=False, fill_value=None)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["E"], d["N"], d["Z"])

    def save(self, path):
        np.savez(path, E=self.E, N=self.N, Z=self.Z)

    def height_at(self, e, n):
        return self._I(np.c_[np.atleast_1d(n), np.atleast_1d(e)])


    def ground_ref(self, e, n):
        return self.z_ref if self.z_ref is not None else float(self.height_at(np.array([e]), np.array([n]))[0])


def flat_ground_error(R, terrain):
    R = R.copy()
    R["z_obj"] = terrain.height_at(R.E.values, R.N.values)
    zr = terrain.z_ref if terrain.z_ref is not None else terrain.height_at(R.E_drone.values, R.N_drone.values)
    R["dz"] = R.z_obj - zr
    R["flat_err_pct"] = 100 * R.dz / R.h                  # ground higher than the reference -> true distance shorter by dz/h
    return R
