"""Synthetic checks of the raw-frame reading and the polarisation views. Run: python -m pytest tests"""
import io
import os
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dronerange import polar  # noqa: E402

ANGLE_AT = {(0, 0): 90, (0, 1): 45, (1, 0): 135, (1, 1): 0}     # polariser angle inside a 2 x 2 cell, as stored
COLOUR_AT = {(0, 0): "R", (0, 1): "G", (1, 0): "G", (1, 1): "B"}  # colour of the 2 x 2 cells inside the 4 x 4 cell


def pack_raw12(px):
    p = px.ravel().astype(np.uint16)
    a, b = p[0::2], p[1::2]
    out = np.c_[a >> 4, b >> 4, ((b & 15) << 4) | (a & 15)].astype(np.uint8).ravel()
    return out.tobytes() + bytes(770048)


def mosaic(h, w, s0, rho, phi_deg, gains=(0.7, 1.0, 0.6)):
    """stored sensor picture of a scene with one degree and angle of polarisation (Malus: I = S0 / 2 (1 + rho cos(2 theta
    - 2 phi))), per colour intensity S0 * gain"""
    g = dict(zip("RGB", gains))
    px = np.zeros((h, w))
    for (r, c), th in ANGLE_AT.items():
        for (R, C), col in COLOUR_AT.items():
            sl = (slice(2 * R + r, None, 4), slice(2 * C + c, None, 4))
            px[sl] = s0 * g[col] / 2 * (1 + rho * np.cos(np.radians(2 * th - 2 * phi_deg)))
    return np.round(px).astype(np.uint16)


def test_raw12_roundtrip():
    rng = np.random.default_rng(0)
    px = rng.integers(0, 4096, (polar.SENSOR_H, polar.SENSOR_W), dtype=np.uint16)
    assert np.array_equal(polar.unpack_raw12(pack_raw12(px)), px)


def test_untile_inverts_the_archive_tiles():
    rng = np.random.default_rng(1)
    px = rng.integers(0, 4096, (24, 32), dtype=np.uint16)
    th, tw = 6, 8
    tiles = np.zeros_like(px)
    for tr in range(4):
        for tc in range(4):
            tiles[tr * th:(tr + 1) * th, tc * tw:(tc + 1) * tw] = px[tr::4, tc::4]
    assert np.array_equal(polar.untile(tiles), px)


@pytest.mark.parametrize("rho,phi", [(0.05, 30.0), (0.2, -60.0), (0.4, 85.0)])
def test_stokes_recover_degree_and_angle(rho, phi):
    px = mosaic(64, 96, 3000, rho, phi)
    P = polar.products(px, full=True)
    r, a = polar.degree_angle(P["S0"], P["S1"], P["S2"])
    inner = (slice(4, -4), slice(4, -4))
    assert abs(np.median(r[inner]) - rho) < 0.004
    d = (np.degrees(a[inner]) - phi + 90) % 180 - 90
    assert np.abs(d).max() < 1.0
    rgb = polar.s0_colour(polar.angles(px), np.r_[np.eye(3), np.zeros((1, 3))])
    assert rgb[inner][..., 1].mean() > rgb[inner][..., 0].mean() > rgb[inner][..., 2].mean()   # red and blue in place


def test_views_follow_the_application():
    P = {"S0": np.full((2, 2), 1000.0), "S1": np.zeros((2, 2)), "S2": np.zeros((2, 2))}
    P["S1"][:] = 1000.0 * 0.35                                       # rho 0.35, phi 0
    nz = polar.view("nz", P)
    assert (nz == 128).all()                                         # rho / 0.7
    nxy = polar.view("nxy", P)
    assert nxy[0, 0, 1] > 200 and nxy[0, 0, 0] < 160                 # jet in the middle: green
    nxyz = polar.view("nxyz", P)
    t = np.arctan(polar.NXYZ_K * 0.35)
    assert abs(int(nxyz[0, 0, 2]) - round(255 * np.cos(t))) <= 1
    assert abs(int(nxyz[0, 0, 0]) - int(nxyz[0, 0, 1])) <= 1         # |cos 45| = |sin 45|
    for v in ("pn_vyzlut", "pn_pxdiffuse", "pn_pxspecularv2"):
        assert polar.view(v, P)[0, 0, 2] >= 128                      # z towards the camera


def test_fresnel_inversion():
    n = polar.N_REFR
    for deg in (10, 30, 50, 70, 85):
        t = np.radians(deg)
        s2 = np.sin(t) ** 2
        rd = ((n - 1 / n) ** 2 * s2) / (2 + 2 * n * n - (n + 1 / n) ** 2 * s2 + 4 * np.cos(t) * np.sqrt(n * n - s2))
        assert abs(np.degrees(polar.zenith(np.float32(rd), "diffuse")) - deg) < 0.05
        if deg < 56:
            rs = (2 * s2 * np.cos(t) * np.sqrt(n * n - s2)) / (n * n - s2 - n * n * s2 + 2 * s2 * s2)
            assert abs(np.degrees(polar.zenith(np.float32(rs), "specular")) - deg) < 0.05
    assert abs(np.degrees(polar.zenith(np.float32(0.6), "diffuse")) - 90) < 1e-3     # beyond the diffuse maximum


def test_raw_store_reads_archives_and_partial_ones(tmp_path):
    imagecodecs = pytest.importorskip("imagecodecs")
    names = [f"2026_10_23_18_15_{55 + k:02d}_000000_et_1_iso_100" for k in range(3)]
    pd.DataFrame({"frame": [n + ".raw" for n in names]}).to_csv(tmp_path / "FRAMES.tsv", sep="\t", index=False)
    rng = np.random.default_rng(2)
    pix = [rng.integers(0, 4096, (24, 32), dtype=np.uint16) for _ in names]

    def tiles(px):
        th, tw = px.shape[0] // 4, px.shape[1] // 4
        t = np.zeros_like(px)
        for tr in range(4):
            for tc in range(4):
                t[tr * th:(tr + 1) * th, tc * tw:(tc + 1) * tw] = px[tr::4, tc::4]
        return t

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for n, px in zip(names, pix):
            data = imagecodecs.jpegls_encode(tiles(px))
            ti = tarfile.TarInfo(f"seq/{n}.jls"); ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    whole = buf.getvalue()
    (tmp_path / "a.tar").write_bytes(whole)
    st = polar.RawStore([str(tmp_path)], str(tmp_path / "FRAMES.tsv"))
    assert st.frames == [0, 1, 2]
    assert np.array_equal(st.pixels(1), pix[1])
    os.remove(tmp_path / "a.tar")
    last = st.loc[2]
    (tmp_path / "a.tar.0123abcd.partial").write_bytes(whole[: last[1] + last[2] // 2])     # cut inside the last frame
    st = polar.RawStore([str(tmp_path)], str(tmp_path / "FRAMES.tsv"))
    assert st.frames == [0, 1]
