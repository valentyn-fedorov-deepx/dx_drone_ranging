"""Presentation video (`demo`): a title, sections of the flight and a closing card, 1920 x 1080 at 30 fps.

The camera view fills the height (the stabilised view, the camera's aspect); a side panel carries the drone's
altitude, the camera's tilt, the number of objects detected, a satellite map with the tracks, the nearest objects and
the altitude profile of the whole flight with the playhead. Labels keep the VyzAI fields (CLASS id, Xo / Yo / Zo,
Range, Velocity with the direction arrow), sit next to their object with a thin leader to it, and show what the
system has in each frame only. The recording is 5 fps and stays so: every camera frame is held for its 0.2 s (a
timelapse shows every n-th frame); only the titles fade and the views of one frame cross-fade. The sections and the
texts come from the config (`demo:`); with `captions: false` the video carries no explanatory text (no section
titles, no view names or colour scales), only the system's data.
"""
import datetime
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import polar
from .hud import NAMES, label_pointer, place_labels, nice_rmax
from .mapview import MapView
from .ranging import footprint
from .stabilise import angles_from_R

W, H = 1920, 1080
CAM_W, CAM_H = 1474, 1080                    # the camera's aspect at full height
PX, PW = CAM_W, W - CAM_W                    # side panel
SS = 2                                       # overlays are drawn at twice the size and scaled down (smooth edges)
FPS = 30
BG = (12, 15, 20); PANEL = (17, 21, 28); LINE = (44, 52, 64); TEXT = (234, 238, 243); DIM = (139, 150, 165)
ACCENT = (64, 210, 240)
CLASS_COL = {"person": (88, 224, 140), "vehicle": (252, 192, 46), "bicycle": (110, 200, 255)}
VIEW_TEXT = {"rgb": ("RGB", "the colour video"),
             "s0": ("S0", "total intensity, in the unit's calibrated colour"),
             "nz": ("Nz", "how far each surface turns away from the camera"),
             "nxy": ("Nxy", "which way each surface faces in the image"),
             "nxyz": ("Nxyz", "surface orientation x, y, z as colour"),
             "pn_vyzlut": ("Physical data: VyzLut", "Fresnel model for dielectric surfaces"),
             "pn_pxdiffuse": ("Physical data: PxDiffuse", "model for matte surfaces"),
             "pn_pxspecularv2": ("Physical data: PxSpecular", "model for shiny surfaces")}


def _font(names, size):
    for n in names:
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu"):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, int(size * SS))
    import matplotlib
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), int(size * SS))


class Fonts:
    def __init__(self):
        num, txt, semi, light = ["bahnschrift.ttf", "DejaVuSans.ttf"], ["segoeui.ttf", "DejaVuSans.ttf"], ["seguisb.ttf", "DejaVuSans-Bold.ttf"], ["segoeuil.ttf", "DejaVuSans.ttf"]
        self.num = {s: _font(num, s) for s in (13, 14, 15, 30, 48, 84)}
        self.txt = {s: _font(txt, s) for s in (11, 12, 13, 15, 20)}
        self.semi = {s: _font(semi, s) for s in (12, 14, 15, 16, 22, 34)}
        self.light = {s: _font(light, s) for s in (28, 30)}


def S(*v):
    return tuple(int(round(x * SS)) for x in v)


def spaced(d, xy, text, font, fill, track=1.5):
    """letter-spaced capitals"""
    x, y = xy
    for ch in text:
        d.text((x * SS, y * SS), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) / SS + track
    return x


def arrow(d, x, y, ang_deg, length, color, width=2):
    """arrow centred at (x, y), angle clockwise from up (overlay coordinates / SS)"""
    a = math.radians(ang_deg)
    dx, dy = math.sin(a), -math.cos(a)
    x0, y0, x1, y1 = x - dx * length / 2, y - dy * length / 2, x + dx * length / 2, y + dy * length / 2
    d.line(S(x0, y0, x1, y1), fill=color, width=int(width * SS))
    for s in (-1, 1):
        b = a + math.pi + s * 0.55
        d.line(S(x1, y1, x1 + math.sin(b) * length * 0.45, y1 - math.cos(b) * length * 0.45), fill=color, width=int(width * SS))


class Demo:
    def __init__(self, hr, stab, store=None, matrix=None, cfg=None, log=print):
        """hr: HudRenderer (the frame's rows and detections); stab: Stabiliser with out_size (CAM_W, CAM_H)"""
        self.hr, self.stab, self.store, self.matrix = hr, stab, store, matrix
        self.sess, self.track = hr.sess, hr.track
        self.cfg = cfg or {}
        self.captions = self.cfg.get("captions", True)
        self.log = log
        self.f = Fonts()
        self.mapviews = {}
        self.t0 = self.sess.ft[0]
        self.last_offset = {}
        hh = self.track.h.values
        self.profile = (self.sess.ft - self.t0, hh)
        self._vid = None

    # ---- the camera picture
    def picture(self, i, view, P=None):
        if view == "rgb":
            if self._vid is None:
                self._vid = self.sess.video()
            im = self._vid.read(i)
        else:
            im = cv2.cvtColor(polar.view(view, P), cv2.COLOR_RGB2BGR)
        return self.stab.warp(im, i)

    def shown(self, i):
        """(row, box in the camera view) of the detections of frame i at least a quarter inside the view"""
        out = []
        for r in self.hr.rows(i):
            b = self.hr.label_box(i, r)
            if b is None:
                continue
            q = self.stab.to_view(i, [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]])
            x1, y1, x2, y2 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
            c = (max(x1, 0), max(y1, 0), min(x2, CAM_W), min(y2, CAM_H))
            if c[2] > c[0] and c[3] > c[1] and (c[2] - c[0]) * (c[3] - c[1]) >= 0.25 * max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6):
                out.append((r, c))
        return out

    # ---- one composed frame (without the fading titles)
    def frame(self, i, view="rgb", P=None, section=None):
        t = self.sess.ft[i]
        h = float(self.track.h.values[i])
        base = np.empty((H, W, 3), np.uint8)
        base[:] = PANEL
        base[:, :CAM_W] = cv2.cvtColor(self.picture(i, view, P), cv2.COLOR_BGR2RGB)
        base[:, CAM_W:CAM_W + 1] = LINE
        shown = self.shown(i)
        ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        self._labels(d, i, shown)
        self._timecode(d, t, view)
        mp = self._panel(d, i, t, h, shown, view, section)
        img = Image.fromarray(base).convert("RGBA")
        if mp is not None:
            img.paste(mp[0], mp[1], mp[2])
        img = Image.alpha_composite(img, ov.resize((W, H), Image.LANCZOS))
        return np.asarray(img.convert("RGB"))

    def _labels(self, d, i, shown):
        f = self.f
        lw, rh, top = 160, 18, 30
        items, info = [], []
        order = sorted(shown, key=lambda rb: (rb[0].grp != "person", not rb[0].moving, rb[0].range))
        for r, box in order:
            unconf = int(r.id) < 0
            name = NAMES.get(r.cls, r.cls.upper())
            head = name if unconf else f"{name} {int(r.id)}"
            still = not r.moving
            vel = "-" if unconf else f"{0 if still else round(r.speed_kmh):.0f} km/h"
            rows = [("Xo", f"{r.Xo:.2f} m"), ("Yo", f"{r.Yo:.2f} m"), ("Zo", f"{r.Zo:.2f} m"), ("Range", f"{r.range:.2f} m"), ("Velocity", vel)]
            items.append((box, lw, top + rh * len(rows) + 8))
            info.append((r, head, rows, (not still) and not unconf))
        prefer = [self.last_offset.get(int(r.id)) if int(r.id) >= 0 else None for r, _, _, _ in info]
        for (box, lw_, lh), (lx, ly), (r, head, rows, moving) in zip(items, place_labels(items, CAM_W, CAM_H, top=10, prefer=prefer, gap=16), info):
            col = CLASS_COL.get(r.grp, (200, 200, 200))
            x1, y1, x2, y2 = box
            if int(r.id) >= 0:
                self.last_offset[int(r.id)] = (lx - 0.5 * (x1 + x2), ly - 0.5 * (y1 + y2))
            unconf = int(r.id) < 0
            d.rounded_rectangle(S(lx, ly, lx + lw_, ly + lh), radius=7 * SS, fill=(10, 13, 18, 205), outline=(255, 255, 255, 34), width=SS)
            tri, lead = label_pointer(box, (lx, ly, lx + lw_, ly + lh), size=12, half=8, margin=14)   # the triangle points at the object
            if lead is not None:
                d.line(S(*lead[0], *lead[1]), fill=col + (210,), width=int(1.4 * SS))
            if tri is not None:
                d.polygon([S(*v) for v in tri], fill=col + ((170,) if unconf else (245,)))
            d.rounded_rectangle(S(lx, ly, lx + 4, ly + lh), radius=2 * SS, fill=col + ((150,) if unconf else (255,)))
            d.text(S(lx + 13, ly + 6), head, font=f.semi[15], fill=TEXT + (255,))
            for k, (nm, val) in enumerate(rows):
                yy = ly + 29 + 18 * k
                d.text(S(lx + 13, yy), nm, font=f.txt[12], fill=DIM + (255,))
                right = lx + lw_ - (26 if (moving and nm == "Velocity") else 10)
                d.text((S(right)[0] - d.textlength(val, font=f.num[14]), S(0, yy - 1)[1]), val, font=f.num[14], fill=TEXT + (255,))
                if moving and nm == "Velocity":
                    arrow(d, lx + lw_ - 15, yy + 8, r.rel_dir, 11, col + (255,), 1.8)

    def _timecode(self, d, t, view):
        f = self.f
        s = datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S UTC")
        s2 = f"T+{t - self.t0:5.1f} s"
        w = 22 + (d.textlength(s, font=f.num[14]) + d.textlength(s2, font=f.num[14])) / SS + 18
        d.rounded_rectangle(S(18, 18, 18 + w, 46), radius=14 * SS, fill=(10, 13, 18, 190))
        d.ellipse(S(30, 28, 38, 36), fill=(239, 68, 68, 255))
        x = 46
        d.text(S(x, 22), s, font=f.num[14], fill=TEXT + (255,))
        x += d.textlength(s, font=f.num[14]) / SS + 12
        d.text(S(x, 22), s2, font=f.num[14], fill=DIM + (255,))
        if view != "rgb" and self.captions:
            name = VIEW_TEXT[view][0]
            w = d.textlength(name, font=f.semi[16]) / SS + 30
            d.rounded_rectangle(S(CAM_W - 18 - w, 18, CAM_W - 18, 48), radius=15 * SS, fill=(10, 13, 18, 200), outline=ACCENT + (180,), width=SS)
            d.text(S(CAM_W - 18 - w + 15, 22), name, font=f.semi[16], fill=ACCENT + (255,))

    def _panel(self, d, i, t, h, shown, view, section):
        f = self.f
        x0, pad = PX, 28
        xl, xr = x0 + pad, W - pad
        y = 14
        if self.captions:                                            # the drone's name heads the panel only with captions
            spaced(d, (xl, 26), self.hr.drone_name.upper(), f.num[30], TEXT + (255,), 2.5)
            y = 70
        if self.cfg.get("subtitle"):
            d.text(S(xl, 66), self.cfg["subtitle"], font=f.txt[13], fill=DIM + (255,))
            y = 96
        if section and self.captions:
            spaced(d, (xl, y + 4), section.upper(), f.semi[12], ACCENT + (255,), 1.2)
            y += 30
        y += 12
        # stats (the tilt as measured: the real camera, not the stabilised view)
        el = angles_from_R(self.sess.R_wc(t)[None])[1][0]
        stats = [("ALTITUDE", f"{h:.1f}", "m"), ("CAMERA TILT", f"{-math.degrees(el):.1f}", "deg"), ("DETECTED", f"{len(shown)}", "")]
        cw = (xr - xl) / 3
        for k, (lab, val, unit) in enumerate(stats):
            cx = xl + k * cw
            spaced(d, (cx, y), lab, f.txt[11], DIM + (255,), 1.0)
            d.text(S(cx, y + 16), val, font=f.num[30], fill=TEXT + (255,))
            if unit:
                d.text(S(cx + d.textlength(val, font=f.num[30]) / SS + 5, y + 32), unit, font=f.txt[13], fill=DIM + (255,))
        y += 72
        d.line(S(xl, y, xr, y), fill=LINE + (255,), width=SS)
        # satellite map: as tall as the objects list, the view block and the altitude profile leave
        my0 = y + 18
        mh = min(430, 930 - my0 - 24 - 160 - (120 if self.captions else 0))   # no taller than the satellite tiles reach
        mw = xr - xl
        mp = self._map(d, i, t, h, shown, (xl, my0, mw, mh))
        # nearest objects
        y = my0 + mh + 24
        spaced(d, (xl, y), "NEAREST OBJECTS", f.txt[11], DIM + (255,), 1.0)
        n_rows = max(5, int(((820 if self.captions else 860) - (y + 22)) // 27))   # as many as the room below the map holds
        rows = sorted([r for r, _ in shown], key=lambda r: r.range)[:n_rows]
        for k, r in enumerate(rows):
            yy = y + 22 + k * 27
            col = CLASS_COL.get(r.grp, (200, 200, 200))
            d.ellipse(S(xl, yy + 6, xl + 9, yy + 15), fill=col + (255,))
            name = NAMES.get(r.cls, r.cls.upper()) + ("" if int(r.id) < 0 else f" {int(r.id)}")
            d.text(S(xl + 18, yy), name, font=f.semi[15], fill=TEXT + (255,))
            rng = f"{r.range:.1f} m"
            d.text(S(xl + 190, yy + 1), rng, font=f.num[15], fill=TEXT + (255,))
            if int(r.id) < 0:
                sp = "-"
            else:
                sp = f"{0 if not r.moving else round(r.speed_kmh):.0f} km/h"
            d.text((S(xr - (22 if r.moving and int(r.id) >= 0 else 0))[0] - d.textlength(sp, font=f.num[15]), S(0, yy + 1)[1]), sp, font=f.num[15], fill=DIM + (255,))
            if r.moving and int(r.id) >= 0:
                arrow(d, xr - 8, yy + 10, r.rel_dir, 12, col + (255,), 1.8)
        if not rows:
            d.text(S(xl, y + 22), "none in this frame", font=f.txt[13], fill=DIM + (255,))
        # view
        if self.captions:
            y = 820
            d.line(S(xl, y - 14, xr, y - 14), fill=LINE + (255,), width=SS)
            spaced(d, (xl, y), "VIEW", f.txt[11], DIM + (255,), 1.0)
            name, what = VIEW_TEXT[view]
            d.text(S(xl, y + 16), name, font=f.semi[16], fill=(ACCENT if view != "rgb" else TEXT) + (255,))
            d.text(S(xl, y + 40), what, font=f.txt[12], fill=DIM + (255,))
            self._legend(d, view, xl, y + 62, xr - xl)
        # altitude profile of the flight
        if self.captions:
            self._profile(d, t - self.t0, xl, 966, xr - xl, 74)
        else:
            self._profile(d, t - self.t0, xl, 904, xr - xl, 132)
        return mp

    def _legend(self, d, view, x, y, w):
        if view == "nz":
            for k in range(int(w)):
                v = int(255 * k / w)
                d.line(S(x + k, y, x + k, y + 8), fill=(v, v, v, 255), width=SS)
            d.text(S(x, y + 11), "0", font=self.f.txt[11], fill=DIM + (255,))
            d.text((S(x + w)[0] - d.textlength("0.7", font=self.f.txt[11]), S(0, y + 11)[1]), "0.7", font=self.f.txt[11], fill=DIM + (255,))
        elif view == "nxy":
            jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], cv2.COLORMAP_JET)[0, :, ::-1]
            for k in range(int(w)):
                c = jet[int(255 * k / max(w - 1, 1))]
                d.line(S(x + k, y, x + k, y + 8), fill=(int(c[0]), int(c[1]), int(c[2]), 255), width=SS)
            for frac, s in ((0, "-90 deg"), (0.5, "0"), (1, "+90 deg")):
                tw = d.textlength(s, font=self.f.txt[11]) / SS
                d.text(S(x + frac * w - frac * tw, y + 11), s, font=self.f.txt[11], fill=DIM + (255,))

    def _profile(self, d, tt, x, y, w, h):
        f = self.f
        spaced(d, (x, y - 20), "ALTITUDE OVER THE FLIGHT", f.txt[11], DIM + (255,), 1.0)
        ts, hs = self.profile
        T, Hm = ts[-1], max(35.0, float(np.nanmax(hs)))
        k = np.linspace(0, len(ts) - 1, 300).astype(int)
        pts = [(x + ts[j] / T * w, y + h - hs[j] / Hm * h) for j in k]
        d.polygon([S(*p) for p in pts] + [S(x + w, y + h), S(x, y + h)], fill=ACCENT + (38,))
        d.line([S(*p) for p in pts], fill=ACCENT + (230,), width=int(1.6 * SS))
        d.line(S(x, y + h, x + w, y + h), fill=LINE + (255,), width=SS)
        px_ = x + tt / T * w
        hh = float(np.interp(tt, ts, hs))
        py_ = y + h - hh / Hm * h
        d.line(S(px_, y - 2, px_, y + h), fill=(255, 255, 255, 120), width=SS)
        d.ellipse(S(px_ - 4, py_ - 4, px_ + 4, py_ + 4), fill=(255, 255, 255, 255))
        for frac, s in ((0, "0"), (0.5, f"{T / 120:.0f} min"), (1, f"{T / 60:.0f} min")):
            tw = d.textlength(s, font=f.txt[11]) / SS
            d.text(S(x + frac * w - frac * tw, y + h + 4), s, font=f.txt[11], fill=DIM + (255,))

    def _map(self, d, i, t, h, shown, rect):
        """satellite map with the tracks; returns the (image, position, mask) to paste"""
        x, y, w, hgt = rect
        sess, hr = self.sess, self.hr
        rmax = nice_rmax(h)
        mpp = round(max(0.3, 2.3 * rmax / w), 2)
        if mpp not in self.mapviews:
            self.mapviews[mpp] = MapView(hr.mosaic, int(w), int(hgt), mpp, dim=0.62)
        c = hr.map_centre(i)
        img, to_panel = self.mapviews[mpp].render(c[0], c[1])
        E0, N0 = sess.drone_EN(t, smooth=True)
        Rv = self.stab.Rv[i]
        ax = Rv @ np.array([0, 0, 1.0]); brg = math.degrees(math.atan2(ax[0], ax[1]))
        Gf = footprint(sess.cam, Rv, h, rmax)
        fx, fy = to_panel(E0 + Gf[:, 0], N0 + Gf[:, 1])
        d.polygon([S(x + a, y + b) for a, b in zip(fx, fy)], fill=ACCENT + (34,), outline=ACCENT + (170,))
        m = (sess.ti >= t + sess.toff - 60) & (sess.ti <= t + sess.toff)
        if m.sum() > 2:
            px_, py_ = to_panel(sess.E_s[m][::9], sess.N_s[m][::9])
            d.line([S(x + a, y + b) for a, b in zip(px_, py_)], fill=(255, 255, 255, 110), width=int(1.5 * SS))
        seen_ids = {int(r.id) for r, _ in shown}
        for r in hr.rows(i):
            if r.status == "tentative" or int(r.id) < 0:
                continue
            X, Y = to_panel(r.E, r.N)
            if not (6 < X < w - 6 and 6 < Y < hgt - 6):
                continue
            col = CLASS_COL.get(r.grp, (200, 200, 200))
            rad = 4.5
            if int(r.id) in seen_ids:
                d.ellipse(S(x + X - rad, y + Y - rad, x + X + rad, y + Y + rad), fill=col + (255,), outline=(10, 12, 16, 255), width=SS)
                d.text(S(x + X + 7, y + Y - 8), str(int(r.id)), font=self.f.num[13], fill=TEXT + (255,), stroke_width=2 * SS // 2, stroke_fill=(0, 0, 0, 200))
            else:
                d.ellipse(S(x + X - rad, y + Y - rad, x + X + rad, y + Y + rad), outline=col + (150,), width=int(1.5 * SS))
        DX, DY = to_panel(E0, N0)
        b = math.radians(brg)
        tri = [(DX + math.sin(b) * 11, DY - math.cos(b) * 11), (DX + math.sin(b + 2.5) * 8, DY - math.cos(b + 2.5) * 8),
               (DX + math.sin(b - 2.5) * 8, DY - math.cos(b - 2.5) * 8)]
        d.polygon([S(x + a, y + bb) for a, bb in tri], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
        sb = 50 if mpp * w > 220 else 25
        d.rectangle(S(x + 12, y + hgt - 18, x + 12 + sb / mpp, y + hgt - 15), fill=(255, 255, 255, 220))
        d.text(S(x + 12, y + hgt - 36), f"{sb} m", font=self.f.txt[11], fill=(255, 255, 255, 230))
        d.text(S(x + w - 20, y + 8), "N", font=self.f.semi[14], fill=(255, 255, 255, 230))
        att = "Imagery: Esri, Maxar, Earthstar Geographics"
        d.text((S(x + w - 10)[0] - d.textlength(att, font=self.f.txt[11]), S(0, y + hgt - 20)[1]), att, font=self.f.txt[11], fill=(255, 255, 255, 120))
        mask = Image.new("L", (int(w), int(hgt)), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, int(w) - 1, int(hgt) - 1), radius=10, fill=255)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (int(x), int(y)), mask

    # ---- cards and titles
    def backdrop(self, i):
        im = self.frame_plain(i)
        im = cv2.GaussianBlur(im, (0, 0), 9)
        return (im.astype(np.float32) * 0.35).astype(np.uint8)

    def frame_plain(self, i):
        pic = cv2.cvtColor(self.picture(i, "rgb"), cv2.COLOR_BGR2RGB)
        return cv2.resize(pic, (W, int(W * CAM_H / CAM_W)))[(int(W * CAM_H / CAM_W) - H) // 2:][:H]

    def card(self, back, entries):
        """title / closing card from the config's lines: {text, style (big / light / text / accent), gap} centred, or
        {num, text, gap}: a fact with its number in an accent column"""
        f = self.f
        ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        fonts = {"big": f.num[84], "light": f.light[30], "text": f.txt[20], "accent": f.semi[22]}
        cols = {"big": TEXT, "light": TEXT, "text": DIM, "accent": ACCENT}
        track = {"big": 6.0, "accent": 3.0}
        facts = [e for e in entries if "num" in e]
        nw = max((d.textlength(e["num"], font=f.num[48]) / SS for e in facts), default=0)
        tw = max((d.textlength(e["text"], font=f.light[28]) / SS for e in facts), default=0)
        fx = (W - (nw + 30 + tw)) / 2
        height = lambda e: 64 if "num" in e else fonts[e.get("style", "text")].size / SS * 1.3 + (18 if e.get("style") == "big" else 0)
        y = (H - sum(e.get("gap", 10) + height(e) for e in entries)) / 2
        for e in entries:
            y += e.get("gap", 10)
            if "num" in e:
                n = e["num"]
                d.text(S(fx + nw - d.textlength(n, font=f.num[48]) / SS, y), n, font=f.num[48], fill=ACCENT + (255,))
                d.text(S(fx + nw + 30, y + 15), e["text"], font=f.light[28], fill=TEXT + (255,))
            else:
                st = e.get("style", "text"); fnt = fonts[st]; text = e.get("text", "")
                if st in track:
                    wt = sum(d.textlength(ch, font=fnt) / SS for ch in text) + track[st] * (len(text) - 1)
                    spaced(d, ((W - wt) / 2, y), text, fnt, cols[st] + (255,), track[st])
                    if st == "big":
                        d.rounded_rectangle(S(W / 2 - 34, y + fnt.size / SS * 1.3 + 4, W / 2 + 34, y + fnt.size / SS * 1.3 + 8), radius=2 * SS, fill=ACCENT + (255,))
                elif text:
                    wt = d.textlength(text, font=fnt) / SS
                    d.text(S((W - wt) / 2, y), text, font=fnt, fill=cols[st] + (255,))
            y += height(e)
        img = Image.alpha_composite(Image.fromarray(back).convert("RGBA"), ov.resize((W, H), Image.LANCZOS))
        return np.asarray(img.convert("RGB"))

    def section_title(self, name, note=None):
        """RGBA layer: the section's name at the lower left of the camera view"""
        ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        g = np.zeros((H * SS, CAM_W * SS, 4), np.uint8)               # soft dark band for legibility
        ramp = np.clip((np.arange(H * SS) - (H - 260) * SS) / (200 * SS), 0, 1)
        g[..., 3] = (ramp[:, None] * 170).astype(np.uint8)
        ov.paste(Image.fromarray(g, "RGBA"), (0, 0))
        d.text(S(46, H - 118), name, font=self.f.semi[34], fill=TEXT + (255,))
        if note:
            d.text(S(48, H - 70), note, font=self.f.txt[20], fill=ACCENT + (255,))
        return np.asarray(ov.resize((W, H), Image.LANCZOS)).astype(np.float32)


class VideoOut:
    """frames to libx264 through ffmpeg"""

    def __init__(self, path, fps=FPS):
        self.path = str(path)
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                                   "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart", self.path], stdin=subprocess.PIPE)
        self.n = 0

    def put(self, rgb, times=1):
        b = np.ascontiguousarray(rgb, np.uint8).tobytes()
        for _ in range(times):
            self.p.stdin.write(b)
        self.n += times

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def blend(a, b, w):
    return (a.astype(np.float32) * (1 - w) + b.astype(np.float32) * w).astype(np.uint8)


def over(img, layer, alpha):
    """RGBA float layer over an RGB image with an extra alpha factor"""
    if layer is None or alpha <= 0:
        return img
    a = layer[..., 3:4] / 255.0 * alpha
    return (img.astype(np.float32) * (1 - a) + layer[..., :3] * a).astype(np.uint8)


def render(demo, sections, out, log=print, workers=4):
    """sections: dicts with kind title / video / views / outro (see the config's demo block)"""
    vo = VideoOut(out)
    prev = None                                                     # last frame, for the cross-fade into the next section
    fade = int(0.5 * FPS)

    def emit_transition(first):
        nonlocal prev
        start = prev if prev is not None else np.zeros_like(first)  # the first section fades in from black
        for k in range(1, fade + 1):
            vo.put(blend(start, first, k / (fade + 1)))

    for sec in sections:
        kind = sec["kind"]
        if kind in ("title", "outro"):
            back = demo.backdrop(sec.get("backdrop_frame", 2522))
            card = demo.card(back, sec["lines"])
            if prev is None:
                for k in range(fade):
                    vo.put(blend(np.zeros_like(card), card, (k + 1) / fade))
            else:
                emit_transition(card)
            vo.put(card, int(sec.get("seconds", 4) * FPS))
            prev = card
            continue
        frames = list(range(sec["start"], sec["end"] + 1, sec.get("step", 1)))
        hold = sec.get("hold", 6)                                    # output frames per camera frame (6 = real time at 30 fps)
        title = demo.section_title(sec["name"], sec.get("note")) if demo.captions else None
        views = None
        pf = None
        if kind == "views":
            order = sec.get("views", ["rgb", "s0", "nz", "nxy", "nxyz", "pn_vyzlut", "pn_pxdiffuse", "pn_pxspecularv2"])
            per = max(1, int(round(sec.get("view_s", 6.0) * FPS / hold)))
            views = {fr: order[min(k // per, len(order) - 1)] for k, fr in enumerate(frames)}
            pf = polar.Prefetcher(demo.store, demo.matrix, frames, workers, full=True)   # full resolution: the view is large
        n_out = 0
        title_s = sec.get("title_s", 4.0)
        try:
            for k, fr in enumerate(frames):
                view = views[fr] if views else "rgb"
                P = pf.get(fr) if pf is not None else None
                img = demo.frame(fr, view, P, sec.get("panel", sec["name"]))
                if k == 0:
                    emit_transition(over(img, title, 1.0))
                nxt_view = views[frames[k + 1]] if views and k + 1 < len(frames) else view
                if views and nxt_view != view:                          # cross-fade into the next view on this frame
                    img2 = demo.frame(fr, nxt_view, P, sec.get("panel", sec["name"]))
                    for j in range(hold):
                        tt = n_out / FPS
                        a = min(1.0, max(0.0, (title_s - tt) / 0.6)) if tt < title_s else 0.0
                        vo.put(over(blend(img, img2, (j + 1) / hold), title, a)); n_out += 1
                    prev = img2
                    continue
                for j in range(hold):
                    tt = n_out / FPS
                    a = min(1.0, max(0.0, (title_s - tt) / 0.6)) if tt < title_s else 0.0
                    vo.put(over(img, title, a)); n_out += 1
                prev = img
                if k % 100 == 0:
                    log(f"demo: {sec['name']}: frame {fr}")
        finally:
            if pf is not None:
                pf.close()
    for k in range(fade):                                            # fade out
        vo.put(blend(prev, np.zeros_like(prev), (k + 1) / fade))
    vo.close()
    log(f"{out}: {vo.n / FPS:.1f} s")
    return out
