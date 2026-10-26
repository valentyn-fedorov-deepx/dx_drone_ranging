"""Demo video and viewer frames, 1920 x 1080: the camera view with labelled detections, a drone-centred radar, a
north-up satellite map and a target table.

Labels follow the VyzAI UI: "CLASS id", then Xo / Yo / Zo (camera frame: right, down, along the optical axis, metres),
Range (slant distance to the tracked point) and Velocity with an arrow for the direction of motion. The arrow and the
radar use the drone's view: up = moving away from the drone along the camera's ground heading, right = to the right.
Label modes: full (every detection), compact (class, id, range), auto (full for people and moving objects, compact
for parked vehicles), off. Labels show what the system has in that frame: every detection of the frame is labelled,
nothing else (a missed detection is a missing label); a detection the tracker has not confirmed yet shows the class
without an id and no velocity. Labels sit next to their object where there is room and move away (with a thin leader)
where they would cover another label or object. The radar and the map show the tracker's state: filled = detected in
this frame, hollow = not detected (kept by the tracker); the table lists the objects detected in this frame. An object counts as moving when its net displacement over a
few seconds says so (hysteresis), so the wander of a parked car's ranged position does not show as speed. The map
shows every remembered track by id: filled while in view, hollow when out of view.

The camera view is the video frame or any picture of the same view (the polarisation views of the raw frames),
optionally stabilised (stabilise.py: the view turned to a smooth virtual camera, labels mapped the same way; the radar
footprint then follows the virtual camera too).
"""
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .mapview import MapView
from .ranging import footprint

CW, CH = 1920, 1080
VW, VH = 1280, 938                     # camera view
TOP = 30                               # header bar over the camera view
RX, RW, RH = 1280, 640, 440            # radar panel
MX, MY, MW, MH = 1280, 440, 640, 640   # map panel
BG = (12, 14, 12); GREEN = (60, 230, 90); GDIM = (25, 90, 40); ORANGE = (255, 165, 40); CYAN = (90, 220, 255); GREY = (170, 170, 170)
WHITE = (240, 240, 240)
COLORS = {"person": GREEN, "vehicle": ORANGE, "bicycle": CYAN}
NAMES = {"person": "PERSON", "car": "CAR", "truck": "TRUCK", "bus": "BUS", "motorcycle": "MOTORCYCLE", "bicycle": "BICYCLE"}
OPTICAL_AXIS = np.array([0, 0, 1.0])
LABEL_MODES = ("full", "auto", "compact", "off")


def _font(size, bold=False):
    cands = ["C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"]
    try:
        import matplotlib
        cands.append(str(Path(matplotlib.get_data_path()) / "fonts/ttf" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")))
    except Exception:
        pass
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def nice_rmax(h):
    for r in (40, 60, 80, 100, 120, 160, 200):
        if r >= 3.6 * h:
            return r
    return 200


def arrow(dr, x, y, ang_deg, length, color, width=2):
    """arrow centred at (x, y), angle clockwise from up"""
    a = math.radians(ang_deg)
    dx, dy = math.sin(a), -math.cos(a)
    x0, y0 = x - dx * length / 2, y - dy * length / 2
    x1, y1 = x + dx * length / 2, y + dy * length / 2
    dr.line((x0, y0, x1, y1), fill=color, width=width)
    hl = max(4, length * 0.45)
    for s in (-1, 1):
        b = a + math.pi + s * 0.5
        dr.line((x1, y1, x1 + math.sin(b) * hl, y1 - math.cos(b) * hl), fill=color, width=width)


def label_lines(r, mode, still):
    name = NAMES.get(r.cls, r.cls.upper())
    head = name if int(r.id) < 0 else f"{name} {int(r.id)}"
    if mode == "compact" or (mode == "auto" and r.grp != "person" and still):
        return head, [f"Range: {r.range:.1f} m"], False
    vel = "Velocity: -" if int(r.id) < 0 else f"Velocity: {0 if still else round(r.speed_kmh):.0f} km/h"   # unconfirmed: no velocity yet
    return head, [f"Xo: {r.Xo:.2f} m", f"Yo: {r.Yo:.2f} m", f"Zo: {r.Zo:.2f} m", f"Range: {r.range:.2f} m", vel], not still and int(r.id) >= 0


def _overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def place_labels(items, W, H, top=TOP + 4, prefer=None):
    """items: (box, w, h) in drawing order -> the top-left corner of each label: the first spot (above, below, right,
    left of its object, then stacked further up or down) that covers no placed label and no object; when every spot
    covers something, the one that covers least. prefer: per item an offset (dx, dy) of the label from its box centre
    to try first (where it was in the previous frame), or None"""
    boxes = [b for b, _, _ in items]
    placed, out = [], []
    for k, (box, w, h) in enumerate(items):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        cands = [(cx - w / 2, y1 - h - 6), (cx - w / 2, y2 + 6), (x2 + 8, cy - h / 2), (x1 - w - 8, cy - h / 2)]
        if prefer is not None and prefer[k] is not None:
            cands.insert(0, (cx + prefer[k][0], cy + prefer[k][1]))
        for s in range(1, 8):
            for dx in (0.0, -0.6 * w, 0.6 * w):
                cands += [(cx - w / 2 + dx, y1 - h - 6 - s * 0.55 * h), (cx - w / 2 + dx, y2 + 6 + s * 0.55 * h)]
        best, best_cost = None, None
        for j, (x, y) in enumerate(cands):
            x = min(max(x, 2.0), W - w - 2.0); y = min(max(y, top), H - h - 2.0)
            rect = (x, y, x + w, y + h)
            cover = 4 * sum(_overlap(rect, p) for p in placed) + sum(_overlap(rect, b) for m, b in enumerate(boxes) if m != k)
            cover += 2 * _overlap(rect, box)
            cost = cover + 25.0 * j
            if best_cost is None or cost < best_cost:
                best, best_cost = (x, y), cost
            if cover == 0:
                break
        placed.append((best[0], best[1], best[0] + w, best[1] + h))
        out.append(best)
    return out


class HudRenderer:
    """One HUD frame at a time, any frame in any order (the video export and the viewer use it)."""

    def __init__(self, sess, R, T, track, mosaic, hud=None, drone_name="DRONE", stab=None):
        self.sess, self.R, self.track, self.mosaic = sess, R, track, mosaic
        self.stab = stab                                             # Stabiliser or None
        self.last_offset = {}                                        # track id -> label offset from its box centre
        self.hud = hud = hud or {}
        self.labels = hud.get("labels", "auto")
        self.boxes = hud.get("boxes", False)                         # detection boxes off: the labels mark the objects
        self.trail_s = hud.get("trail_s", 60.0)
        self.min_mv = hud.get("min_moving_s", {"person": 0.0, "vehicle": 1.0, "bicycle": 0.6})
        self.drone_name = drone_name
        self.f = dict(B=_font(17, True), S=_font(14), T=_font(13), L=_font(13), H=_font(14, True), M=_font(12, True),
                      Ls=_font(11), Hs=_font(12, True), A=_font(10))
        self.byT = {k: g for k, g in T.groupby("frame")}
        self.sx, self.sy = VW / sess.cam.width, VH / sess.cam.height
        self.mapviews = {}
        self._cand = {}
        self._vid = None

    def label_box(self, i, r):
        """camera-pixel box of the row's detection in this frame, or None (not detected in this frame)"""
        if r.status in ("seen", "tentative") and r.det_row >= 0:
            o = self.R.loc[int(r.det_row)]
            return o.x1, o.y1, o.x2, o.y2
        return None

    # ---- coordinates of the camera view
    def view_to_camera(self, x, y, i=None):
        """HUD pixel -> camera pixel (frame i when stabilised), or None outside the camera view"""
        if not (0 <= x < VW and TOP <= y < VH):
            return None
        if self.stab is not None:
            u, v = self.stab.to_camera(i, [[x, y]])[0]
            return float(u), float(v)
        return x / self.sx, y / self.sy

    def camera_to_view(self, u, v, i=None):
        if self.stab is not None:
            x, y = self.stab.to_view(i, [[u, v]])[0]
            return float(x), float(y)
        return u * self.sx, v * self.sy

    def view_box(self, i, b):
        """camera-pixel box -> HUD box"""
        if self.stab is None:
            return b[0] * self.sx, b[1] * self.sy, b[2] * self.sx, b[3] * self.sy
        q = self.stab.to_view(i, [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]])
        return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()

    def video_frame(self, i):
        if self._vid is None:
            self._vid = self.sess.video()
        return self._vid.read(i)

    def rows(self, i):
        g = self.byT.get(i)
        rows = [] if g is None else list(g.itertuples())
        return [r._replace(moving=bool(r.moving and r.moving_s >= self.min_mv.get(r.grp, 0.0))) for r in rows]   # vehicles: debounce

    def geometry(self, i, display=False):
        """time, camera rotation (the virtual camera's with display=True when stabilised), height, camera bearing,
        drone position"""
        t = self.sess.ft[i]
        Rwc = self.stab.Rv[i] if display and self.stab is not None else self.sess.R_wc(t)
        h = float(self.track.h.values[i])
        ax = Rwc @ OPTICAL_AXIS
        E0, N0 = self.sess.drone_EN(t, smooth=True)
        return t, Rwc, h, math.degrees(math.atan2(ax[0], ax[1])), E0, N0

    def _centre_cand(self, i):
        if i not in self._cand:
            t, Rwc, h, _, E0, N0 = self.geometry(i)
            Gf = footprint(self.sess.cam, Rwc, h, nice_rmax(h))
            self._cand[i] = np.array([E0 + 0.5 * Gf[:, 0].mean(), N0 + 0.5 * Gf[:, 1].mean()])
        return self._cand[i]

    def map_centre(self, i, n=30, a=0.1):
        """the map follows the drone and its footprint with the lag of a 0.1 running average over the frames before"""
        ks = range(max(0, i - n), i + 1)
        w = np.array([(1 - a) ** (i - k) for k in ks])
        return (np.array([self._centre_cand(k) for k in ks]) * w[:, None]).sum(0) / w.sum()

    # ---- the frame
    def compose(self, i, picture=None, view_name=None):
        """frame i with the camera view = picture (BGR, any size of the camera's aspect) or the video frame -> BGR"""
        if picture is None:
            picture = self.video_frame(i)
            if picture is None:
                return None
        sess = self.sess
        t, Rwc, h, cam_brg, E0, N0 = self.geometry(i, display=True)
        if self.stab is not None:
            frame = self.stab.warp(picture, i)
        else:
            frame = cv2.resize(picture, (VW, VH), interpolation=cv2.INTER_AREA if picture.shape[1] > VW else cv2.INTER_LINEAR)
        canvas = Image.new("RGB", (CW, CH), BG)
        canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, 0))
        rows = self.rows(i)
        active = [r for r in rows if r.status in ("seen", "coast")]
        detected = [r for r in rows if r.status in ("seen", "tentative") and r.det_row >= 0]
        if self.labels != "off":
            self._labels(canvas, i, rows)
        dr = ImageDraw.Draw(canvas)
        sats = int(np.interp(t + sess.toff, sess.ti, sess.nsat))
        dr.rectangle((0, 0, VW, TOP), fill=(0, 0, 0))
        dr.text((10, 6), f"{self.drone_name}  //  T+{t - sess.ft[0]:6.1f} s  //  ALT {h:4.1f} m AGL  //  GNSS 3D {sats} sats  //  detected {len(detected)}",
                font=self.f["S"], fill=GREY)
        if view_name:
            s = f"VIEW {view_name}"
            dr.text((VW - 12 - dr.textbbox((0, 0), s, font=self.f["H"])[2], 6), s, font=self.f["H"], fill=GREEN)
        rmax = nice_rmax(h)
        Gf = footprint(sess.cam, Rwc, h, rmax)
        self._radar(canvas, active, Gf, cam_brg, E0, N0, rmax)
        self._map(canvas, i, t, rows, Gf, cam_brg, E0, N0, rmax)
        self._table(canvas, [r for r in detected if r.status == "seen"], str(self.track.src.values[i]))
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    def _labels(self, canvas, i, rows):
        f = self.f
        ov = Image.new("RGBA", (VW, VH), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
        dr = ImageDraw.Draw(canvas)
        labelled = [(r, b) for r in rows for b in [self.label_box(i, r)] if b is not None]
        small = len(labelled) > 14                                    # crowded: smaller labels
        FH, FL, lh = (f["Hs"], f["Ls"], 13) if small else (f["H"], f["L"], 16)
        order = sorted(labelled, key=lambda rb: (rb[0].grp != "person", not rb[0].moving, rb[0].range))
        items, info = [], []
        for r, b in order:
            box = self.view_box(i, b)
            if not (0 <= 0.5 * (box[0] + box[2]) < VW and TOP <= 0.5 * (box[1] + box[3]) < VH):
                continue                                              # outside the (zoomed) view
            head, lines, with_arrow = label_lines(r, self.labels, not r.moving)
            wid = max([od.textbbox((0, 0), head, font=FH)[2]] + [od.textbbox((0, 0), s, font=FL)[2] for s in lines]) + (22 if with_arrow else 0) + 12
            hgt = lh + 4 + lh * len(lines) + 4
            items.append((box, wid, hgt)); info.append((r, head, lines, with_arrow))
        prefer = [self.last_offset.get(int(r.id)) if int(r.id) >= 0 else None for r, _, _, _ in info]
        for (box, wid, hgt), (lx, ly), (r, head, lines, with_arrow) in zip(items, place_labels(items, VW, VH, prefer=prefer), info):
            col = COLORS.get(r.grp, GREY)
            x1, y1, x2, y2 = box
            if int(r.id) >= 0:                                       # keep the label on the same side next frame
                self.last_offset[int(r.id)] = (lx - 0.5 * (x1 + x2), ly - 0.5 * (y1 + y2))
            if self.boxes:
                dr.rectangle(box, outline=col, width=2)
            gap_x = max(x1 - (lx + wid), lx - x2, 0); gap_y = max(y1 - (ly + hgt), ly - y2, 0)
            if gap_x > 10 or gap_y > 10:                               # away from its object: a thin leader
                ex, ey = min(max(0.5 * (x1 + x2), lx + 4), lx + wid - 4), (ly + hgt if ly + hgt < y1 else (ly if ly > y2 else ly + hgt / 2))
                tx, ty = min(max(ex, x1), x2), min(max(ey, y1), y2)
                od.line((ex, ey, tx, ty), fill=col + (210,), width=1)
                od.ellipse((tx - 2, ty - 2, tx + 2, ty + 2), fill=col + (230,))
            unconfirmed = int(r.id) < 0
            od.rectangle((lx, ly, lx + wid, ly + hgt), fill=(28, 34, 28, 185), outline=col + ((150,) if unconfirmed else (230,)), width=1)
            od.text((lx + 6, ly + 3), head, font=FH, fill=WHITE + (255,))
            for k, s in enumerate(lines):
                od.text((lx + 6, ly + lh + 5 + lh * k), s, font=FL, fill=WHITE + (255,))
            if with_arrow:
                tw = od.textbbox((0, 0), lines[-1], font=FL)[2]
                arrow(od, lx + 6 + tw + 12, ly + lh + 5 + lh * (len(lines) - 1) + lh / 2, r.rel_dir, 13 if not small else 11, col + (255,), 2)
        canvas.paste(ov, (0, 0), ov)

    def _radar(self, canvas, active, Gf, cam_brg, E0, N0, rmax):
        f = self.f
        dr = ImageDraw.Draw(canvas)
        rr = 185; rcx, rcy = RX + RW // 2, 40 + rr
        dr.line((RX, 0, RX, CH), fill=(45, 45, 45), width=2)
        dr.text((RX + 20, 10), f"RADAR // DRONE-CENTRED // {rmax} M", font=f["B"], fill=GREEN)
        sc = rr / rmax
        for k in range(1, 5):
            r_ = rr * k / 4
            dr.ellipse((rcx - r_, rcy - r_, rcx + r_, rcy + r_), outline=GDIM, width=1)
            dr.text((rcx + 4, rcy - r_ - 16), f"{rmax * k // 4} m", font=f["T"], fill=GDIM)
        dr.line((rcx - rr, rcy, rcx + rr, rcy), fill=GDIM); dr.line((rcx, rcy - rr, rcx, rcy + rr), fill=GDIM)
        a = math.radians(cam_brg)
        xr = Gf[:, 0] * math.cos(a) - Gf[:, 1] * math.sin(a); yf = Gf[:, 0] * math.sin(a) + Gf[:, 1] * math.cos(a)
        ov = Image.new("RGBA", (CW, CH), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
        od.polygon([(rcx + x * sc, rcy - y * sc) for x, y in zip(xr, yf)], fill=(60, 230, 90, 38), outline=(60, 230, 90, 140))
        canvas.paste(ov, (0, 0), ov); dr = ImageDraw.Draw(canvas)
        dr.polygon([(rcx, rcy - 11), (rcx - 9, rcy + 9), (rcx + 9, rcy + 9)], fill=GREEN)
        for r in active:
            de, dn = r.E - E0, r.N - N0
            x = de * math.cos(a) - dn * math.sin(a); y = de * math.sin(a) + dn * math.cos(a)
            if math.hypot(x, y) > rmax:
                continue
            X, Y = rcx + x * sc, rcy - y * sc
            col = COLORS.get(r.grp, GREY)
            if r.moving:
                L = min(3.0 * r.speed * sc, 60)
                b_ = math.radians(r.rel_dir)
                dr.line((X, Y, X + math.sin(b_) * L, Y - math.cos(b_) * L), fill=col, width=2)
            rad = 5 if r.grp == "person" else 6
            dr.ellipse((X - rad, Y - rad, X + rad, Y + rad), fill=col if r.status == "seen" else None, outline=col, width=2)
            dr.text((X + 7, Y - 7), str(int(r.id)), font=f["M"], fill=col)

    def _map(self, canvas, i, t, rows, Gf, cam_brg, E0, N0, rmax):
        f, sess = self.f, self.sess
        Gw = np.c_[E0 + Gf[:, 0], N0 + Gf[:, 1]]
        centre = self.map_centre(i)
        mpp = round(max(self.hud.get("map_min_m_per_px", 0.26), 2.4 * rmax / MW), 2)
        if mpp not in self.mapviews:
            self.mapviews[mpp] = MapView(self.mosaic, MW, MH, mpp)
        mimg, to_panel = self.mapviews[mpp].render(centre[0], centre[1])
        mp = Image.fromarray(cv2.cvtColor(mimg, cv2.COLOR_BGR2RGB))
        md = ImageDraw.Draw(mp, "RGBA")
        m = (sess.ti >= t + sess.toff - self.trail_s) & (sess.ti <= t + sess.toff)
        if m.sum() > 2:
            px_, py_ = to_panel(sess.E_s[m][::9], sess.N_s[m][::9])
            md.line(list(zip(px_, py_)), fill=(255, 255, 255, 150), width=2)
        fx, fy = to_panel(Gw[:, 0], Gw[:, 1])
        md.polygon(list(zip(fx, fy)), fill=(60, 230, 90, 45), outline=(60, 230, 90, 200))
        for r in rows:
            if r.status == "tentative":
                continue
            X, Y = to_panel(r.E, r.N)
            if not (-20 < X < MW + 20 and -20 < Y < MH + 20):
                continue
            col = COLORS.get(r.grp, GREY)
            rad = 5 if r.grp == "person" else 6
            if r.status != "seen":                                   # not detected in this frame: the tracker's memory
                md.ellipse((X - rad, Y - rad, X + rad, Y + rad), outline=col + (160,), width=2)
                md.text((X + 7, Y - 7), str(int(r.id)), font=f["M"], fill=col + (170,))
                continue
            if r.moving:
                L = min(3.0 * r.speed / mpp, 70)
                b_ = math.radians(r.dir)
                md.line((X, Y, X + math.sin(b_) * L, Y - math.cos(b_) * L), fill=col + (255,), width=2)
            md.ellipse((X - rad, Y - rad, X + rad, Y + rad), fill=col + (255,), outline=(0, 0, 0, 255))
            md.text((X + 7, Y - 8), str(int(r.id)), font=f["M"], fill=WHITE + (255,), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        DX, DY = to_panel(E0, N0)
        b_ = math.radians(cam_brg)
        tri = [(DX + math.sin(b_) * 12, DY - math.cos(b_) * 12), (DX + math.sin(b_ + 2.5) * 9, DY - math.cos(b_ + 2.5) * 9),
               (DX + math.sin(b_ - 2.5) * 9, DY - math.cos(b_ - 2.5) * 9)]
        md.polygon(tri, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
        sb = 50 if mpp * MW > 200 else 25                                 # scale bar
        md.rectangle((14, MH - 30, 14 + sb / mpp, MH - 25), fill=(255, 255, 255, 230))
        md.text((14, MH - 48), f"{sb} m", font=f["T"], fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        md.text((MW - 24, 34), "N", font=f["B"], fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        arrow(md, MW - 19, 66, 0, 18, (255, 255, 255, 255), 2)
        md.text((MW - 230, MH - 18), self.hud.get("map_attribution", "Imagery: Esri, Maxar, Earthstar Geographics"), font=f["A"], fill=(220, 220, 220, 200))
        canvas.paste(mp, (MX, MY))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle((MX, MY, MX + MW, MY + 28), fill=(0, 0, 0))
        lat, lon = sess.en_to_latlon(E0, N0)
        dr.text((MX + 20, MY + 6), f"MAP // NORTH UP // {lat:.5f}, {lon:.5f}", font=f["B"], fill=GREEN)
        dr.line((MX, MY, MX + MW, MY), fill=(45, 45, 45), width=2)

    def _table(self, canvas, active, src):
        f = self.f
        dr = ImageDraw.Draw(canvas)
        ty = VH + 6
        cols_x = (16, 70, 210, 330, 450, 580)
        for x, s in zip(cols_x, ("ID", "CLASS", "RANGE", "GROUND", "VELOCITY", "HEADING")):
            dr.text((x, ty), s, font=f["T"], fill=GREY)
        dr.line((10, ty + 20, VW - 10, ty + 20), fill=(60, 60, 60))
        for k, r in enumerate(sorted(active, key=lambda r: r.range)[:4]):
            y0 = ty + 26 + k * 26
            col = COLORS.get(r.grp, GREY)
            dr.rectangle((cols_x[0] - 8, y0 + 4, cols_x[0] - 2, y0 + 14), fill=col)
            dr.text((cols_x[0], y0), str(int(r.id)), font=f["S"], fill=WHITE)
            dr.text((cols_x[1], y0), NAMES.get(r.cls, r.cls.upper()), font=f["S"], fill=(220, 220, 220))
            dr.text((cols_x[2], y0), f"{r.range:6.1f} m", font=f["S"], fill=WHITE)
            dr.text((cols_x[3], y0), f"{r.D:6.1f} m", font=f["S"], fill=(200, 200, 200))
            still = not r.moving
            dr.text((cols_x[4], y0), "0 km/h" if still else f"{r.speed_kmh:.0f} km/h", font=f["S"], fill=(200, 200, 200))
            if not still:
                arrow(dr, cols_x[5] + 8, y0 + 9, r.rel_dir, 13, col, 2)
                dr.text((cols_x[5] + 24, y0), f"{(r.dir + 360) % 360:3.0f}°", font=f["S"], fill=(200, 200, 200))
        dr.text((900, VH + 6), f"altitude source: {src.replace('_', ' ')}", font=f["T"], fill=(110, 110, 110))
        dr.text((900, VH + 26), "Xo right, Yo down, Zo along the camera axis", font=f["T"], fill=(110, 110, 110))
        dr.text((900, VH + 46), "arrows: up = away from the drone", font=f["T"], fill=(110, 110, 110))


def render(sess, R, T, track, mosaic, out, f0=0, f1=10 ** 9, fps=5.0, hud=None, drone_name="DRONE", log=print,
           frames=None, picture=None, view_name=None, stab=None):
    """video of frames f0..f1 (or the given frame numbers); picture(i) -> the camera view (BGR) instead of the video
    frame; view_name: a name or i -> name for the header; stab: Stabiliser"""
    hr = HudRenderer(sess, R, T, track, mosaic, hud, drone_name, stab)
    if frames is None:
        frames = range(f0, min(f1, sess.n_frames - 1) + 1)
    out = str(out)
    tmp = out.replace(".mp4", "_tmp.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (CW, CH))
    for n, i in enumerate(frames):
        img = hr.compose(i, picture(i) if picture else None, view_name(i) if callable(view_name) else view_name)
        if img is None:
            break
        vw.write(img)
        if n % 500 == 0:
            log("render", i)
    vw.release()
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                        "-movflags", "+faststart", out], check=True)    # faststart: plays while it downloads
        os.remove(tmp)
    except (OSError, subprocess.CalledProcessError):
        os.replace(tmp, out)
    log(out)
    return out
