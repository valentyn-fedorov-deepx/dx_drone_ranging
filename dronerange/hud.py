"""Demo video, 1920 x 1080: the camera view with labelled tracks, a drone-centred radar, a north-up satellite map and a
target table.

Labels follow the VyzAI UI: "CLASS id", then Xo / Yo / Zo (camera frame: right, down, along the optical axis, metres),
Range (slant distance to the tracked point) and Velocity with an arrow for the direction of motion. The arrow and the
radar use the drone's view: up = moving away from the drone along the camera's ground heading, right = to the right.
Label modes: full (every track), compact (class, id, range), auto (full for people and moving objects, compact for
parked vehicles). An object counts as moving when its net displacement over 3 s says so (hysteresis), so the
wander of a parked car's ranged position does not show as speed. The map shows every remembered track by id: filled while in view, hollow when out of view.
"""
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .mapview import MapView
from .ranging import footprint

CW, CH = 1920, 1080
VW, VH = 1280, 938                     # camera view
RX, RW, RH = 1280, 640, 440            # radar panel
MX, MY, MW, MH = 1280, 440, 640, 640   # map panel
BG = (12, 14, 12); GREEN = (60, 230, 90); GDIM = (25, 90, 40); ORANGE = (255, 165, 40); CYAN = (90, 220, 255); GREY = (170, 170, 170)
WHITE = (240, 240, 240)
COLORS = {"person": GREEN, "vehicle": ORANGE, "bicycle": CYAN}
NAMES = {"person": "PERSON", "car": "CAR", "truck": "TRUCK", "bus": "BUS", "motorcycle": "MOTORCYCLE", "bicycle": "BICYCLE"}
OPTICAL_AXIS = np.array([0, 0, 1.0])


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
    head = f"{name} {int(r.id)}"
    if mode == "compact" or (mode == "auto" and r.grp != "person" and still):
        return head, [f"Range: {r.range:.1f} m"], False
    v = 0 if still else round(r.speed_kmh)
    return head, [f"Xo: {r.Xo:.2f} m", f"Yo: {r.Yo:.2f} m", f"Zo: {r.Zo:.2f} m", f"Range: {r.range:.2f} m", f"Velocity: {v:.0f} km/h"], not still


def render(sess, R, T, track, mosaic, out, f0=0, f1=10 ** 9, fps=5.0, hud=None, drone_name="DRONE", log=print):
    hud = hud or {}
    mode = hud.get("labels", "auto")
    trail_s = hud.get("trail_s", 60.0)
    FB, FS, FT, FL, FH = _font(17, True), _font(14), _font(13), _font(13), _font(14, True)
    FM = _font(12, True)
    cam, ft = sess.cam, sess.ft
    byT = {k: g for k, g in T.groupby("frame")}
    views = {}
    W, H = cam.width, cam.height
    sx, sy = VW / W, VH / H
    vid = sess.video()
    out = str(out)
    tmp = out.replace(".mp4", "_tmp.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (CW, CH))
    f1 = min(f1, sess.n_frames - 1)
    centre = None
    trail_t = sess.ti
    for i in range(f0, f1 + 1):
        im = vid.read(i)
        if im is None:
            break
        t = ft[i]
        Rwc = sess.R_wc(t); h = float(track.h.values[i]); src = str(track.src.values[i])
        ax = Rwc @ OPTICAL_AXIS; cam_brg = math.degrees(math.atan2(ax[0], ax[1]))
        E0, N0 = sess.drone_EN(t, smooth=True)
        rmax = nice_rmax(h)
        frame = cv2.resize(im, (VW, VH), interpolation=cv2.INTER_AREA)
        canvas = Image.new("RGB", (CW, CH), BG)
        canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, 0))
        g = byT.get(i)
        rows = [] if g is None else list(g.itertuples())
        min_mv = hud.get("min_moving_s", {"person": 0.0, "vehicle": 1.0, "bicycle": 0.6})
        rows = [r._replace(moving=bool(r.moving and r.moving_s >= min_mv.get(r.grp, 0.0))) for r in rows]   # vehicles: debounce the moving flag
        seen = [r for r in rows if r.status == "seen" and r.det_row >= 0]
        active = [r for r in rows if r.status in ("seen", "coast")]
        # ---- boxes and labels on the camera view
        ov = Image.new("RGBA", (VW, VH), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
        dr = ImageDraw.Draw(canvas)
        used = []
        for r in sorted(seen, key=lambda r: r.range):
            o = R.loc[int(r.det_row)]
            col = COLORS.get(r.grp, GREY)
            x1, y1, x2, y2 = o.x1 * sx, o.y1 * sy, o.x2 * sx, o.y2 * sy
            dr.rectangle((x1, y1, x2, y2), outline=col, width=2)
            still = not r.moving
            head, lines, with_arrow = label_lines(r, mode, still)
            wid = max([od.textbbox((0, 0), head, font=FH)[2]] + [od.textbbox((0, 0), s, font=FL)[2] for s in lines]) + (22 if with_arrow else 0) + 12
            hgt = 20 + 16 * len(lines) + 4
            lx = min(max(x1, 2), VW - wid - 2)
            ly = y1 - hgt - 4
            if ly < 34:
                ly = y2 + 4                                           # no room above: below the box
            for (ux, uy, uw, uh) in sorted(used, key=lambda u: u[1]):
                if lx < ux + uw and ux < lx + wid and ly < uy + uh and uy < ly + hgt:
                    ly = uy - hgt - 3
            ly = min(max(ly, 34), VH - hgt - 2)
            used.append((lx, ly, wid, hgt))
            od.rectangle((lx, ly, lx + wid, ly + hgt), fill=(28, 34, 28, 185), outline=col + (230,), width=1)
            od.text((lx + 6, ly + 3), head, font=FH, fill=WHITE + (255,))
            for k, s in enumerate(lines):
                od.text((lx + 6, ly + 21 + 16 * k), s, font=FL, fill=WHITE + (255,))
            if with_arrow:
                tw = od.textbbox((0, 0), lines[-1], font=FL)[2]
                arrow(od, lx + 6 + tw + 12, ly + 21 + 16 * (len(lines) - 1) + 8, r.rel_dir, 13, col + (255,), 2)
        canvas.paste(ov, (0, 0), ov)
        dr = ImageDraw.Draw(canvas)
        # header
        sats = int(np.interp(t + sess.toff, sess.ti, sess.nsat))
        dr.rectangle((0, 0, VW, 30), fill=(0, 0, 0))
        dr.text((10, 6), f"{drone_name}  //  T+{t - ft[0]:6.1f} s  //  ALT {h:4.1f} m AGL  //  GNSS 3D {sats} sats  //  tracks in view {len(active)}",
                font=FS, fill=GREY)
        # ---- radar
        rr = 185; rcx, rcy = RX + RW // 2, 40 + rr
        dr.line((RX, 0, RX, CH), fill=(45, 45, 45), width=2)
        dr.text((RX + 20, 10), f"RADAR // DRONE-CENTRED // {rmax} M", font=FB, fill=GREEN)
        sc = rr / rmax
        for k in range(1, 5):
            r_ = rr * k / 4
            dr.ellipse((rcx - r_, rcy - r_, rcx + r_, rcy + r_), outline=GDIM, width=1)
            dr.text((rcx + 4, rcy - r_ - 16), f"{rmax * k // 4} m", font=FT, fill=GDIM)
        dr.line((rcx - rr, rcy, rcx + rr, rcy), fill=GDIM); dr.line((rcx, rcy - rr, rcx, rcy + rr), fill=GDIM)
        Gf = footprint(cam, Rwc, h, rmax)
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
            dr.text((X + 7, Y - 7), str(int(r.id)), font=FM, fill=col)
        # ---- map
        Gw = np.c_[E0 + Gf[:, 0], N0 + Gf[:, 1]]
        cand = np.array([E0 + 0.5 * (Gw[:, 0].mean() - E0), N0 + 0.5 * (Gw[:, 1].mean() - N0)])
        centre = cand if centre is None else 0.9 * centre + 0.1 * cand
        mpp = max(hud.get("map_min_m_per_px", 0.26), 2.4 * rmax / MW)
        mpp = round(mpp, 2)
        if mpp not in views:
            views[mpp] = MapView(mosaic, MW, MH, mpp)
        mimg, to_panel = views[mpp].render(centre[0], centre[1])
        mp = Image.fromarray(cv2.cvtColor(mimg, cv2.COLOR_BGR2RGB))
        md = ImageDraw.Draw(mp, "RGBA")
        m = (trail_t >= t + sess.toff - trail_s) & (trail_t <= t + sess.toff)
        if m.sum() > 2:
            px_, py_ = to_panel(sess.E_s[m][::9], sess.N_s[m][::9])
            md.line(list(zip(px_, py_)), fill=(255, 255, 255, 150), width=2)
        fx, fy = to_panel(Gw[:, 0], Gw[:, 1])
        md.polygon(list(zip(fx, fy)), fill=(60, 230, 90, 45), outline=(60, 230, 90, 200))
        for r in rows:
            X, Y = to_panel(r.E, r.N)
            if not (-20 < X < MW + 20 and -20 < Y < MH + 20):
                continue
            col = COLORS.get(r.grp, GREY)
            rad = 5 if r.grp == "person" else 6
            if r.status == "memory":
                md.ellipse((X - rad, Y - rad, X + rad, Y + rad), outline=col + (160,), width=2)
                md.text((X + 7, Y - 7), str(int(r.id)), font=FM, fill=col + (170,))
                continue
            if r.moving:
                L = min(3.0 * r.speed / mpp, 70)
                b_ = math.radians(r.dir)
                md.line((X, Y, X + math.sin(b_) * L, Y - math.cos(b_) * L), fill=col + (255,), width=2)
            md.ellipse((X - rad, Y - rad, X + rad, Y + rad), fill=col + (255,), outline=(0, 0, 0, 255))
            md.text((X + 7, Y - 8), str(int(r.id)), font=FM, fill=WHITE + (255,), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        DX, DY = to_panel(E0, N0)
        b_ = math.radians(cam_brg)
        tri = [(DX + math.sin(b_) * 12, DY - math.cos(b_) * 12), (DX + math.sin(b_ + 2.5) * 9, DY - math.cos(b_ + 2.5) * 9),
               (DX + math.sin(b_ - 2.5) * 9, DY - math.cos(b_ - 2.5) * 9)]
        md.polygon(tri, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
        sb = 50 if mpp * MW > 200 else 25                                 # scale bar
        md.rectangle((14, MH - 30, 14 + sb / mpp, MH - 25), fill=(255, 255, 255, 230))
        md.text((14, MH - 48), f"{sb} m", font=FT, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        md.text((MW - 24, 34), "N", font=FB, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        arrow(md, MW - 19, 66, 0, 18, (255, 255, 255, 255), 2)
        md.text((MW - 230, MH - 18), hud.get("map_attribution", "Imagery: Esri, Maxar, Earthstar Geographics"), font=_font(10), fill=(220, 220, 220, 200))
        canvas.paste(mp, (MX, MY))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle((MX, MY, MX + MW, MY + 28), fill=(0, 0, 0))
        lat, lon = sess.en_to_latlon(E0, N0)
        dr.text((MX + 20, MY + 6), f"MAP // NORTH UP // {lat:.5f}, {lon:.5f}", font=FB, fill=GREEN)
        dr.line((MX, MY, MX + MW, MY), fill=(45, 45, 45), width=2)
        # ---- target table
        ty = VH + 6
        cols_x = (16, 70, 210, 330, 450, 580)
        for x, s in zip(cols_x, ("ID", "CLASS", "RANGE", "GROUND", "VELOCITY", "HEADING")):
            dr.text((x, ty), s, font=FT, fill=GREY)
        dr.line((10, ty + 20, VW - 10, ty + 20), fill=(60, 60, 60))
        for k, r in enumerate(sorted(active, key=lambda r: r.range)[:4]):
            y0 = ty + 26 + k * 26
            col = COLORS.get(r.grp, GREY)
            dr.rectangle((cols_x[0] - 8, y0 + 4, cols_x[0] - 2, y0 + 14), fill=col)
            dr.text((cols_x[0], y0), str(int(r.id)), font=FS, fill=WHITE)
            dr.text((cols_x[1], y0), NAMES.get(r.cls, r.cls.upper()), font=FS, fill=(220, 220, 220))
            dr.text((cols_x[2], y0), f"{r.range:6.1f} m", font=FS, fill=WHITE)
            dr.text((cols_x[3], y0), f"{r.D:6.1f} m", font=FS, fill=(200, 200, 200))
            still = not r.moving
            dr.text((cols_x[4], y0), "0 km/h" if still else f"{r.speed_kmh:.0f} km/h", font=FS, fill=(200, 200, 200))
            if not still:
                arrow(dr, cols_x[5] + 8, y0 + 9, r.rel_dir, 13, col, 2)
                dr.text((cols_x[5] + 24, y0), f"{(r.dir + 360) % 360:3.0f}°", font=FS, fill=(200, 200, 200))
        dr.text((900, VH + 6), f"altitude source: {src.replace('_', ' ')}", font=FT, fill=(110, 110, 110))
        dr.text((900, VH + 26), "Xo right, Yo down, Zo along the camera axis", font=FT, fill=(110, 110, 110))
        dr.text((900, VH + 46), "arrows: up = away from the drone", font=FT, fill=(110, 110, 110))
        vw.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        if (i - f0) % 500 == 0:
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
