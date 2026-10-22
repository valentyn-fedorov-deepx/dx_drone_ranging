"""Demo video: detections with ground distance on the frame + a drone-centred radar (dashcam Mode 2 style).

Radar: drone at the centre, the camera's ground heading up (compass-independent), range rings scaled to the height,
the camera footprint on the ground as a wedge, people / vehicles as dots, a target list (class, ground distance,
slant distance, relative bearing). Distances are smoothed per track (median of the last 5 frames; the history is
reset when the distance jumps by more than 25 %, i.e. the tracker gave the id to another object). In hover a track
missed by the detector is kept for up to 4 frames, unless the camera turned or the object is already drawn.
"""
import math
import os
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VW, VH = 1280, 938                       # video area
PW = 560                                 # panel width
CW, CH = VW + PW, VH
BG = (12, 14, 12); GREEN = (60, 230, 90); GDIM = (25, 90, 40); ORANGE = (255, 165, 40); CYAN = (90, 220, 255); GREY = (170, 170, 170)
CLS = {"person": ("PERSON", GREEN), "car": ("VEHICLE", ORANGE), "truck": ("VEHICLE", ORANGE), "bus": ("VEHICLE", ORANGE),
       "motorcycle": ("VEHICLE", ORANGE), "bicycle": ("BICYCLE", CYAN)}
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


def _iou(p, q):
    ix = max(0.0, min(p.x2, q.x2) - max(p.x1, q.x1)); iy = max(0.0, min(p.y2, q.y2) - max(p.y1, q.y1))
    inter = ix * iy
    return inter / ((p.x2 - p.x1) * (p.y2 - p.y1) + (q.x2 - q.x1) * (q.y2 - q.y1) - inter + 1e-9)


def ground_polygon(cam, Rwc, h, rmax, px):
    """image points -> ground (E, N) offsets, clipped to the radar range; rays above the horizon go to the rim"""
    b = cam.bearings(px) @ Rwc.T
    out = []
    for v in b:
        if v[2] < -1e-4:
            lam = h / -v[2]; e, n = v[0] * lam, v[1] * lam
            r = math.hypot(e, n)
            if r > rmax:
                e, n = e / r * rmax, n / r * rmax
        else:
            hz = math.hypot(v[0], v[1]); e, n = v[0] / hz * rmax, v[1] / hz * rmax
        out.append((e, n))
    return np.array(out)


def render(sess, R, track, out, f0=0, f1=10 ** 9, fps=5.0, drone_name="DRONE", log=print):
    cam, ft, ti, toff = sess.cam, sess.ft, sess.ti, sess.toff
    FB, FS, FT, FL = _font(17, True), _font(14), _font(13), _font(16, True)
    by = {k: g for k, g in R.groupby("frame")}
    gyr = sess.gyr
    gcum = np.r_[0, np.cumsum(np.linalg.norm(gyr[1:], axis=1) * np.diff(ti))]   # accumulated |rotation| (rad), upper bound of the net angle

    def gyro_deg_between(t1, t2):
        return math.degrees(abs(np.interp(t2 + toff, ti, gcum) - np.interp(t1 + toff, ti, gcum)))

    W, H = cam.width, cam.height
    edge = np.array([[0, 0], [W // 2, 0], [W - 1, 0], [W - 1, H // 2], [W - 1, H - 1], [W // 2, H - 1], [0, H - 1], [0, H // 2]], float)
    hist = defaultdict(lambda: deque(maxlen=5))
    last = {}                                 # track -> (frame, row): short coasting through detector misses (hover only)
    vid = sess.video()
    out = str(out)
    tmp = out.replace(".mp4", "_tmp.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (CW, CH))
    sx, sy = VW / W, VH / H
    f1 = min(f1, sess.n_frames - 1)
    for i in range(f0, f1 + 1):
        im = vid.read(i)
        if im is None:
            break
        Rwc = sess.R_wc(ft[i]); h = float(track.h.values[i]); src = str(track.src.values[i])
        ax = Rwc @ OPTICAL_AXIS; cam_brg = math.degrees(math.atan2(ax[0], ax[1]))
        rmax = nice_rmax(h)
        frame = cv2.resize(im, (VW, VH), interpolation=cv2.INTER_AREA)
        canvas = Image.new("RGB", (CW, CH), BG)
        canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, 0))
        dr = ImageDraw.Draw(canvas)
        objs = []
        g = by.get(i)
        seen = set()
        if g is not None:
            for o in g.itertuples():
                if not np.isfinite(o.D) or o.conf < (0.2 if o.cls == "person" else 0.35):
                    continue
                key = o.track if o.track >= 0 else (i, o.x1)
                seen.add(key)
                last[key] = (i, o)
                if hist[key] and abs(o.D - np.median([a for a, _ in hist[key]])) > 0.25 * o.D:
                    hist[key].clear()                                # the tracker moved the id to another object
                hist[key].append((o.D, o.rel_brg))
                Ds = float(np.median([a for a, _ in hist[key]])); rb = float(np.median([b for _, b in hist[key]]))
                name, col = CLS.get(o.cls, ("OBJECT", GREY))
                objs.append((Ds, rb, name, col, o))
        if np.interp(ft[i] + toff, ti, sess.speed) < 0.5:          # hover: keep a missed track for up to 4 frames (0.8 s) ...
            for key, (j, o) in sorted(last.items(), key=lambda kv: -kv[1][0]):
                if key in seen or i - j > 4:
                    continue
                hgt = max(o.y2 - o.y1, 1.0)
                ang = math.radians(gyro_deg_between(ft[j], ft[i]))  # ... only while the image barely moves (camera rotation)
                if cam.f * ang > 0.25 * hgt:
                    continue
                near = any(((o.cls == "person") == (q[4].cls == "person")) and
                           math.hypot(0.5 * (o.x1 + o.x2 - q[4].x1 - q[4].x2), 0.5 * (o.y1 + o.y2 - q[4].y1 - q[4].y2)) < 2.0 * hgt for q in objs)
                if near or any(_iou(o, q[4]) > 0.15 for q in objs):  # already drawn under a new track id: no ghost
                    continue
                Ds = float(np.median([a for a, _ in hist[key]])); rb = float(np.median([b for _, b in hist[key]]))
                name, col = CLS.get(o.cls, ("OBJECT", GREY))
                objs.append((Ds, rb, name, col, o))
        for key in [k for k, (j, _) in last.items() if i - j > 4]:
            del last[key]
        objs.sort(key=lambda z: z[0])
        used = []
        for Ds, rb, name, col, o in objs:
            x1, y1, x2, y2 = o.x1 * sx, o.y1 * sy, o.x2 * sx, o.y2 * sy
            dr.rectangle((x1, y1, x2, y2), outline=col, width=2)
            txt = f"{name} {Ds:.0f} m"
            tw, th = dr.textbbox((0, 0), txt, font=FL)[2:]
            lx, ly = min(max(x1, 2), VW - tw - 6), y1 - th - 8
            for (ux, uy, uw, uh) in used:
                if lx < ux + uw and ux < lx + tw + 6 and ly < uy + uh and uy < ly + th + 6:
                    ly = uy - th - 8
            ly = max(ly, 40)
            used.append((lx, ly, tw + 6, th + 6))
            dr.rectangle((lx - 2, ly - 2, lx + tw + 4, ly + th + 4), fill=col)
            dr.text((lx + 1, ly), txt, font=FL, fill=(0, 0, 0))
        # header on the video
        t = ft[i] - ft[0]
        sats = int(np.interp(ft[i] + toff, ti, sess.nsat))
        dr.rectangle((0, 0, VW, 30), fill=(0, 0, 0))
        dr.text((10, 6), f"{drone_name}  //  T+{t:6.1f} s  //  ALT {h:4.1f} m AGL  //  GNSS 3D {sats} sats  //  ranging: camera ray to ground", font=FS, fill=GREY)
        # ---- radar panel
        px0 = VW; rr = 230; rcx, rcy = px0 + PW // 2, 40 + rr
        dr.line((px0, 0, px0, CH), fill=(45, 45, 45), width=2)
        dr.text((px0 + 20, 10), f"RADAR // DRONE-CENTRED // {rmax} M", font=FB, fill=GREEN)
        sc = rr / rmax
        for k in range(1, 5):
            r = rr * k / 4
            dr.ellipse((rcx - r, rcy - r, rcx + r, rcy + r), outline=GDIM, width=1)
            dr.text((rcx + 4, rcy - r - 16), f"{rmax * k // 4} m", font=FT, fill=GDIM)
        dr.line((rcx - rr, rcy, rcx + rr, rcy), fill=GDIM); dr.line((rcx, rcy - rr, rcx, rcy + rr), fill=GDIM)
        # camera footprint (image corners and edge midpoints on the ground), rotated so that the camera heading is up
        G = ground_polygon(cam, Rwc, h, rmax, edge)
        a = math.radians(cam_brg)
        xr = G[:, 0] * math.cos(a) - G[:, 1] * math.sin(a); yf = G[:, 0] * math.sin(a) + G[:, 1] * math.cos(a)
        poly = [(rcx + x * sc, rcy - y * sc) for x, y in zip(xr, yf)]
        ov = Image.new("RGBA", (CW, CH), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
        od.polygon(poly, fill=(60, 230, 90, 38), outline=(60, 230, 90, 140))
        canvas.paste(ov, (0, 0), ov); dr = ImageDraw.Draw(canvas)
        dr.polygon([(rcx, rcy - 11), (rcx - 9, rcy + 9), (rcx + 9, rcy + 9)], fill=GREEN)
        dr.text((rcx + 12, rcy + 4), "DRONE", font=FT, fill=GREEN)
        for Ds, rb, name, col, o in objs:
            if Ds > rmax:
                continue
            x = Ds * math.sin(math.radians(rb)); y = Ds * math.cos(math.radians(rb))
            X, Y = rcx + x * sc, rcy - y * sc
            rad = 5 if name == "PERSON" else 6
            dr.ellipse((X - rad, Y - rad, X + rad, Y + rad), fill=col, outline=(0, 0, 0))
        # target list
        ly = rcy + rr + 30
        dr.line((px0 + 20, ly, CW - 20, ly), fill=(60, 60, 60))
        dr.text((px0 + 30, ly + 6), "TARGET", font=FT, fill=GREY); dr.text((px0 + 200, ly + 6), "GROUND DIST", font=FT, fill=GREY)
        dr.text((px0 + 330, ly + 6), "SLANT", font=FT, fill=GREY); dr.text((px0 + 430, ly + 6), "BEARING", font=FT, fill=GREY)
        dr.line((px0 + 20, ly + 26, CW - 20, ly + 26), fill=(60, 60, 60))
        for k, (Ds, rb, name, col, o) in enumerate(objs[:8]):
            y0 = ly + 34 + k * 26
            dr.rectangle((px0 + 30, y0 + 3, px0 + 42, y0 + 15), fill=col)
            dr.text((px0 + 52, y0), name, font=FS, fill=(220, 220, 220))
            dr.text((px0 + 200, y0), f"{Ds:5.1f} m", font=FS, fill=(255, 255, 255))
            dr.text((px0 + 330, y0), f"{math.hypot(Ds, h):5.1f} m", font=FS, fill=(200, 200, 200))
            side = "R" if rb >= 0 else "L"
            dr.text((px0 + 430, y0), f"{side} {abs(rb):4.1f}°", font=FS, fill=(200, 200, 200))
        dr.text((px0 + 20, CH - 26), f"altitude source: {src.replace('_', ' ')}", font=FT, fill=(110, 110, 110))
        vw.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        if (i - f0) % 500 == 0:
            log("render", i)
    vw.release()
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", out], check=True)
        os.remove(tmp)
    except (OSError, subprocess.CalledProcessError):
        os.replace(tmp, out)                  # no ffmpeg: keep the mp4v file
    log(out)
    return out
