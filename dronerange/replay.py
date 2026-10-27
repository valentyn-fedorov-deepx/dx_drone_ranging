"""Real-time simulation (`replay`): the recording plays at its own pace and every frame goes through what the unit would
run live, one frame at a time, with the past only: detection (YOLO + ByteTrack on the GPU, or the cached detections),
the live height and heading, ranging, the tracker, and the presentation frame, shown at once. A frame the chain cannot
keep up with is dropped, as live. Detection (GPU) and the height (CPU) of a frame run side by side, as a device would
pipeline them; the clock starts after the first frame (the detector's warm-up). No stabilisation (the stabilised path
looks ahead). The window title shows what each stage costs and how far the chain is behind real time; a summary is
logged at the end.
"""
import math
import time
import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from . import demo as D
from .hud import HudRenderer, nice_rmax
from .live_height import LiveHeight
from .ranging import annotate_tracks, footprint, range_frame
from .worldtrack import WorldTracker

STAGES = ("detect", "height", "track", "draw")


class _Window:
    """a Qt window that shows the frames (OpenCV here is built without windows); keys: space pause, q / Esc quit"""

    def __init__(self, title, size):
        from PySide6 import QtWidgets
        from .gui import Canvas, application
        self.app = application()
        keys = self.keys = []

        class Main(QtWidgets.QMainWindow):
            def keyPressEvent(self, ev):
                keys.append(ev.text().lower() if ev.text() else ev.key())

        self.w = Main()
        self.c = Canvas()
        self.w.setCentralWidget(self.c)
        self.w.resize(*size)
        self.w.setWindowTitle(title)
        self.w.show()
        self.app.processEvents()

    def show(self, bgr, title):
        self.c.set_image(bgr)
        self.w.setWindowTitle(title)
        self.app.processEvents()

    def wait(self, seconds):
        """sleep while keeping the window alive"""
        end = time.perf_counter() + seconds
        while True:
            self.app.processEvents()
            left = end - time.perf_counter()
            if left <= 0:
                return
            time.sleep(min(left, 0.01))

    def key(self):
        return self.keys.pop(0) if self.keys else None

    @property
    def open(self):
        return self.w.isVisible()


class LiveYaw:
    """heading model for the session: the gyro yaw plus the estimator's current correction"""

    def __init__(self, est):
        self.est = est

    def yaw(self, t):
        return self.est.yaw(t)

    def yaws(self, t):
        return np.array([self.est.yaw(x) for x in np.atleast_1d(t)])


class IdentityView:
    """the camera picture scaled to the presentation view, no turn (the stabiliser's interface)"""

    class _Rv:
        def __init__(self, sess):
            self.sess = sess

        def __getitem__(self, i):
            return self.sess.R_wc(self.sess.ft[int(i)])

    def __init__(self, sess, out=(D.CAM_W, D.CAM_H)):
        self.sess, (self.ow, self.oh) = sess, out
        self.W, self.H = sess.cam.width, sess.cam.height
        self.Rv = self._Rv(sess)

    def warp(self, picture, i):
        return cv2.resize(picture, (self.ow, self.oh), interpolation=cv2.INTER_AREA)

    def to_view(self, i, pts):
        return np.atleast_2d(np.asarray(pts, float)) * [self.ow / self.W, self.oh / self.H]

    def to_camera(self, i, pts):
        return np.atleast_2d(np.asarray(pts, float)) * [self.W / self.ow, self.H / self.oh]


class StreamRenderer(HudRenderer):
    """rows and detection boxes arrive frame by frame"""

    def __init__(self, sess, track, mosaic, hud, drone_name):
        super().__init__(sess, pd.DataFrame(columns=["x1", "y1", "x2", "y2"]), pd.DataFrame(columns=["frame"]), track, mosaic, hud, drone_name)
        self.frame_rows, self.boxes = {}, {}

    def rows(self, i):
        return [r._replace(moving=bool(r.moving and r.moving_s >= self.min_mv.get(r.grp, 0.0))) for r in self.frame_rows.get(i, [])]

    def label_box(self, i, r):
        if r.status in ("seen", "tentative") and r.det_row >= 0:
            return self.boxes.get(int(r.det_row))
        return None

    def _centre_cand(self, i):
        if i not in self._cand:
            t = self.sess.ft[i]
            h = float(self.track.h.values[i]); h = h if np.isfinite(h) else 30.0
            E0, N0 = self.sess.drone_EN(t, smooth=True)
            Gf = footprint(self.sess.cam, self.sess.R_wc(t), h, nice_rmax(h))
            self._cand[i] = np.array([E0 + 0.5 * Gf[:, 0].mean(), N0 + 0.5 * Gf[:, 1].mean()])
        return self._cand[i]


class Detector:
    """YOLO + ByteTrack frame by frame (the detect step's settings)"""

    def __init__(self, P):
        from ultralytics import YOLO
        self.m, self.P = YOLO(P["weights"]), P

    def __call__(self, im, k):
        P = self.P
        r = self.m.track(im, imgsz=P["imgsz"], conf=P["conf"], classes=P["classes"], persist=True, verbose=False,
                         tracker=P["tracker"], half=True)[0]
        rows = []
        if r.boxes is not None and len(r.boxes):
            b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.cls.cpu().numpy().astype(int); s = r.boxes.conf.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else -np.ones(len(b), int)
            rows = [(k, int(ii), self.m.names[int(cc)], float(ss), *map(float, bb)) for bb, cc, ss, ii in zip(b, c, s, ids)]
        return pd.DataFrame(rows, columns=["frame", "track", "cls", "conf", "x1", "y1", "x2", "y2"])


def _unranged(det, start_row):
    """no height yet: the detections still show, without numbers"""
    out = []
    for n, o in enumerate(det.itertuples()):
        out.append(dict(frame=int(o.frame), id=-1, cls=o.cls, grp="person" if o.cls == "person" else ("bicycle" if o.cls == "bicycle" else "vehicle"),
                        status="tentative", det_row=start_row + n, moving=False, moving_s=0.0, range=np.nan, Xo=np.nan, Yo=np.nan,
                        Zo=np.nan, speed=0.0, speed_kmh=0.0, rel_dir=0.0, dir=0.0, E=np.nan, N=np.nan, D=np.nan, lat=np.nan, lon=np.nan))
    return out


def run(cfg, sess, mosaic, start=0, end=None, speed=1.0, detect="live", dets=None, window=(1600, 900), log=print, out_csv=None):
    end = sess.n_frames - 1 if end is None else min(end, sess.n_frames - 1)
    D.SS = 1                                                         # live: no supersampling of the overlays (speed)
    est = LiveHeight(sess, {**(cfg["altitude"].get("live") or {}), "feature_scale": 0.5, "features": 1500})
    est.P.pop("use", None)
    sess.heading = LiveYaw(est)
    wt = WorldTracker(sess, cfg.get("track"))
    track = types.SimpleNamespace(h=types.SimpleNamespace(values=np.full(sess.n_frames, np.nan)),
                                  src=types.SimpleNamespace(values=np.array(["live"] * sess.n_frames, dtype=object)))
    hr = StreamRenderer(sess, track, mosaic, cfg.get("hud", {}), cfg.get("drone_name", "DRONE"))
    dm = D.Demo(hr, IdentityView(sess), None, None, {"captions": False}, log)
    detector = Detector(cfg["detect"]) if detect == "live" else None
    cached = {f: g for f, g in dets.groupby("frame")} if dets is not None else {}
    vid = cv2.VideoCapture(cfg["data"]["video"])
    vid.set(cv2.CAP_PROP_POS_FRAMES, start)
    name = "dronerange live"
    win = _Window(name, window)
    ft = sess.ft
    stats, dropped, processed, lag_max = defaultdict(list), 0, 0, 0.0
    next_row = 0
    pool = ThreadPoolExecutor(1)                                     # detection on the GPU while the height runs on the CPU
    per_frame = []                                                   # what the live run had at each processed frame
    wall0 = None
    k = start
    paused_at = None
    while k <= end:
        target = (ft[k] - ft[start]) / speed
        now = time.perf_counter() - wall0 if wall0 is not None else 0.0
        if wall0 is not None and now > target + 1.5 * 0.2 / speed and k < end:   # behind real time: drop this frame, as live
            vid.grab(); k += 1; dropped += 1
            continue
        if now < target:
            win.wait(target - now)
        if not win.open:
            break
        ok, im = vid.read()
        if not ok:
            break
        ta = time.perf_counter()
        empty = pd.DataFrame(columns=["frame", "track", "cls", "conf", "x1", "y1", "x2", "y2"])
        fut = pool.submit(lambda: (detector(im, k), time.perf_counter())) if detector is not None else None
        est.step(k, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        t_h = time.perf_counter()
        det, t_d = fut.result() if fut is not None else (cached.get(k, empty), ta)
        tb, tc = ta + (t_d - ta), ta + (t_h - ta)                    # detect and height ran side by side
        tc_all = max(t_d, t_h)
        h = est.h
        track.h.values[k] = h if h is not None else np.nan
        det = det.reset_index(drop=True)
        det.index = np.arange(next_row, next_row + len(det))
        for o in det.itertuples():
            hr.boxes[int(o.Index)] = (o.x1, o.y1, o.x2, o.y2)
        if h is not None and len(det):
            R = pd.DataFrame(range_frame(sess, k, det, h), index=det.index)
            Rwc = sess.R_wc(ft[k]); E0, N0 = sess.drone_EN(ft[k], smooth=True)
            poly = [(E0 + e, N0 + n) for e, n in footprint(sess.cam, Rwc, h, 300.0)]
            rows = wt.step(k, float(ft[k]), R, poly)
            rows = list(annotate_tracks(sess, pd.DataFrame(rows), track).itertuples(index=False)) if rows else []
        elif h is not None:
            Rwc = sess.R_wc(ft[k]); E0, N0 = sess.drone_EN(ft[k], smooth=True)
            rows = wt.step(k, float(ft[k]), None, [(E0 + e, N0 + n) for e, n in footprint(sess.cam, Rwc, h, 300.0)])
            rows = list(annotate_tracks(sess, pd.DataFrame(rows), track).itertuples(index=False)) if rows else []
        else:
            rows = list(pd.DataFrame(_unranged(det, next_row)).itertuples(index=False)) if len(det) else []
        next_row += len(det)
        hr.frame_rows[k] = rows
        for old in [f for f in hr.frame_rows if f < k - 40]:
            hr.frame_rows.pop(old)
        td = time.perf_counter()
        comp = dm.frame(k, "rgb", None, None, image=im)
        te = time.perf_counter()
        if wall0 is None:                                            # the clock starts once the first frame is through
            wall0 = time.perf_counter() - target
        lag = time.perf_counter() - wall0 - target
        lag_max = max(lag_max, lag)
        per_frame.append(dict(frame=k, t=ft[k] - ft[0], h=h if h is not None else np.nan, heading_c=est.c, detections=len(det),
                              labelled=sum(1 for r in rows if r.status in ("seen", "tentative") and r.det_row >= 0),
                              confirmed=sum(1 for r in rows if r.status == "seen"), detect_ms=1000 * (tb - ta), height_ms=1000 * (tc - ta),
                              track_ms=1000 * (td - tc_all), draw_ms=1000 * (te - td), lag_s=max(lag, 0.0), dropped_so_far=dropped))
        if processed > 0:                                            # the first frame carries the detector's warm-up
            for st, v in zip(STAGES + ("frame",), (tb - ta, tc - ta, td - tc_all, te - td, te - ta)):
                stats[st].append(v)
        processed += 1
        win.show(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR),
                 f"{name}  |  frame {k}  |  detect {1000 * (tb - ta):.0f} ms  height {1000 * (tc - ta):.0f} ms (side by side)  "
                 f"track {1000 * (td - tc_all):.0f} ms  draw {1000 * (te - td):.0f} ms  |  frame {1000 * (te - ta):.0f} ms of {200 / speed:.0f}  |  "
                 f"behind {max(lag, 0):.2f} s  |  dropped {dropped}  |  height {'%.1f m' % h if h is not None else 'unknown'}")
        key = win.key()
        if key in ("q", "\x1b"):
            break
        if key == " ":                                               # pause: the clock stops
            paused_at = time.perf_counter()
            while win.open and win.key() != " ":
                win.wait(0.05)
            wall0 += time.perf_counter() - paused_at
        if not win.open:
            break
        k += 1
    win.w.close()
    pool.shutdown(wait=False)
    if out_csv:
        pd.DataFrame(per_frame).to_csv(out_csv, index=False)
    med = {st: 1000 * float(np.median(v)) for st, v in stats.items() if v}
    p95 = {st: 1000 * float(np.percentile(v, 95)) for st, v in stats.items() if v}
    log(f"replay: frames {start}-{k} at {speed}x: processed {processed}, dropped {dropped}; per frame (median / p95 ms): "
        + ", ".join(f"{st} {med[st]:.0f} / {p95[st]:.0f}" for st in STAGES + ("frame",) if st in med)
        + f" (budget {200 / speed:.0f} ms; detect and height run side by side)"
        + f"; max lag {lag_max:.2f} s; height first known at frame {next((f for f in est.processed if np.isfinite(track.h.values[f])), None)}")
    return med
