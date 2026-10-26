"""Viewer: the flight under the HUD in any camera view - the video, or the polarisation views of the raw frames (S0,
Nz, Nxy, Nxyz and the three physical models PN VyzLut / PxDiffuse / PxSpecularV2, computed as the VyzAI application
computes them) - with every detection labelled. Click the picture for the values behind the views at that pixel and
for the point where its ray meets the ground (distance from the drone, lat / lon).

    python -m dronerange --config configs/drone2244_20261023.yaml gui [--frame N] [--view nxyz]

Keys: 1-8 views, space play / pause, left / right one frame (in a polarisation view: one raw frame), page up / down
ten, Home / End, L label mode, B boxes, Ctrl+S save the frame as PNG.
"""
import bisect
import collections
import copy
import math
import os
import threading
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import polar
from .camera import intersect_ground, world_rays
from .hud import CH, CW, LABEL_MODES, NAMES, HudRenderer

MODES = [("rgb", "RGB")] + list(polar.VIEWS.items())
FPS = 5.0                                   # recording rate
CACHE = 48                                  # decoded raw frames kept


class Canvas(QtWidgets.QWidget):
    """the HUD frame scaled to the widget; reports clicks in HUD pixels and marks the clicked point"""
    clicked = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.qimg, self.buf, self.marker, self.note = None, None, None, None
        self.setMinimumSize(960, 540)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setCursor(QtCore.Qt.CrossCursor)

    def set_image(self, bgr):
        self.buf = np.ascontiguousarray(bgr[..., ::-1])
        h, w = self.buf.shape[:2]
        self.qimg = QtGui.QImage(self.buf.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.update()

    def geom(self):
        s = min(self.width() / CW, self.height() / CH)
        return s, (self.width() - CW * s) / 2, (self.height() - CH * s) / 2

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(12, 14, 12))
        if self.qimg is None:
            return
        s, ox, oy = self.geom()
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        p.drawImage(QtCore.QRectF(ox, oy, CW * s, CH * s), self.qimg)
        if self.marker:
            x, y = ox + self.marker[0] * s, oy + self.marker[1] * s
            for col, w in ((QtGui.QColor(0, 0, 0), 4), (QtGui.QColor(255, 60, 200), 2)):
                p.setPen(QtGui.QPen(col, w))
                for a, b, c, d in ((-14, 0, -5, 0), (5, 0, 14, 0), (0, -14, 0, -5), (0, 5, 0, 14)):
                    p.drawLine(QtCore.QPointF(x + a, y + b), QtCore.QPointF(x + c, y + d))
        if self.note:
            p.setPen(QtGui.QColor(255, 220, 90))
            p.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
            p.drawText(QtCore.QRectF(ox + 12, oy + 40 * s, 600, 30), self.note)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self.qimg is not None:
            s, ox, oy = self.geom()
            self.clicked.emit((ev.position().x() - ox) / s, (ev.position().y() - oy) / s)


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, sess, R, T, track, mosaic, store, matrix, cfg, frame=None, view="rgb", workers=3):
        super().__init__()
        hudcfg = dict(cfg.get("hud", {}))
        hudcfg["labels"] = cfg.get("gui", {}).get("labels", "full")
        self.hr = HudRenderer(sess, R, T, track, mosaic, hudcfg, cfg.get("drone_name", "DRONE"))
        self.sess, self.R, self.store, self.matrix, self.cfg = sess, R, store, matrix, cfg
        self.pool = ProcessPoolExecutor(workers)
        self.cache, self.futs = collections.OrderedDict(), {}
        self.n = sess.n_frames
        self.mode = view if view in dict(MODES) else "rgb"
        self.i = 0 if frame is None else int(frame)
        self.click = None
        self.playing, self.speed = False, 1.0
        self.setWindowTitle(f"dronerange viewer - {cfg.get('name', '')}")
        self._build()
        self.play_timer = QtCore.QTimer(self); self.play_timer.timeout.connect(self.tick)
        self.poll_timer = QtCore.QTimer(self); self.poll_timer.timeout.connect(self.poll); self.poll_timer.start(30)
        self.new_loc, self.scanning = None, False
        self.scan_timer = QtCore.QTimer(self); self.scan_timer.timeout.connect(self.rescan); self.scan_timer.start(60000)
        self.set_mode(self.mode)

    # ---- layout
    def _build(self):
        tb = self.addToolBar("views")
        tb.setMovable(False)
        self.group = QtWidgets.QButtonGroup(self)
        for k, (key, name) in enumerate(MODES):
            b = QtWidgets.QToolButton(); b.setText(f"{name}"); b.setCheckable(True); b.setToolTip(f"{name}  (key {k + 1})")
            b.setMinimumWidth(64)
            self.group.addButton(b, k); tb.addWidget(b)
            QtGui.QShortcut(QtGui.QKeySequence(str(k + 1)), self).activated.connect(lambda key=key: self.set_mode(key))
        self.group.idClicked.connect(lambda k: self.set_mode(MODES[k][0]))
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" labels "))
        self.lab = QtWidgets.QComboBox(); self.lab.addItems(LABEL_MODES); self.lab.setCurrentText(self.hr.labels)
        self.lab.currentTextChanged.connect(self.set_labels); tb.addWidget(self.lab)
        self.boxes = QtWidgets.QCheckBox("boxes"); self.boxes.setChecked(self.hr.boxes)
        self.boxes.toggled.connect(self.set_boxes); tb.addWidget(self.boxes)
        tb.addSeparator()
        self.rawinfo = QtWidgets.QLabel(); tb.addWidget(self.rawinfo)

        self.canvas = Canvas(); self.canvas.clicked.connect(self.on_click)
        bottom = QtWidgets.QWidget(); hb = QtWidgets.QHBoxLayout(bottom); hb.setContentsMargins(6, 2, 6, 4)
        self.play_btn = QtWidgets.QPushButton("Play"); self.play_btn.setFixedWidth(70); self.play_btn.clicked.connect(self.toggle_play)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, self.n - 1)
        self.slider.valueChanged.connect(self.on_slider)
        self.speed_box = QtWidgets.QComboBox(); self.speed_box.addItems(["0.5x", "1x", "2x", "4x"]); self.speed_box.setCurrentText("1x")
        self.speed_box.currentTextChanged.connect(self.set_speed)
        self.info = QtWidgets.QLabel(); self.info.setMinimumWidth(420)
        for w in (self.play_btn, self.slider, self.speed_box, self.info):
            hb.addWidget(w)
        central = QtWidgets.QWidget(); vb = QtWidgets.QVBoxLayout(central); vb.setContentsMargins(0, 0, 0, 0); vb.setSpacing(0)
        vb.addWidget(self.canvas, 1); vb.addWidget(bottom)
        self.setCentralWidget(central)

        dock = QtWidgets.QDockWidget("Point", self); dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.readout = QtWidgets.QLabel("Click the picture: the values behind the views at that pixel and where its ray meets the ground.")
        self.readout.setWordWrap(True); self.readout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.readout.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.readout.setFont(QtGui.QFont("Consolas", 10)); self.readout.setMinimumWidth(330); self.readout.setMargin(8)
        dock.setWidget(self.readout); self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        keys = (("Space", self.toggle_play), ("Right", lambda: self.step(1)), ("Left", lambda: self.step(-1)),
                ("PgDown", lambda: self.step(10)), ("PgUp", lambda: self.step(-10)), ("Home", lambda: self.goto(0)),
                ("End", lambda: self.goto(self.n - 1)), ("L", self.cycle_labels), ("B", lambda: self.boxes.toggle()),
                ("Ctrl+S", self.save_png))
        for k, fn in keys:
            QtGui.QShortcut(QtGui.QKeySequence(k), self).activated.connect(fn)
        self.resize(1600, 960)

    # ---- frames of the current view
    def frames(self):
        return None if self.mode == "rgb" else self.store.frames      # None: every frame

    def snap(self, i):
        fr = self.frames()
        if not fr:
            return min(max(i, 0), self.n - 1)
        k = bisect.bisect_left(fr, i)
        cands = [fr[j] for j in (k - 1, k) if 0 <= j < len(fr)]
        return min(cands, key=lambda f: abs(f - i))

    def neighbour(self, i, d):
        fr = self.frames()
        if not fr:
            return min(max(i + d, 0), self.n - 1)
        k = bisect.bisect_left(fr, i)
        if k < len(fr) and fr[k] == i:
            return fr[min(max(k + d, 0), len(fr) - 1)]
        return fr[min(max(k + (d if d < 0 else d - 1), 0), len(fr) - 1)]

    # ---- raw frames, decoded in worker processes
    def products(self, i, request=True):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        if request and i in self.store and i not in self.futs:
            self.futs[i] = self.pool.submit(polar.frame_products, (self.store.job(i), self.matrix))
        return None

    def prefetch(self, i):
        if self.mode == "rgb" or not len(self.store):
            return
        j = i
        for _ in range(8 if self.playing else 3):
            j2 = self.neighbour(j, 1)
            if j2 == j:
                break
            j = j2
            self.products(j)
        self.products(self.neighbour(i, -1))

    def rescan(self):
        """while an archive is still downloading, look for its new frames every minute (in a thread)"""
        if self.scanning or not any(loc[0].endswith(".partial") for loc in self.store.loc.values()):
            return
        self.scanning = True

        def work():
            st = copy.copy(self.store)
            st.scan()
            self.new_loc, self.scanning = st.loc, False
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        if self.new_loc is not None:
            self.store.loc, self.new_loc = self.new_loc, None
            self.update_rawinfo()
        done = [k for k, f in self.futs.items() if f.done()]
        for k in done:
            f = self.futs.pop(k)
            try:
                self.cache[k] = f.result()
            except Exception as e:                                    # e.g. an archive renamed when its download finished
                self.store.scan(); self.update_rawinfo()
                self.canvas.note = f"raw frame {k}: {type(e).__name__}, archives rescanned"
                continue
            while len(self.cache) > CACHE:
                self.cache.popitem(last=False)
            if k == self.i:
                self.display(self.i)

    # ---- display
    def display(self, i):
        self.i = i
        pic, note = None, None
        if self.mode != "rgb":
            P = self.products(i)
            if P is None:
                note = "decoding the raw frame ..." if i in self.store else "no raw frame here"
            else:
                pic = cv2.cvtColor(polar.view(self.mode, P), cv2.COLOR_RGB2BGR)
        if pic is None and note and self.canvas.qimg is not None and i in self.store:
            self.canvas.note = note                                   # keep the last picture until the frame arrives
            self.canvas.update()
            self.update_info()
            return
        if self.mode == "rgb" or pic is not None:
            name = dict(MODES)[self.mode]
        else:
            name = "RGB (raw frame decoding)" if i in self.store else "RGB (no raw frame)"
        img = self.hr.compose(i, pic, name)
        if img is None:
            return
        self.canvas.note = note if pic is None and self.mode != "rgb" else None
        self.canvas.set_image(img)
        self.update_info()
        self.prefetch(i)
        if self.click is not None:
            self.update_readout()

    def update_info(self):
        self.slider.blockSignals(True); self.slider.setValue(self.i); self.slider.blockSignals(False)
        t = self.sess.ft[self.i] - self.sess.ft[0]
        h = float(self.hr.track.h.values[self.i])
        self.info.setText(f"frame {self.i} / {self.n - 1}   T+{t:6.1f} s   ALT {h:4.1f} m   raw {'yes' if self.i in self.store else 'no'}")

    def update_rawinfo(self):
        self.rawinfo.setText(f"  raw frames: {len(self.store)} of {self.n}  ")

    # ---- controls
    def set_mode(self, key):
        self.mode = key
        self.group.button([k for k, _ in MODES].index(key)).setChecked(True)
        self.update_rawinfo()
        self.display(self.snap(self.i))

    def set_labels(self, m):
        self.hr.labels = m
        self.display(self.i)

    def cycle_labels(self):
        self.lab.setCurrentText(LABEL_MODES[(LABEL_MODES.index(self.hr.labels) + 1) % len(LABEL_MODES)])

    def set_boxes(self, on):
        self.hr.boxes = on
        self.display(self.i)

    def set_speed(self, s):
        self.speed = float(s.rstrip("x"))
        if self.playing:
            self.play_timer.start(int(1000 / (FPS * self.speed)))

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.play_timer.start(int(1000 / (FPS * self.speed)))
        else:
            self.play_timer.stop()

    def tick(self):
        j = self.neighbour(self.i, 1)
        if j == self.i:
            self.toggle_play()
            return
        if self.mode != "rgb" and self.products(j) is None:
            self.prefetch(self.i)                                     # wait for the decoders
            return
        self.display(j)

    def step(self, d):
        self.display(self.neighbour(self.i, d))

    def goto(self, i):
        self.display(self.snap(i))

    def on_slider(self, v):
        self.display(self.snap(v))

    def save_png(self):
        if self.canvas.buf is None:
            return
        out = os.path.join(self.cfg["work_dir"], "out", f"view_{self.i:05d}_{self.mode}.png")
        cv2.imwrite(out, np.ascontiguousarray(self.canvas.buf[..., ::-1]))
        self.statusBar().showMessage(f"saved {out}", 6000)

    # ---- the clicked point
    def on_click(self, x, y):
        cam = self.hr.view_to_camera(x, y)
        if cam is None:
            return
        self.click = cam
        self.canvas.marker = (x, y)
        self.update_readout()
        self.canvas.update()

    def update_readout(self):
        u, v = self.click
        i = self.i
        t, Rwc, h, cam_brg, E0, N0 = self.hr.geometry(i)
        lines = [f"frame {i}   pixel ({u:.0f}, {v:.0f}) of 2048 x 1500", ""]
        dE, dN, ok = intersect_ground(world_rays(self.sess.cam, Rwc, np.array([[u, v]], float)), h)
        if ok[0]:
            D = math.hypot(dE[0], dN[0])
            lat, lon = self.sess.en_to_latlon(E0 + dE[0], N0 + dN[0])
            lines += ["GROUND POINT (ray meets the ground)",
                      f"  distance   {D:7.1f} m from the drone",
                      f"  slant      {math.hypot(D, h):7.1f} m",
                      f"  bearing    {math.degrees(math.atan2(dE[0], dN[0])) % 360:7.1f} deg",
                      f"  lat / lon  {float(lat):.6f}, {float(lon):.6f}",
                      f"  height     {h:7.1f} m above the ground"]
        else:
            lines += ["GROUND POINT: the ray points above the horizon"]
        P = self.products(i, request=True)
        lines.append("")
        if P is not None:
            pv = polar.pixel_values(P, min(int(u // 2), P["S0"].shape[1] - 1), min(int(v // 2), P["S0"].shape[0] - 1))
            fmt = lambda a: "(" + ", ".join(f"{c:+.2f}" for c in a) + ")"
            lines += ["POLARISATION (2 x 2 cells)",
                      f"  S0         {pv['S0']:9.0f}",
                      f"  Nz         {pv['rho']:9.3f}  (degree)",
                      f"  Nxy        {pv['phi_deg']:+8.1f} deg  (angle)",
                      f"  Nxyz       {fmt(pv['nxyz'])}",
                      f"  VyzLut     {fmt(pv['pn_vyzlut'])}",
                      f"  PxDiffuse  {fmt(pv['pn_pxdiffuse'])}",
                      f"  PxSpecular {fmt(pv['pn_pxspecularv2'])}"]
        else:
            lines += ["POLARISATION: " + ("decoding the raw frame ..." if i in self.store else "no raw frame at this frame")]
        hit = []
        for r in self.hr.rows(i):
            b = self.hr.label_box(i, r)
            if b is not None and b[0] - 8 <= u <= b[2] + 8 and b[1] - 8 <= v <= b[3] + 8:
                hit.append(r)
        for r in hit:
            name = NAMES.get(r.cls, r.cls.upper()) + ("" if r.id < 0 else f" {int(r.id)}")
            lines += ["", f"OBJECT {name}" + ("  (unconfirmed)" if r.id < 0 else ""),
                      f"  range      {r.range:7.1f} m   ground {r.D:.1f} m",
                      f"  Xo Yo Zo   {r.Xo:.2f} {r.Yo:.2f} {r.Zo:.2f} m",
                      "  speed            -  (none before the track is confirmed)" if r.id < 0 else f"  speed      {0 if not r.moving else r.speed_kmh:7.0f} km/h",
                      f"  lat / lon  {r.lat:.6f}, {r.lon:.6f}"]
        self.readout.setText("\n".join(lines))
        if P is None and i in self.store:
            QtCore.QTimer.singleShot(400, lambda: self.click is not None and self.i == i and self.update_readout())

    def closeEvent(self, ev):
        self.play_timer.stop(); self.poll_timer.stop(); self.scan_timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(ev)


def application():
    """the Qt application with a dark palette"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, col in ((QtGui.QPalette.Window, (30, 33, 30)), (QtGui.QPalette.WindowText, (225, 225, 225)),
                      (QtGui.QPalette.Base, (22, 24, 22)), (QtGui.QPalette.Text, (225, 225, 225)),
                      (QtGui.QPalette.Button, (44, 48, 44)), (QtGui.QPalette.ButtonText, (225, 225, 225)),
                      (QtGui.QPalette.Highlight, (60, 160, 80)), (QtGui.QPalette.HighlightedText, (255, 255, 255))):
        pal.setColor(role, QtGui.QColor(*col))
    app.setPalette(pal)
    return app


def run(sess, R, T, track, mosaic, store, matrix, cfg, frame=None, view="rgb", workers=3):
    app = application()
    w = Viewer(sess, R, T, track, mosaic, store, matrix, cfg, frame, view, workers)
    w.show()
    return app.exec()
