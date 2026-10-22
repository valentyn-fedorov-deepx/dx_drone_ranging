"""People and vehicles with YOLO + ByteTrack over the whole video (frames read in order, tracker state persists).
Output columns: frame, track (-1 = not yet confirmed by the tracker), cls, conf, x1, y1, x2, y2 (video pixels)."""
import time

import numpy as np
import pandas as pd


def run(sess, P, log=print):
    from ultralytics import YOLO
    m = YOLO(P["weights"])
    vid = sess.video()
    rows, t0 = [], time.time()
    for k in range(sess.n_frames):
        im = vid.read(k)
        if im is None:
            break
        r = m.track(im, imgsz=P["imgsz"], conf=P["conf"], classes=P["classes"], persist=True, verbose=False, tracker=P["tracker"])[0]
        if r.boxes is not None and len(r.boxes):
            b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.cls.cpu().numpy().astype(int); s = r.boxes.conf.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else -np.ones(len(b), int)
            for bb, cc, ss, ii in zip(b, c, s, ids):
                rows.append((k, ii, m.names[cc], float(ss), *map(float, bb)))
        if k % 500 == 0:
            log(k, f"{time.time() - t0:.0f}s")
    D = pd.DataFrame(rows, columns=["frame", "track", "cls", "conf", "x1", "y1", "x2", "y2"])
    log("detections", len(D), f"{time.time() - t0:.0f}s")
    return D
