"""Lens calibration from a ChArUco take: the decoded lossless PNG frames (upright 2048 x 1500, the application's own
decoder), OpenCV's model (dist = k1, k2, p1, p2, k3).

Passes: detect the board in every frame, calibrate, drop the views whose RMS error is over 3x the median, calibrate
again. Variants for comparison: k1 + k2 only (no tangential, no k3), and k1 only with the principal point fixed at
the image centre (the model of the in-flight self-calibration). The result is written in the form the data package
asks for (width, height, fx, fy, cx, cy, dist, rms_px) plus the variants and the standard deviations.
"""
import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

BOARD = {"kind": "charuco", "squares": [24, 17], "square_mm": 30.0, "marker_mm": 22.0, "dict": "DICT_5X5_1000"}


def make_board(spec=BOARD):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dict"]))
    return cv2.aruco.CharucoBoard(tuple(spec["squares"]), spec["square_mm"] / 1000.0, spec["marker_mm"] / 1000.0, d)


def iter_frames(src):
    """(name, BGR image) from a tar of PNGs, a folder of PNGs, or a video (every frame)"""
    src = Path(src)
    if src.suffix == ".tar":
        with tarfile.open(src) as tf:
            for m in sorted((m for m in tf.getmembers() if m.name.lower().endswith(".png")), key=lambda m: m.name):
                buf = np.frombuffer(tf.extractfile(m).read(), np.uint8)
                yield Path(m.name).name, cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif src.is_dir():
        for p in sorted(src.glob("*.png")):
            yield p.name, cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(src)); k = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            yield f"frame_{k:05d}", im; k += 1


def detect_all(src, board, min_corners=12, workers=8, log=print):
    det_params = cv2.aruco.CharucoParameters()
    det_params.minMarkers = 2

    def one(item):
        name, im = item
        det = cv2.aruco.CharucoDetector(board, det_params)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cc, ci, mc, mi = det.detectBoard(g)
        if ci is None or len(ci) < min_corners:
            return name, None, None, g.shape[::-1]
        obj, img = board.matchImagePoints(cc, ci)
        return name, obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32), g.shape[::-1]

    names, objs, imgs, size = [], [], [], None
    with ThreadPoolExecutor(workers) as ex:
        for name, o, p, sz in ex.map(one, iter_frames(src)):
            size = sz
            if o is None:
                continue
            names.append(name); objs.append(o); imgs.append(p)
    log(f"board found in {len(names)} frames, {sum(len(p) for p in imgs)} corners")
    return names, objs, imgs, size


def _calibrate(objs, imgs, size, flags, K0=None, d0=None):
    r = cv2.calibrateCameraExtended(objs, imgs, size, None if K0 is None else K0.copy(), None if d0 is None else d0.copy(), flags=flags,
                                    criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 40, 1e-9))
    rms, K, dist, rvecs, tvecs, std_int, std_ext, per_view = r
    return dict(rms=float(rms), K=K, dist=dist.ravel(), std=std_int.ravel(), per_view=per_view.ravel())


def calibrate(src, camera_name, out_path, board_spec=BOARD, log=print):
    board = make_board(board_spec)
    names, objs, imgs, (W, H) = detect_all(src, board, log=log)
    full = _calibrate(objs, imgs, (W, H), 0)
    med = np.median(full["per_view"])
    keep = full["per_view"] <= 3 * med
    log(f"full model pass 1: rms {full['rms']:.3f} px; dropping {int((~keep).sum())} views over 3x the median view error ({med:.3f} px)")
    objs_k = [o for o, k in zip(objs, keep) if k]; imgs_k = [p for p, k in zip(imgs, keep) if k]
    full = _calibrate(objs_k, imgs_k, (W, H), 0)
    k12 = _calibrate(objs_k, imgs_k, (W, H), cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    K0 = np.array([[full["K"][0, 0], 0, W / 2 - 0.5], [0, full["K"][0, 0], H / 2 - 0.5], [0, 0, 1.0]])
    k1c = _calibrate(objs_k, imgs_k, (W, H), cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
                     cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3, K0, np.zeros(5))
    K, d, s = full["K"], full["dist"], full["std"]
    out = {
        "camera": camera_name,
        "width": int(W), "height": int(H),
        "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "dist": [float(x) for x in d[:5]],
        "rms_px": full["rms"],
        "image": f"{W}x{H}, the application's own decode, upright (as in the delivered video); for the sensor's own orientation "
                 f"(turned 180 deg): cx -> {W} - cx, cy -> {H} - cy, p1 and p2 change sign",
        "model": "OpenCV pinhole + dist (k1, k2, p1, p2, k3)",
        "board": board_spec,
        "views_used": int(keep.sum()), "views_found": len(names), "corners_used": int(sum(len(p) for p in imgs_k)),
        "per_view_rms_median_px": float(np.median(full["per_view"])),
        "std": {"fx": float(s[0]), "fy": float(s[1]), "cx": float(s[2]), "cy": float(s[3]), "k1": float(s[4]), "k2": float(s[5]),
                "p1": float(s[6]), "p2": float(s[7]), "k3": float(s[8])},
        "fov_h_deg": float(2 * np.degrees(np.arctan(W / 2 / K[0, 0]))), "fov_v_deg": float(2 * np.degrees(np.arctan(H / 2 / K[1, 1]))),
        "variants": {
            "k1_k2": {"fx": float(k12["K"][0, 0]), "fy": float(k12["K"][1, 1]), "cx": float(k12["K"][0, 2]), "cy": float(k12["K"][1, 2]),
                      "dist": [float(x) for x in k12["dist"][:5]], "rms_px": k12["rms"]},
            "k1_centre": {"f": float(k1c["K"][0, 0]), "cx": float(k1c["K"][0, 2]), "cy": float(k1c["K"][1, 2]), "k1": float(k1c["dist"][0]),
                          "rms_px": k1c["rms"], "note": "the in-flight self-calibration's model (one radial term, centred principal point)"},
        },
        "excluded_views": [n for n, k in zip(names, keep) if not k],
    }
    json.dump(out, open(out_path, "w"), indent=1)
    log(f"{camera_name}: fx {out['fx']:.1f} fy {out['fy']:.1f} cx {out['cx']:.1f} cy {out['cy']:.1f} dist {np.round(d[:5], 4).tolist()} "
        f"rms {out['rms_px']:.3f} px ({out['views_used']} views); k1-only centred: f {k1c['K'][0, 0]:.1f} k1 {k1c['dist'][0]:+.4f} rms {k1c['rms']:.3f}")
    return out
