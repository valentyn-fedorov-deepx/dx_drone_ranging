"""python -m dronerange --config configs/<session>.yaml <command>

Commands (each reads the products of the previous ones from work_dir):
    frames     frame_times.csv (frame name -> Unix time)
    calib      calib.json: f, k1, camera-IMU rotation, IMU time offset, gyro bias
    altitude   altitude_pairs.csv, altitude_vo_<run>.csv, altitude_track.csv, altitude.png
    cars       height_cars.csv: parked-car width check of the height track
    detect     dets.csv (YOLO + ByteTrack)
    range      ranges.csv (add --terrain to intersect the rays with the terrain model)
    validate   pilot triangle + static-vehicle trilateration (validate_*.csv, validate_summary.json, pilot_tracks.jpg)
    terrain    dem_grid.npz from USGS 3DEP + flat_ground_error.csv (needs internet, US sites only)
    render     out/<preset>.mp4 (presets in the config) or --start/--end/--fps/--out
    all        calib, altitude, detect, range, validate, render (all presets)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from . import config as C


def _session(cfg, need_calib=True):
    from .session import Session
    s = Session(cfg)
    if need_calib:
        p = C.work(cfg, "calib.json")
        if not os.path.exists(p):
            sys.exit(f"{p} missing: run the calib step first")
        s.attach_calib(p)
    return s


def cmd_frames(cfg, args, log):
    s = _session(cfg, need_calib=False)
    s.write_frame_times(C.work(cfg, "frame_times.csv"))
    log("frames", s.n_frames, "dt median %.4f s" % np.median(np.diff(s.ft)), "first frame - first IMU sample %.3f s" % (s.ft[0] - s.ti[0]))


def cmd_calib(cfg, args, log):
    from . import calib
    s = _session(cfg, need_calib=False)
    calib.run(s, cfg["calib"], C.work(cfg, "calib.json"), log)


def cmd_altitude(cfg, args, log):
    from . import altitude as A
    s = _session(cfg)
    P = cfg["altitude"]
    pp = C.work(cfg, "altitude_pairs.csv")
    if args.reuse and os.path.exists(pp):
        Pd = pd.read_csv(pp)
    else:
        Pd = A.pairs(s, P["pairs"], log)
        Pd.to_csv(pp, index=False)
    g = A.good_pairs(Pd, P["good"])
    log("GNSS+homography pairs", len(Pd), "good", len(g))
    vo = {}
    for name, r in (P.get("vo") or {}).items():
        vp = C.work(cfg, f"altitude_vo_{name}.csv")
        if args.reuse and os.path.exists(vp):
            vo[name] = pd.read_csv(vp)
        else:
            vo[name] = A.vertical_odometry(s, r["start"], r["end"], r["h_start"], r["h_end"], r.get("step", 2), log)
            vo[name].to_csv(vp, index=False)
    gs = A.smoothed_pairs(Pd, P["good"], P.get("smooth_window", 7))
    T = A.build_track(s, P["track"], gs, vo, P.get("telemetry"))
    T.to_csv(C.work(cfg, "altitude_track.csv"), index=False)
    A.plot(C.work(cfg, "altitude.png"), s, Pd, P["good"], T, vo)
    for t0 in (100, 200, 260, 300, 400, 450, 470, 500, 600, 700, 790):
        k = int(np.argmin(np.abs(T.t.values - t0)))
        log(f"t {t0:3d}s frame {k:4d}: h {T.h.values[k]:5.1f} m ({T.src.values[k]})")


def _track(cfg):
    p = C.work(cfg, "altitude_track.csv")
    if not os.path.exists(p):
        sys.exit(f"{p} missing: run the altitude step first")
    return pd.read_csv(p)


def cmd_cars(cfg, args, log):
    from . import altitude as A
    s = _session(cfg)
    D = A.cars_check(s, cfg["altitude"]["cars_check"], cfg["detect"]["weights"], _track(cfg), log)
    D.to_csv(C.work(cfg, "height_cars.csv"), index=False)


def cmd_detect(cfg, args, log):
    from . import detect
    s = _session(cfg, need_calib=False)
    detect.run(s, cfg["detect"], log).to_csv(C.work(cfg, "dets.csv"), index=False)


def _terrain(cfg):
    from .terrain import Terrain
    p = C.work(cfg, "dem_grid.npz")
    if not os.path.exists(p):
        sys.exit(f"{p} missing: run the terrain step first")
    T = Terrain.load(p)
    if cfg.get("terrain", {}).get("height_reference", "takeoff") == "takeoff":
        T.z_ref = float(T.height_at(np.array([0.0]), np.array([0.0]))[0])
    return T


def cmd_range(cfg, args, log):
    from . import ranging
    s = _session(cfg)
    use_t = args.terrain or cfg.get("terrain", {}).get("use_in_ranging", False)
    R = ranging.run(s, pd.read_csv(C.work(cfg, "dets.csv")), _track(cfg), _terrain(cfg) if use_t else None, log)
    R.to_csv(C.work(cfg, "ranges.csv"), index=False)


def cmd_validate(cfg, args, log):
    from . import validate as V
    s = _session(cfg)
    R = pd.read_csv(C.work(cfg, "ranges.csv"))
    P = cfg.get("validate", {})
    pilot, tracks = V.pilot_check(R, P.get("pilot_windows", []), P.get("pilot_max_frame"))
    tracks.to_csv(C.work(cfg, "validate_person_tracks.csv"))
    pilot.to_csv(C.work(cfg, "validate_pilot.csv"), index=False)
    V.crop_sheet(C.work(cfg, "pilot_tracks.jpg"), s, R, tracks)
    pd.set_option("display.width", 200)
    log("low-hover person tracks (n>=10):\n" + tracks.round(1).to_string())
    log("pilot windows:\n" + pilot.round(1).to_string(index=False))
    tp = P.get("trilateration", {})
    T = V.trilateration(R, tp.get("min_n", 15), tp.get("min_span_m", 15.0))
    T.to_csv(C.work(cfg, "validate_trilateration.csv"), index=False)
    summ = dict(pilot_abs_err_pct=pilot.err_pct.abs().round(2).tolist() if "err_pct" in pilot else [],
                trilateration_n=int(len(T)))
    if len(T):
        summ.update(trilateration_median_scatter_pct=float(T.mad_rel.median()), trilateration_median_abs_bias_pct=float(T.bias_rel.abs().median()),
                    imu_heading_err_median_deg=float(T.yaw_err.median()),
                    imu_heading_err_iqr_deg=[float(T.yaw_err.quantile(.25)), float(T.yaw_err.quantile(.75))])
        log(f"{len(T)} static-vehicle trilaterations (drone baseline >= {tp.get('min_span_m', 15.0)} m): median per-frame scatter "
            f"{T.mad_rel.median():.1f} %, median |bias| {T.bias_rel.abs().median():.1f} %; implied IMU heading error median "
            f"{T.yaw_err.median():.1f} deg (IQR {T.yaw_err.quantile(.25):.1f}..{T.yaw_err.quantile(.75):.1f})")
    json.dump(summ, open(C.work(cfg, "validate_summary.json"), "w"), indent=1)


def cmd_terrain(cfg, args, log):
    from . import terrain as TR
    s = _session(cfg, need_calib=False)
    R = pd.read_csv(C.work(cfg, "ranges.csv"))
    E = np.r_[R.E.values, R.E_drone.values]; N = np.r_[R.N.values, R.N_drone.values]
    e0, e1 = np.nanpercentile(E, 1) - 20, np.nanpercentile(E, 99) + 20
    n0, n1 = np.nanpercentile(N, 1) - 20, np.nanpercentile(N, 99) + 20
    step = cfg.get("terrain", {}).get("grid_step_m", 10.0)
    ee, nn, Z = TR.fetch_grid(s.lat0, s.lon0, e0, e1, n0, n1, step)
    T = TR.Terrain(ee, nn, Z)
    T.save(C.work(cfg, "dem_grid.npz"))
    zt = float(T.height_at(np.array([0.0]), np.array([0.0]))[0])
    log(f"DEM grid {len(ee)} x {len(nn)} over {e1 - e0:.0f} x {n1 - n0:.0f} m, valid {np.isfinite(Z).mean():.3f}; elevation "
        f"min/median/max {np.nanmin(Z):.2f} / {np.nanmedian(Z):.2f} / {np.nanmax(Z):.2f} m; take-off point {zt:.2f} m")
    T = _terrain(cfg)
    F = TR.flat_ground_error(R, T)
    F[["frame", "track", "cls", "D", "h", "z_obj", "dz", "flat_err_pct"]].to_csv(C.work(cfg, "flat_ground_error.csv"), index=False)
    log("object ground minus reference ground: median %.2f m, p5 %.2f, p95 %.2f" % (F.dz.median(), F.dz.quantile(.05), F.dz.quantile(.95)))
    log("implied flat-ground distance error: median %+.1f %%, p5 %+.1f, p95 %+.1f" % (F.flat_err_pct.median(), F.flat_err_pct.quantile(.05), F.flat_err_pct.quantile(.95)))


def cmd_render(cfg, args, log):
    from . import hud
    s = _session(cfg)
    R = pd.read_csv(C.work(cfg, "ranges.csv"))
    T = _track(cfg)
    presets = cfg.get("render", {})
    jobs = []
    if args.start is not None:
        jobs.append(dict(start=args.start, end=args.end if args.end is not None else s.n_frames - 1, fps=args.fps or 5.0,
                         out=args.out or f"render_{args.start}_{args.end}.mp4"))
    else:
        names = [args.preset] if args.preset else list(presets)
        jobs += [presets[n] for n in names]
    for j in jobs:
        out = j["out"] if os.path.isabs(j["out"]) else C.work(cfg, os.path.join("out", j["out"]))
        hud.render(s, R, T, out, j["start"], j["end"], j["fps"], cfg.get("drone_name", "DRONE"), log)


def cmd_all(cfg, args, log):
    args.reuse = False; args.terrain = False; args.preset = None; args.start = None
    for f in (cmd_frames, cmd_calib, cmd_altitude, cmd_detect, cmd_range, cmd_validate):
        log(f"==== {f.__name__[4:]}")
        f(cfg, args, log)
    if not args.no_render:
        log("==== render")
        cmd_render(cfg, args, log)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dronerange", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--work", help="override work_dir")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("frames", "calib", "cars", "detect", "validate", "terrain"):
        sub.add_parser(name)
    a = sub.add_parser("altitude"); a.add_argument("--reuse", action="store_true", help="reuse altitude_pairs / altitude_vo csv files")
    r = sub.add_parser("range"); r.add_argument("--terrain", action="store_true", help="intersect rays with dem_grid.npz")
    rd = sub.add_parser("render")
    rd.add_argument("preset", nargs="?"); rd.add_argument("--start", type=int); rd.add_argument("--end", type=int)
    rd.add_argument("--fps", type=float); rd.add_argument("--out")
    al = sub.add_parser("all"); al.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)
    cfg = C.load(args.config, args.work)
    log = C.Log(cfg)
    log(f"## dronerange {args.cmd} ({args.config})")
    globals()["cmd_" + args.cmd](cfg, args, log)


if __name__ == "__main__":
    main()
