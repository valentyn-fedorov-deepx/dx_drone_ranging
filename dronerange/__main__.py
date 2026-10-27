"""python -m dronerange --config configs/<session>.yaml <command>

Commands (each reads the products of the previous ones from work_dir):
    lens       ChArUco lens calibration: --src <decoded tar | folder | video> --name <camera> --out <camera.json>
    frames     frame_times.csv (frame name -> Unix time)
    calib      calib.json: camera-IMU rotation, IMU time offset, gyro bias (+ f, k1 without a lens file; + gyro scale with one)
    altitude   altitude_pairs.csv, altitude_vo_<run>.csv, altitude_track.csv, altitude.png
    heading    heading.csv, heading_anchors.csv, heading.png: gyro yaw anchored by the visual motion direction
    cars       height_cars.csv: parked-car width check of the height track
    detect     dets.csv (YOLO + ByteTrack)
    range      ranges.csv (add --terrain to intersect the rays with the terrain model)
    track      tracks.csv, tracks.geojson: ground-plane tracker with persistent ids, velocity, direction
    validate   pilot triangle + static-vehicle trilateration (validate_*.csv, validate_summary.json, pilot_tracks.jpg)
    terrain    dem_grid.npz from USGS 3DEP + flat_ground_error.csv (needs internet, US sites only)
    render     out/<preset>.mp4 (presets in the config) or --start/--end/--fps/--out; --view <polarisation view | tour>
    polar      polarisation views of the raw frames against the application's decodes of the reference frame
               (polar_verify.json, polar_verify.jpg); --fit-colour refits the S0 colour matrix
    gui        viewer: every camera view under the HUD, click for the pixel's values and its ground point
    stabilise  stabilise.npz: a smooth virtual camera for the HUD view (hud.stabilise: true uses it)
    replay     real-time simulation: the recording at its own pace through detection, live height and heading,
               ranging, tracking and the presentation frame, frame by frame, in a window (--start --end --speed
               --detect live|cached)
    live_height altitude_live.csv: height from the unit's own data only, causally (altitude.live in the config); with
               altitude.live.use the altitude track is this estimate; altitude.reference = a telemetry block to compare
    demo       out/<demo.out>: the presentation video (title, sections of the flight, closing card) from the demo block
    all        frames, calib, altitude, heading, detect, range, track, validate, render (all presets)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from . import config as C


def _session(cfg, need_calib=True, heading=False):
    from .session import Session
    s = Session(cfg)
    if need_calib:
        p = C.work(cfg, "calib.json")
        if not os.path.exists(p):
            sys.exit(f"{p} missing: run the calib step first")
        s.attach_calib(p)
    if heading and cfg.get("heading", {}).get("use", True):
        p = C.work(cfg, "heading.csv")
        if os.path.exists(p):
            from .heading import HeadingModel
            s.heading = HeadingModel.load(p)
    return s


def cmd_lens(cfg, args, log):
    from . import lens
    if not (args.src and args.out):
        sys.exit("lens needs --src and --out")
    lens.calibrate(args.src, args.name or "camera", args.out, log=log)


def cmd_frames(cfg, args, log):
    s = _session(cfg, need_calib=False)
    s.write_frame_times(C.work(cfg, "frame_times.csv"))
    log("frames", s.n_frames, "dt median %.4f s" % np.median(np.diff(s.ft)), "first frame - first IMU sample %.3f s" % (s.ft[0] - s.ti[0]))


def cmd_calib(cfg, args, log):
    from . import calib
    from .camera import Camera
    s = _session(cfg, need_calib=False)
    lens_file = cfg["calib"].get("lens")
    if lens_file:
        cam = Camera.from_calib(json.load(open(lens_file)))
        calib.run_fixed(s, cfg["calib"], cam, C.work(cfg, "calib.json"), lens_file, log)
    else:
        calib.run(s, cfg["calib"], C.work(cfg, "calib.json"), log)


def cmd_heading(cfg, args, log):
    from . import heading as HD
    s = _session(cfg)
    Pd = pd.read_csv(C.work(cfg, "altitude_pairs.csv"))
    P = cfg.get("heading", {})
    if P.get("source") == "dji":
        from . import telemetry as TM
        dji = TM.load(cfg["data"][P.get("log", "flight_log")], P.get("t_col", "utc"), P.get("offset_s", 0.0))
        model, A, T = HD.fit_dji(s, Pd, cfg["altitude"]["good"], dji, P.get("yaw_col", "yaw_deg"), log)
    else:
        model, A, T = HD.fit(s, Pd, cfg["altitude"]["good"], P.get("window_s", 30.0), P.get("min_anchors", 5), P.get("gap_s", 20.0), log)
    T.to_csv(C.work(cfg, "heading.csv"), index=False)
    A.to_csv(C.work(cfg, "heading_anchors.csv"), index=False)
    HD.plot(C.work(cfg, "heading.png"), s, A, T)


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
    g = A.good_pairs(Pd, P["good"]) if len(Pd) else Pd
    log("GNSS+homography pairs", len(Pd), "good", len(g))
    vo = {}
    for name, r in (P.get("vo") or {}).items():
        vp = C.work(cfg, f"altitude_vo_{name}.csv")
        if args.reuse and os.path.exists(vp):
            vo[name] = pd.read_csv(vp)
        else:
            vo[name] = A.vertical_odometry(s, r["start"], r["end"], r["h_start"], r["h_end"], r.get("step", 2), log)
            vo[name].to_csv(vp, index=False)
    gs = A.smoothed_pairs(Pd, P["good"], P.get("smooth_window", 7)) if len(Pd) else None
    tel = P.get("telemetry")
    if tel and tel.get("source"):
        from . import telemetry as TM
        tel = dict(tel, df=TM.load(cfg["data"][tel["source"]], tel.get("t_col", "utc"), tel.get("offset_s", 0.0)))
    T = A.build_track(s, P.get("track") or [], gs, vo, tel)
    T.to_csv(C.work(cfg, "altitude_track.csv"), index=False)
    A.plot(C.work(cfg, "altitude.png"), s, Pd, P["good"], T, vo)
    for t0 in (100, 200, 260, 300, 400, 450, 470, 500, 600, 700, 790):
        k = int(np.argmin(np.abs(T.t.values - t0)))
        log(f"t {t0:3d}s frame {k:4d}: h {T.h.values[k]:5.1f} m ({T.src.values[k]})")


def cmd_replay(cfg, args, log):
    from . import replay as RP
    from .mapview import mosaic_for_session
    s = _session(cfg)
    mosaic = mosaic_for_session(s, (s.E_s, s.N_s), cfg.get("map", {}), C.work(cfg, "tiles"), log)
    dets = pd.read_csv(C.work(cfg, "dets.csv")) if args.detect == "cached" else None
    if args.record:                                                  # the live screen written to a video: every frame, no clock
        lv = cfg["live_video"]
        secs = [tuple(x) for x in lv["sections"]]
        out = lv.get("out", "live_view.mp4")
        out = out if os.path.isabs(out) else C.work(cfg, os.path.join("out", out))
        RP.run(cfg, s, mosaic, secs[0][0], secs[-1][1], detect=args.detect, dets=dets, log=log, out_csv=C.work(cfg, "live_view.csv"),
               record=(out, secs))
        return
    RP.run(cfg, s, mosaic, args.start or 0, args.end, args.speed, args.detect, dets, log=log, out_csv=C.work(cfg, "replay.csv"))


def cmd_live_height(cfg, args, log):
    from . import live_height as LH
    s = _session(cfg)
    lc = dict(cfg["altitude"].get("live") or {})
    use = lc.pop("use", False)
    H, Fx, An = LH.run(s, lc, log=log)
    H.to_csv(C.work(cfg, "altitude_live.csv"), index=False)
    Fx.to_csv(C.work(cfg, "altitude_live_fixes.csv"), index=False)
    if cfg.get("heading", {}).get("source") == "live":               # the causal heading from the same fast pairs
        HT = LH.heading_track(s, An)
        HT.to_csv(C.work(cfg, "heading.csv"), index=False)
        log(f"live heading: {len(An)} anchors; magnetic yaw error at them median "
            f"{np.median(An.e) if len(An) else float('nan'):+.1f} deg; heading.csv written")
    first = H.h.first_valid_index()
    log(f"live height: started at frame {H.frame[first] if first is not None else None} (t {H.t[first] if first is not None else float('nan'):.1f} s); "
        f"fixes: {Fx.status.value_counts().to_dict() if len(Fx) else {}}; odometry steps ok {H.vo_ok.mean() * 100:.0f} %")
    ref = cfg["altitude"].get("reference")
    if ref and (ref.get("source") or ref.get("track_csv")):
        if ref.get("source"):                                         # the aircraft's log
            from . import telemetry as TM
            D = TM.load(cfg["data"][ref["source"]], ref.get("t_col", "utc"), ref.get("offset_s", 0.0))
            href = np.interp(s.ft, D.t_unix.values, D[ref.get("h_col", "height_above_takeoff_m")].values)
        else:                                                         # another run's height track, per frame
            D = pd.read_csv(ref["track_csv"])
            href = np.interp(np.arange(s.n_frames), D.frame.values, D.h.values)
        H["h_ref"] = href[H.frame.values]
        e = 100 * (H.h / H.h_ref - 1)
        H["err_pct"] = e
        H.to_csv(C.work(cfg, "altitude_live.csv"), index=False)
        for name, a, b in [("all", 0, 10 ** 9)] + [(k, v[0], v[1]) for k, v in (ref.get("segments") or {}).items()]:
            m = (H.frame >= a) & (H.frame <= b) & H.h.notna()
            if m.any():
                log(f"  {name:12s} frames {max(a, int(H.frame[m].min()))}-{min(b, int(H.frame[m].max()))}: live vs reference median {e[m].median():+.1f} %, "
                    f"p5 {e[m].quantile(.05):+.1f}, p95 {e[m].quantile(.95):+.1f} %, |err| median {e[m].abs().median():.1f} %")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            ax[0].plot(H.t, H.h_ref, color="0.5", lw=1.5, label=ref.get("label", "reference"))
            ax[0].plot(H.t, H.h, color="C0", lw=1.2, label="live estimate (unit only)")
            if len(Fx):
                u = Fx[Fx.status.isin(["used", "start"])]
                ax[0].scatter(u.t, u.h_fix, s=8, color="C2", label="GNSS + homography fixes")
            ax[0].set_ylabel("height (m)"); ax[0].legend(); ax[0].grid(alpha=.3)
            ax[1].plot(H.t, e, color="C3", lw=1); ax[1].axhline(0, color="0.5", lw=.8)
            ax[1].set_ylabel("error (%)"); ax[1].set_xlabel("s from the first frame"); ax[1].grid(alpha=.3); ax[1].set_ylim(-30, 30)
            fig.tight_layout(); fig.savefig(C.work(cfg, "altitude_live.png"), dpi=110); plt.close(fig)
        except Exception as ex:
            log("plot:", ex)
    if use:
        T = pd.DataFrame(dict(frame=np.arange(s.n_frames), t=s.ft - s.ft[0], h=np.nan, src="live"))
        T.loc[H.frame.values, "h"] = H.h.values
        T.to_csv(C.work(cfg, "altitude_track.csv"), index=False)
        log("altitude_track.csv <- the live estimate (no height before its first fixes)")


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
    s = _session(cfg, heading=True)
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
        U = T[T.unambiguous]
        summ.update(trilateration_median_scatter_pct=float(T.mad_rel.median()), trilateration_median_abs_bias_pct=float(T.bias_rel.abs().median()),
                    heading_check_n=int(len(U)), heading_err_median_deg=float(U.yaw_err.median()) if len(U) else None)
        log(f"{len(T)} static-vehicle trilaterations (drone baseline >= {tp.get('min_span_m', 15.0)} m): median per-frame scatter "
            f"{T.mad_rel.median():.1f} %, median |bias| {T.bias_rel.abs().median():.1f} %; unambiguous for a heading check: {len(U)}"
            + (f", heading error median {U.yaw_err.median():+.1f} deg" if len(U) else ""))
    json.dump(summ, open(C.work(cfg, "validate_summary.json"), "w"), indent=1)


def cmd_track(cfg, args, log):
    from . import ranging, worldtrack
    s = _session(cfg, heading=True)
    R = pd.read_csv(C.work(cfg, "ranges.csv"))
    trk = _track(cfg)
    fp = ranging.footprints(s, trk)
    T = worldtrack.run(s, R, fp, cfg.get("track"), log)
    if not len(T):                                                    # e.g. a live run without any height: nothing ranged
        pd.DataFrame(columns=["frame", "t", "id", "cls", "grp", "status", "E", "N", "det_row"]).to_csv(C.work(cfg, "tracks.csv"), index=False)
        log("tracks.csv: no tracks (no detection has a ground position)")
        return
    T = ranging.annotate_tracks(s, T, trk)
    T.to_csv(C.work(cfg, "tracks.csv"), index=False)
    feats = []
    for tid, g in T[T.status == "seen"].groupby("id"):
        g = g.sort_values("frame")
        coords = [[round(a, 7), round(b, 7)] for a, b in zip(g.lon, g.lat)]
        feats.append({"type": "Feature",
                      "properties": {"id": int(tid), "class": g.cls.mode().iloc[0], "first_s": round(float(g.t.min()), 1),
                                     "last_s": round(float(g.t.max()), 1), "max_speed_kmh": round(float(g.speed_kmh.max()), 1)},
                      "geometry": {"type": "LineString", "coordinates": coords} if len(coords) > 1 else {"type": "Point", "coordinates": coords[0]}})
    json.dump({"type": "FeatureCollection", "features": feats}, open(C.work(cfg, "tracks.geojson"), "w"))
    st = T[T.status == "seen"].groupby("id").agg(grp=("grp", "first"), n=("frame", "size"), vmax=("speed_kmh", "max"))
    log(f"tracks.csv: {len(T)} rows, {T[T.id >= 0].id.nunique()} ids; seen in >= 10 frames: {int((st.n >= 10).sum())}; "
        f"moving (max > 5 km/h, >= 10 frames): {int(((st.vmax > 5) & (st.n >= 10)).sum())}")


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


def _hud_data(cfg, log):
    from .mapview import mosaic_for_session
    s = _session(cfg, heading=True)
    R = pd.read_csv(C.work(cfg, "ranges.csv"))
    trk = _track(cfg)
    T = pd.read_csv(C.work(cfg, "tracks.csv"))
    mosaic = mosaic_for_session(s, (T.E.values, T.N.values), cfg.get("map", {}), C.work(cfg, "tiles"), log)
    return s, R, T, trk, mosaic


def _stabiliser(cfg, s, log=print):
    """the stabilised view when hud.stabilise is on and the stabilise step has run"""
    p = C.work(cfg, "stabilise.npz")
    if not cfg.get("hud", {}).get("stabilise"):
        return None
    if not os.path.exists(p):
        log(f"hud.stabilise is on but {p} is missing (run the stabilise step): not stabilised")
        return None
    from .stabilise import Stabiliser
    return Stabiliser.load(s, p)


def cmd_stabilise(cfg, args, log):
    import cv2
    from .stabilise import Stabiliser
    s = _session(cfg, heading=True)
    sc = cfg.get("stabilise", {})
    st = Stabiliser(s, zoom=sc.get("zoom", 1.25), turn_share=sc.get("turn_share", 0.6), roll_deg=sc.get("roll_deg", 0.4), w1=sc.get("w1", 1.0))
    vid = s.video()
    small = {}

    def read(i):
        if i not in small:
            small.clear()
            small[i] = cv2.resize(vid.read(i), (1024, 750), interpolation=cv2.INTER_AREA)
        return small[i]
    st.residual(read, share=sc.get("residual_share", 0.35), w1=sc.get("w1", 1.0), log=log)
    st.save(C.work(cfg, "stabilise.npz"))
    # how much the view moves from frame to frame, raw and stabilised (phase correlation, HUD view pixels)
    win = cv2.createHanningWindow((640, 469), cv2.CV_32F)
    prev, d_raw, d_st = None, [], []
    for i in range(s.n_frames):
        im = read(i)
        g = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (640, 469), interpolation=cv2.INTER_AREA).astype(np.float32)
             for x in (cv2.resize(im, (1280, 938)), st.warp(im, i))]
        if prev is not None and s.ft[i] - s.ft[i - 1] <= 0.5:
            d_raw.append(cv2.phaseCorrelate(prev[0], g[0], win)[0]); d_st.append(cv2.phaseCorrelate(prev[1], g[1], win)[0])
        prev = g
    r0, r1 = (np.sqrt(((2 * np.array(d)) ** 2).mean(0)) for d in (d_raw, d_st))
    log(f"stabilise: zoom {st.zoom}, frame-to-frame motion of the view (rms, x / y px of the 1280 x 938 view): "
        f"raw {r0[0]:.1f} / {r0[1]:.1f}, stabilised {r1[0]:.1f} / {r1[1]:.1f}")


def _raw_store(cfg):
    from . import polar
    d = cfg["data"]
    dirs = [d[k] for k in ("raw", "reference_frame") if k in d]
    return polar.RawStore(dirs, d["frames_tsv"]), polar.load_matrix(cfg.get("polar", {}).get("s0_colour"))


def _polar_job(cfg, j, view, log):
    """frames, picture and view name for a video in a polarisation view (or a tour of all views); frames without a raw
    frame are left out"""
    import cv2
    from . import polar
    store, M = _raw_store(cfg)
    frames = [i for i in range(j["start"], j["end"] + 1) if i in store]
    if view == "tour":
        order = ["rgb"] + list(polar.VIEWS)
        n = max(1, int(round(j.get("tour_s", 6.0) * j["fps"])))
        vmap = {i: order[(k // n) % len(order)] for k, i in enumerate(frames)}
    else:
        vmap = {i: view for i in frames}
    log(f"{view}: {len(frames)} frames with a raw frame of {j['end'] - j['start'] + 1}")
    pf = polar.Prefetcher(store, M, [i for i in frames if vmap[i] != "rgb"], cfg.get("polar", {}).get("workers", 4))

    def picture(i):
        return None if vmap[i] == "rgb" else cv2.cvtColor(polar.view(vmap[i], pf.get(i)), cv2.COLOR_RGB2BGR)
    return dict(frames=frames, picture=picture, view_name=lambda i: "RGB" if vmap[i] == "rgb" else polar.VIEWS[vmap[i]]), pf


def cmd_render(cfg, args, log):
    from . import hud
    s, R, T, trk, mosaic = _hud_data(cfg, log)
    presets = cfg.get("render", {})
    jobs = []
    if args.start is not None:
        jobs.append(dict(start=args.start, end=args.end if args.end is not None else s.n_frames - 1, fps=args.fps or 5.0,
                         out=args.out or f"render_{args.start}_{args.end}{'_' + args.view if args.view else ''}.mp4",
                         view=args.view, tour_s=args.tour_s))
    else:
        names = [args.preset] if args.preset else list(presets)
        jobs += [presets[n] for n in names]
    stab = _stabiliser(cfg, s, log)
    for j in jobs:
        out = j["out"] if os.path.isabs(j["out"]) else C.work(cfg, os.path.join("out", j["out"]))
        view = j.get("view") or "rgb"
        kw, pf = ({}, None) if view == "rgb" else _polar_job(cfg, j, view, log)
        try:
            hud.render(s, R, T, trk, mosaic, out, j["start"], j["end"], j["fps"], cfg.get("hud", {}), cfg.get("drone_name", "DRONE"), log,
                       stab=stab, **kw)
        finally:
            if pf is not None:
                pf.close()


def cmd_polar(cfg, args, log):
    import cv2
    from . import polar
    ref = cfg["data"].get("reference_frame")
    if not ref:
        sys.exit("data.reference_frame missing: a folder with one raw frame and the application's decodes of it")
    path = cfg.get("polar", {}).get("s0_colour")
    if args.fit_colour:
        M = polar.fit_s0_colour(ref)
        json.dump({"matrix": np.round(M, 6).tolist(), "fitted_on": os.path.basename(os.path.normpath(ref)),
                   "what": "demosaiced polariser-cell sums / 16380 (rows R, G, B, offset) -> linear sRGB of the application's S0 view"},
                  open(path, "w"), indent=1)
        log(f"S0 colour matrix -> {path}")
    res, sheet = polar.verify(ref, polar.load_matrix(path))
    json.dump(res, open(C.work(cfg, "polar_verify.json"), "w"), indent=1)
    cv2.imwrite(C.work(cfg, "polar_verify.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    log("our views of the reference frame against the application's (levels of 255):")
    for k, v in res.items():
        log(f"  {k:16s} rms {v['rms']:5.2f} (unclipped pixels {v['rms_unclipped']:5.2f}), median |difference| {v['median_abs']:.0f}")
    store, _ = _raw_store(cfg)
    log(f"raw frames available: {len(store)} of {store.n_frames}")


def cmd_demo(cfg, args, log):
    from . import demo as D
    from .hud import HudRenderer
    from .stabilise import Stabiliser
    s, R, T, trk, mosaic = _hud_data(cfg, log)
    p = C.work(cfg, "stabilise.npz")
    if not os.path.exists(p):
        sys.exit(f"{p} missing: run the stabilise step first")
    stab = Stabiliser.load(s, p, out_size=(D.CAM_W, D.CAM_H))
    store, M = _raw_store(cfg) if "raw" in cfg["data"] else (None, None)
    dc = cfg["demo"]
    hr = HudRenderer(s, R, T, trk, mosaic, cfg.get("hud", {}), cfg.get("drone_name", "DRONE"))
    demo = D.Demo(hr, stab, store, M, dc, log)
    out = dc.get("out", "demo.mp4")
    out = out if os.path.isabs(out) else C.work(cfg, os.path.join("out", out))
    sections = dc["sections"]
    if args.only:                                                    # quick look: one section, or still frames
        sections = [sec for sec in sections if sec.get("name", sec["kind"]).lower().startswith(args.only.lower())]
    D.render(demo, sections, out, log, cfg.get("polar", {}).get("workers", 4))


def cmd_gui(cfg, args, log):
    from . import gui
    s, R, T, trk, mosaic = _hud_data(cfg, log)
    store, M = _raw_store(cfg) if "raw" in cfg["data"] else (None, None)
    if store is None:
        from .polar import RawStore
        store = RawStore([], cfg["data"]["frames_tsv"])
    log(f"viewer: {len(store)} raw frames of {s.n_frames}")
    sys.exit(gui.run(s, R, T, trk, mosaic, store, M, cfg, args.frame, args.view, cfg.get("polar", {}).get("workers", 3), _stabiliser(cfg, s, log)))


def cmd_all(cfg, args, log):
    args.reuse = False; args.terrain = False; args.preset = None; args.start = None
    for f in (cmd_frames, cmd_calib, cmd_altitude, cmd_heading, cmd_detect, cmd_range, cmd_track, cmd_validate):
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
    for name in ("frames", "calib", "heading", "cars", "detect", "track", "validate", "terrain", "stabilise", "live_height"):
        sub.add_parser(name)
    ln = sub.add_parser("lens"); ln.add_argument("--src"); ln.add_argument("--name"); ln.add_argument("--out")
    a = sub.add_parser("altitude"); a.add_argument("--reuse", action="store_true", help="reuse altitude_pairs / altitude_vo csv files")
    r = sub.add_parser("range"); r.add_argument("--terrain", action="store_true", help="intersect rays with dem_grid.npz")
    rd = sub.add_parser("render")
    rd.add_argument("preset", nargs="?"); rd.add_argument("--start", type=int); rd.add_argument("--end", type=int)
    rd.add_argument("--fps", type=float); rd.add_argument("--out")
    rd.add_argument("--view", help="camera view: rgb (the video), s0, nz, nxy, nxyz, pn_vyzlut, pn_pxdiffuse, pn_pxspecularv2 or tour")
    rd.add_argument("--tour-s", type=float, default=6.0, help="tour: seconds of video per view")
    po = sub.add_parser("polar"); po.add_argument("--fit-colour", action="store_true")
    gu = sub.add_parser("gui"); gu.add_argument("--frame", type=int); gu.add_argument("--view", default="rgb")
    de = sub.add_parser("demo"); de.add_argument("--only", help="render only the sections whose name starts with this")
    rp = sub.add_parser("replay"); rp.add_argument("--start", type=int); rp.add_argument("--end", type=int)
    rp.add_argument("--speed", type=float, default=1.0); rp.add_argument("--detect", default="live", choices=["live", "cached"])
    rp.add_argument("--record", action="store_true", help="write the live screen of the config's live_video sections to a video")
    al = sub.add_parser("all"); al.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)
    cfg = C.load(args.config, args.work)
    log = C.Log(cfg)
    log(f"## dronerange {args.cmd} ({args.config})")
    globals()["cmd_" + args.cmd](cfg, args, log)


if __name__ == "__main__":
    main()
