# dx_ranging

Distance to people and vehicles from a single camera, from geometry (no learned depth). Two parts:

- **`dronerange/`**: a drone camera with IMU + GNSS (VyzAI drone 2244). The camera ray through an object's
  ground-contact pixel meets the ground at the drone's height above ground. The camera and the camera-IMU rotation
  are calibrated from the flight itself, the height above ground is measured from the images, and the result is
  shown on a drone-centred radar in the dashcam Mode 2 style.
- **`camera/`**: a static long-range telephoto camera (110 mm, f = 31200 px, 100-1500 m): the real-time pipeline
  `lrr`, pose-based stature, GPS and OSINT evaluation, the IMU motion-baseline study and the research budgets.
  See [camera/README.md](camera/README.md).

## Drone: quick start

```bash
pip install -r requirements.txt
python -m dronerange --config configs/drone2244_20261020.yaml all
```

`all` runs `frames`, `calib`, `altitude`, `detect`, `range`, `validate` and `render`. Each step can be run on its own
(`python -m dronerange --config ... <step>`); every product goes to `work_dir` from the config:

| step | products |
|---|---|
| `frames` | `frame_times.csv` |
| `calib` | `calib.json`: f, k1, camera-IMU rotation, IMU time offset, gyro bias |
| `altitude` | `altitude_pairs.csv`, `altitude_vo_<run>.csv`, `altitude_track.csv` (height per frame + source), `altitude.png` |
| `cars` | `height_cars.csv`: parked-car width check of the height track |
| `detect` | `dets.csv` (YOLO11x + ByteTrack) |
| `range` | `ranges.csv`: ground distance, slant distance, relative and absolute bearing, ENU position per detection |
| `validate` | `validate_pilot.csv`, `validate_person_tracks.csv`, `pilot_tracks.jpg`, `validate_trilateration.csv`, `validate_summary.json` |
| `terrain` | `dem_grid.npz` (USGS 3DEP, needs internet, US only), `flat_ground_error.csv`; then `range --terrain` uses it |
| `render` | `out/<preset>.mp4` from the config presets, or `--start --end --fps --out` |

Data for the 2026-10-20 session (Google Drive folder `1L9jpJLliOZvXhjLXtuRY9K6FrLj_RjPn`): the mp4, `raw/FRAMES.tsv`
(frame times), `gnss/imu_gnss_*.csv` (or `metadata/xsens_*.bin`); the raw tars are not needed. Set `data_root` in
the config.

A new session needs a new config: data paths, `enu_origin` (take-off point), the calibration frame range (hover at
height), and either the drone's height telemetry (`altitude.telemetry`) or a height plan (`altitude.track`).

Live use: the calibration (f, k1, camera-IMU rotation, time offset) is fixed per unit, the height should come from
the flight controller (barometric or laser), and each detection costs one ray-plane intersection, so the ranging
itself adds nothing noticeable to the detector's time.

## Drone: results for the 2026-10-20 flight

Rerun of the whole pipeline from the raw session files on 2026-10-22 (the numbers match the first analysis of
2026-10-21; the code was rewritten after the original scripts were lost):

| quantity | value |
|---|---|
| frames | 3971 at 5 fps; the first frame is 29 ms after the first IMU sample |
| calibration | f = 4926.3 px (HFOV 23.5 deg, VFOV 17.3 deg), k1 = +0.058, IMU time offset -20.7 ms, residual 2.71 px median / 6.46 px p90 (150 hover pairs, 37050 points) |
| camera mount | optical axis = IMU -Z axis, about 23 deg below the drone's horizontal |
| height track | 8.0 m low hover (to 190 s), 23-24 m car park (300-450 s), 31-36 m street (470-658 s), 26 m final hover |
| height sources | GNSS + homography: 229 pairs, 172 good; vertical odometry: car park 720 pairs (537 used), final hover 340 (295 used) |
| detections | 15531 (1366 person, 14165 vehicle), YOLO11x at 2048 px |
| pilot triangle | +4.4 / +0.1 / -1.1 / +1.4 / -6.2 % at 19-29 m (5 windows in the low hover) |
| static vehicles | 34 trilaterations from the moving drone: median per-frame scatter 2.4 %, median bias 0.2 % |
| IMU heading | implied error median -13 deg, IQR -47..+35 deg: the compass is not usable, the radar does not need it |
| terrain (USGS 3DEP) | elevation 36.7-39.8 m over 200 x 400 m; objects -0.39..+0.35 m (p5-p95) from the take-off ground; flat-ground error -3.0..+1.4 % |
| car widths | the car-width height is 0.80 x the track height (IQR 0.63-0.83, 32 frames); 0.82-0.88 in the first analysis: a coarse check only, open until the drone's own altitude log is available |

About the pilot check: the only person in the low hover is the pilot (`pilot_tracks.jpg`). The GNSS distance from
the drone to the take-off point is his distance only while he stands at the take-off point; the 5 windows are the
ones picked in the first analysis. In his other tracks he walks around the car park or the box is cut by the image
border, and the difference reaches -34..+24 %.

Method, formulas, error budget and limits: [docs/drone_method.md](docs/drone_method.md).

## Layout

```
dronerange/   camera.py (camera model, ray-ground / ray-terrain), session.py (frames, IMU/GNSS, attitude, gyro Slerp),
              xsens.py (MTi log reader), calib.py, altitude.py, detect.py, ranging.py, validate.py, terrain.py,
              hud.py (radar video), __main__.py (CLI), config.py
configs/      one YAML per recording session
tests/        synthetic geometry tests (python -m pytest tests)
docs/         drone_method.md
camera/       long-range static camera work (snapshot of G:/longrange_det without data and weights)
```
