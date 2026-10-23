# dx_ranging

Distance to people and vehicles from a single camera, from geometry (no learned depth). Two parts:

- **`dronerange/`**: a drone camera with IMU + GNSS (VyzAI drones 2244 and 2155). The camera ray through an object's
  ground point meets the ground at the drone's height above ground. A ground-plane tracker gives every object a
  persistent id, a velocity and a direction; the video shows VyzAI-style labels, a drone-centred radar (dashcam Mode 2
  style) and a satellite map with the objects by id.
- **`camera/`**: a static long-range telephoto camera (110 mm, f = 31200 px, 100-1500 m): the real-time pipeline
  `lrr`, pose-based stature, GPS and OSINT evaluation, the IMU motion-baseline study and the research budgets.
  See [camera/README.md](camera/README.md).

## Drone: quick start

```bash
pip install -r requirements.txt
python -m dronerange --config configs/drone2244_20261020.yaml all
```

`all` runs `frames`, `calib`, `altitude`, `heading`, `detect`, `range`, `track`, `validate` and `render`. Each step
can be run on its own (`python -m dronerange --config ... <step>`); every product goes to `work_dir` from the config:

| step | products |
|---|---|
| `lens` | lens calibration from a ChArUco take: `--src <decoded tar> --name <camera> --out <camera.json>` (results in `configs/cameras/`) |
| `frames` | `frame_times.csv` |
| `calib` | `calib.json`: camera-IMU rotation, IMU time offset, gyro bias and scale (lens from `calib.lens`; without it f and k1 are self-calibrated) |
| `altitude` | `altitude_pairs.csv`, `altitude_vo_<run>.csv`, `altitude_track.csv` (height per frame + source), `altitude.png` |
| `heading` | `heading.csv`, `heading_anchors.csv`, `heading.png`: gyro yaw anchored by the visual motion direction |
| `cars` | `height_cars.csv`: parked-car width check of the height track |
| `detect` | `dets.csv` (YOLO11x + ByteTrack) |
| `range` | `ranges.csv`: ground distance, slant distance, bearings, ENU positions of the contact and tracked points |
| `track` | `tracks.csv` (per frame and track: id, class, status, E/N, lat/lon, speed, direction, Xo/Yo/Zo, range), `tracks.geojson` |
| `validate` | `validate_pilot.csv`, `validate_person_tracks.csv`, `pilot_tracks.jpg`, `validate_trilateration.csv`, `validate_summary.json` |
| `terrain` | `dem_grid.npz` (USGS 3DEP, needs internet, US only), `flat_ground_error.csv`; then `range --terrain` uses it |
| `render` | `out/<preset>.mp4` (1920 x 1080: video + labels, radar, satellite map, target table) from the config presets, or `--start --end --fps --out` |

Data for the 2026-10-20 session (Google Drive folder `1L9jpJLliOZvXhjLXtuRY9K6FrLj_RjPn`): the mp4, `raw/FRAMES.tsv`
(frame times), `gnss/imu_gnss_*.csv` (or `metadata/xsens_*.bin`); the raw tars are not needed. The ChArUco takes of
both drones are in folder `1wfaxRGK_KMiOnNoZGe4MNVoYOe2kG1Rx` (use `decoded/*_decoded_1fps.tar`). Set `data_root` in
the config.

A new session needs a new config: data paths, `enu_origin` (take-off point), the lens file of the drone, the
calibration frame range (hover at height), and either the drone's height telemetry (`altitude.telemetry`) or a
height plan (`altitude.track`).

## Drone: results for the 2026-10-20 flight (drone 2244)

| quantity | value |
|---|---|
| lens (ChArUco, 2026-10-21) | fx 4843.7, fy 4839.8, cx 1057.8, cy 756.9 px, k1..k3 0.032 / 0.233 / -2.38, RMS 0.124 px (221 views); drone 2155: fx 4870.9, RMS 0.218 px |
| camera-IMU (lens fixed) | gyro scale 1.0176 (the Xsens gyro reads 1.8 % low), IMU time offset -20.7 ms, residual 2.69 px; with f freed the fit returns fx 4843.8 px, the ChArUco value |
| attitude | roll / pitch: gyro below 10 s, Xsens above (the Xsens alone is off by about 1 deg over seconds in hover) |
| heading | gyro yaw + 161 visual-GNSS anchors, residual 0 +- 1 deg; the Xsens magnetic yaw is off by up to 120 deg near the ground |
| height track | 8.0 m low hover, 23-24 m car park, 31-36 m street, 26 m final hover (from the images: GNSS height unusable) |
| detections / tracks | 15531 detections; 252 track ids (11 people, 241 vehicles) where ByteTrack had 1142; 45 ids in the 2 min car park hover |
| moving test | parked vehicles shown as moving in 2.2 % of the car park frames; real traffic 20-45 km/h |
| pilot triangle | +6.4 / +1.5 / -0.7 / +0.8 / -7.9 % at 19-29 m |
| static vehicles | 33 trilaterations from the moving drone: 2.6 % per-frame scatter, 0.2 % median bias |
| terrain (USGS 3DEP) | flat: objects within -0.39..+0.35 m of the take-off ground, flat-ground error -3.0..+1.4 % |

Method, formulas, error budget, the 45 deg mount and the limits: [docs/drone_method.md](docs/drone_method.md).

Live use: the lens and the camera-IMU calibration are fixed per unit, the height should come from the flight
controller (barometric or laser), and ranging plus tracking cost microseconds per object. Offline-only parts in this
repo: the centred smoothing windows (attitude, drone position, heading anchors) and the visual height plan; each has a
causal counterpart (`position_smoothing_causal`, a one-sided window, telemetry height).

## Layout

```
dronerange/   camera.py (lens models, ray-ground / ray-terrain), session.py (frames, IMU/GNSS, fused attitude,
              gyro Slerp, drone position), xsens.py (MTi log reader), lens.py (ChArUco), calib.py, altitude.py,
              heading.py, detect.py, ranging.py, worldtrack.py (ground-plane tracker), validate.py, terrain.py,
              mapview.py (satellite tiles), hud.py (video), __main__.py (CLI), config.py
configs/      one YAML per recording session; cameras/ = lens calibrations (ChArUco)
tests/        synthetic tests (python -m pytest tests)
docs/         drone_method.md
camera/       long-range static camera work (snapshot of G:/longrange_det without data and weights)
```
