# dx_drone_ranging

Distance, position and motion of people and vehicles seen from a VyzAI drone camera (drones 2244 and 2155), from
geometry only (no learned depth). The camera ray through an object's ground point meets the ground at the drone's
height above ground; the lens comes from a ChArUco calibration, the attitude from the IMU in the camera unit, the
height and heading from the aircraft's own log when it is available. A ground-plane tracker gives every object a
persistent id, a velocity and a direction; the video shows VyzAI-style labels on every detection, a drone-centred radar
(dashcam Mode 2 style) and a satellite map with the objects by id. The camera view can also be one of the polarisation
views of the raw frames (S0, Nz, Nxy, Nxyz and the physical models, as the VyzAI application computes them), in the
videos and in a viewer where a click gives the pixel's values and its ground point.

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
| `render` | `out/<preset>.mp4` (1920 x 1080: video + labels, radar, satellite map, target table) from the config presets, or `--start --end --fps --out`; labels only, no detection boxes (`hud.boxes: true` draws them); `--view nxyz` (or any polarisation view, or `tour`) for the raw frames |
| `polar` | the polarisation views of the reference raw frame against the application's own decodes of it: `polar_verify.json`, `polar_verify.jpg`; `--fit-colour` refits the S0 colour matrix |
| `gui` | viewer: every camera view under the HUD (keys 1-8), play / slider, click for the pixel's ground point (distance, lat / lon) and polarisation values (`--frame N --view nxyz`) |
| `stabilise` | `stabilise.npz`: a smooth virtual camera for the HUD view (gyro turn + image residual inside the margin of a 1.25 zoom); `hud.stabilise: true` makes render and gui use it |

Data for the 2026-10-20 session (Google Drive folder `1L9jpJLliOZvXhjLXtuRY9K6FrLj_RjPn`): the mp4, `raw/FRAMES.tsv`
(frame times), `gnss/imu_gnss_*.csv` (or `metadata/xsens_*.bin`); the raw tars only for the polarisation views. The ChArUco takes of
both drones are in folder `1wfaxRGK_KMiOnNoZGe4MNVoYOe2kG1Rx` (use `decoded/*_decoded_1fps.tar`). Set `data_root` in
the config.

A new session needs a new config: data paths, `enu_origin` (take-off point), the lens file of the drone, the
calibration frame range (hover at height), and either the drone's height telemetry (`altitude.telemetry`) or a
height plan (`altitude.track`). With the aircraft's flight log (DJI) the config takes the height, the position and the
heading from it (`altitude.telemetry`, `position_source`, `heading.source: dji`; see the 10-23 config).

## Drone: results for the 2026-10-20 flight (drone 2244)

| quantity | value |
|---|---|
| lens (ChArUco, 2026-10-21) | fx 4843.7, fy 4839.8, cx 1057.8, cy 756.9 px, k1..k3 0.032 / 0.233 / -2.38, RMS 0.124 px (221 views); drone 2155: fx 4870.9, RMS 0.218 px |
| camera-IMU (lens fixed) | gyro scale 1.0176 (the Xsens gyro reads 1.8 % low), IMU time offset -20.7 ms, residual 2.69 px; with f freed the fit returns fx 4843.8 px, the ChArUco value |
| attitude | roll / pitch: gyro below 10 s, Xsens above (the Xsens alone is off by about 1 deg over seconds in hover) |
| heading | gyro yaw + 161 visual-GNSS anchors, residual 0 +- 1 deg; the Xsens magnetic yaw is off by up to 120 deg near the ground |
| height track | 8.0 m low hover, 26-28 m car park, 31-36 m street, 25 m final hover (from the images: GNSS height unusable). Revised 2026-10-25: the odometry used to drop pairs during climbs and put the car park at 23-24 m, its distances 17 % short |
| detections / tracks | 15531 detections; 244 track ids (11 people, 233 vehicles) where ByteTrack had 1142; 45 ids in the 2 min car park hover |
| moving test | parked vehicles shown as moving in 2.2 % of the car park frames; real traffic 20-45 km/h |
| pilot triangle | +6.4 / +1.5 / -0.7 / +0.8 / -0.3 % at 19-29 m (the last window sits on the odometry height: -7.9 % before the fix) |
| static vehicles | 33 trilaterations from the moving drone: 2.6 % per-frame scatter, 0.2 % median bias |
| terrain (USGS 3DEP) | flat: objects within -0.39..+0.35 m of the take-off ground, flat-ground error -3.0..+1.4 % |

## Drone: results for the 2026-10-23 flight (drone 2244 under a DJI Matrice 600 Pro)

The drone flew to a drain in the car park, held its place and climbed 13.1 / 16.2 / 20.0 / 25.4 / 30.2 m (barometer).
Config `configs/drone2244_20261023.yaml`; data: Drive folder `1Gb6XY5_x_GW-TTKCmbIlrM7LzVHH4vDW`.

| quantity | value |
|---|---|
| height | the aircraft's barometer (DJI flight log, `altitude.telemetry`); its clock runs 1.25 s behind the unit's (speed cross-correlation 0.92) |
| drone position | the aircraft's GNSS (`position_source: dji`): during the climbs the unit's own receiver drifted 18 m east while the aircraft held within 1 m |
| heading | the aircraft's heading + one mount constant (-176.9 deg) from 19 visual-GNSS anchors, residual 1.7 deg (MAD) |
| camera-IMU | lens fixed (ChArUco), hover at 30 m: time offset -18.9 ms, residual 3.5 px; R_bc within 0.6 deg of the 10-20 calibration (the IMU sits in the camera unit) |
| camera tilt | measured 23-24 deg below the horizon in the holds, 24.3 deg on the ground before take-off (nominal "30 deg") |
| absolute check | the near edge of the street in frame 2522: 86.4 m with the measured tilt, 82-85 m on the satellite imagery along the same ray; a fixed 30 deg would put it at 54 m. RVs across the street (130 m) land on the RV lot |
| cross-height check | parked cars keep their place across the five heights: median deviation 0.7 m along the ray, 0.25 m across it; ground distances change by 2.0 % (median) with the measured tilt, 2.8 % with a fixed 30 deg |
| tracks | 14036 detections; 59 ids (7 people, 52 vehicles) where ByteTrack had 596 |
| odometry check | the visual vertical odometry against the barometer over the climbs: -27 % with the old pair rejection, -9 % after the fix (all pairs with a homography count) |

`tools/tilt_test.py`, `tools/road_check.py` and `tools/overlay_check.py` reproduce the tilt checks.

## Labels and polarisation views (10-23)

Labels show what the system has in each frame: every detection is labelled (12814 of 14036; the rest are second boxes
on the same vehicle), a missed detection is a missing label, and a detection whose track is not confirmed yet shows
the class without an id or velocity. Radar and map: filled = detected in this frame, hollow = kept by the tracker.

With the raw frames (`raw/` L1 1 fps + L2 5 fps, 35 GB) the camera view can be a polarisation view. The views follow
the drone application; against its own decodes of frame 2522 (rms, levels of 255): S0 1.2, Nz 1.3, Nxy 13 (median 0),
Nxyz 5.8, PN PxSpecularV2 1.8, PN PxDiffuse / VyzLut 7.3 (median 1). How they are computed: docs, section 8.

```bash
python -m dronerange --config configs/drone2244_20261023.yaml gui --frame 2522 --view nxyz
python -m dronerange --config configs/drone2244_20261023.yaml render hold30_views
```

The recording is 5 fps (`session.json`: `record_fps` 5, 3018 frames written, none dropped; the camera itself runs at
30 fps): L1 + L2 hold every recorded frame. The videos are stabilised: the camera has no gimbal and its 0.36 Hz sway
moves the image by 15 / 38 px from frame to frame in the 30 m hold, 0.5 / 1.8 px after stabilisation (docs, section 7).

Method, formulas, error budget, the 45 deg mount and the limits: [docs/drone_method.md](docs/drone_method.md).

Live use: the lens and the camera-IMU calibration are fixed per unit, the height should come from the flight
controller (barometric or laser), and ranging plus tracking cost microseconds per object. Offline-only parts in this
repo: the centred smoothing windows (attitude, drone position, heading anchors) and the visual height plan; each has a
causal counterpart (`position_smoothing_causal`, a one-sided window, telemetry height).

## Layout

```
dronerange/   camera.py (lens models, ray-ground / ray-terrain), session.py (frames, IMU/GNSS, fused attitude,
              gyro Slerp, drone position), xsens.py (MTi log reader), lens.py (ChArUco), calib.py, altitude.py,
              heading.py, telemetry.py (aircraft log), detect.py, ranging.py, worldtrack.py (ground-plane tracker),
              validate.py, terrain.py, polar.py (raw frames, polarisation views), mapview.py (satellite tiles),
              hud.py (HUD frames, video), stabilise.py (stabilised view), gui.py (viewer), __main__.py (CLI),
              config.py
configs/      one YAML per recording session; cameras/ = lens calibrations (ChArUco), S0 colour matrix
tests/        synthetic tests (python -m pytest tests)
docs/         drone_method.md
tools/        tilt checks: tilt_test.py, road_check.py, overlay_check.py
```
