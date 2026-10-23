# Drone ranging and tracking: method, calibration, checks

Distance, position and motion of people and vehicles on the ground from a drone camera, from geometry only (no
learned depth): a ray from the calibrated camera through the object's ground point, turned into world coordinates
with the attitude, meets the ground at the drone's height above ground. A ground-plane tracker gives every object a
persistent id, a velocity and a direction; the result is drawn on the video, a radar and a satellite map.

## 1. Ray to the ground

- Ground point: the bottom centre of the box for people (feet); for vehicles the box-centre ray meets the plane
  0.75 m above the ground (the middle of the body: it stays put when the view of a parked car changes, the box
  bottom slides around the car). The radar and the validation use the bottom centre (contact point).
- Camera ray: the pixel undistorted with the lens model, `b_c = normalize([x, y, 1])`. With a lens file (ChArUco)
  the OpenCV model (k1, k2, p1, p2, k3, own principal point); without one the flight self-calibration's one-term model.
- World ray: `b_w = R_wb(t + toff) R_bc b_c`, `R_wb = Rz(yaw) Ry(pitch) Rx(roll)` (ZYX, ENU, the Xsens convention,
  checked against the accelerometer on the ground). Roll and pitch: gyro + Xsens fused (section 4); yaw: gyro anchored
  by the visual motion direction (section 5).
- Ground hit at height h: `lambda = h / (-b_w,z)`, `dE = b_w,x lambda`, `dN = b_w,y lambda`; ground distance
  `D = hypot(dE, dN)`, slant distance `S = hypot(D, h)`.
- Label coordinates: the tracked point in the camera frame, `[Xo, Yo, Zo] = R_wc^T [E - E_drone, N - N_drone,
  z_point - h]` (Xo right, Yo down, Zo along the optical axis), `Range = |[Xo, Yo, Zo]|` (slant distance).
- Radar bearing: relative to the camera's ground heading. The yaw rotates both directions about the vertical, so D,
  the relative bearing and the radar do not depend on the heading at all.

Error budget (depression angle alpha of the ray, `D = h / tan(alpha)`):

| error source | effect on D |
|---|---|
| height error dh | dD / D = dh / h (1:1) |
| ray angle error d_alpha | dD / D = 2 d_alpha / sin(2 alpha): 2.4 % per 0.5 deg at 23 deg, 1.7 % at 45 deg |
| object ground dz above the reference ground | dD / D = -dz / h |
| heading error | none on D and on the radar; map positions and directions of motion only |

## 2. Lens (`lens`) and camera-IMU calibration (`calib`)

**Lens from the ChArUco takes** (2026-10-21, focus as flown; board 24 x 17 squares of 30 mm, markers 22 mm,
DICT_5X5_1000; decoded lossless frames, upright 2048 x 1500). Detect, calibrate (OpenCV model), drop the views over
3x the median view error, calibrate again:

| drone | fx | fy | cx | cy | dist (k1, k2, p1, p2, k3) | RMS | views |
|---|---|---|---|---|---|---|---|
| 2244 | 4843.7 | 4839.8 | 1057.8 | 756.9 | 0.0319, 0.2332, 0.0002, 0.0009, -2.383 | 0.124 px | 221 of 228 |
| 2155 | 4870.9 | 4867.8 | 1022.6 | 732.6 | 0.0439, 0.0033, -0.0015, -0.0013, -0.244 | 0.218 px | 126 of 130 |

`configs/cameras/drone2244_charuco.json` and `drone2155_charuco.json` hold them in the form the data package asks for
(width, height, fx, fy, cx, cy, dist, rms_px, orientation), with standard deviations (fx +-0.34 px for 2244) and two
simpler variants. For the sensor's own orientation (turned 180 deg): cx -> 2048 - cx, cy -> 1500 - cy, p1 and p2 change
sign. The package's own trial calibration agrees (2244: fx 4848.6, 0.1 %).

**Camera-IMU rotation, time offset, gyro scale.** Hand-eye rotation calibration on hover pairs at height: for
consecutive frames in hover (ground speed below 0.25 m/s, frames >= 1450, gyro rotation 1-10 deg) the image motion is
the rotation measured by the bias-corrected gyro,

    b2 = R_bc^T dR^T R_bc b1,   dR = R_ref_b(t1)^T R_ref_b(t2)  (continuous Slerp of the integrated gyro)

150 pairs, SIFT + RANSAC homography inliers (up to 250 points each), start from the best of the 24 axis-aligned R_bc,
a grid over the time offset (-200..200 ms), robust least squares. With the ChArUco lens fixed the unknowns are R_bc,
the time offset and a gyro scale factor. Drone 2244: **gyro scale 1.0176, time offset -20.7 ms, residual 2.69 px**
median (6.44 px p90); with the scale held at 1 the residual is 2.87 px. Freeing the focal length as well returns
fx = 4843.8 px, the ChArUco value: image rotations about the optical axis do not depend on f, pan and tilt do, so f
and the gyro scale separate. The earlier self-calibration's f = 4926 px was the product 4844 x 1.0176: the Xsens gyro
reads 1.8 % low. The scale is applied wherever the gyro is integrated.

**Mount.** The Xsens sits in the camera unit: the optical axis is the IMU -Z axis within 1.3 deg and the IMU roll
reads the camera tilt (about -66 deg for a camera 24 deg below the horizon). Re-mounting the unit at 45 deg changes
the attitude readings, not R_bc. A minute of hover at height after mounting, with some yaw and pitch, lets `calib`
confirm R_bc and the time offset. At 45 deg the view is short and narrow:

| tilt | height | ground distance near / centre / far | width near / far |
|---|---|---|---|
| 23 deg (this flight) | 30 m | 48.6 / 70.7 / 119.3 m | 23.8 / 51.2 m |
| 45 deg | 20 m | 14.7 / 20.0 / 27.4 m | 10.3 / 14.1 m |
| 45 deg | 30 m | 22.0 / 30.0 / 41.1 m | 15.5 / 21.2 m |
| 45 deg | 40 m | 29.4 / 40.0 / 54.8 m | 20.7 / 28.3 m |

Without a lens file `calib` falls back to the self-calibration of f and k1 (principal point at the centre); it fits
the gyro as well, but its f carries the gyro's scale error.

## 3. Height above ground (`altitude`)

The GNSS height of this unit drifts by -8 to +6 m against the true height, so the height comes from the images.

**GNSS displacement + ground homography (moving drone).** For frames i, j with a few metres of horizontal travel:
ground-plane homography H between the undistorted points, camera rotation R21 from the gyro, ground normal n in
camera-1 coordinates from roll and pitch:

    R21^T K^-1 H K / sigma_2 = I + u n^T,   u = -C2 / d

(C2 = camera-2 centre in camera-1 coordinates, d = height of camera 1 above the plane). In world axes
`u_w = R_wc1 u = -dC / d`, so `d = |dC_h| / |u_w,h|` with |dC_h| from the integrated receiver velocity: only the length
of the horizontal displacement enters, not its direction. Per-pair checks: the dominant right singular vector of
`M - I` within 4 deg of the IMU normal, rank-1 residual below 0.15, over 60 inliers, at least 2 m of travel. Pairs from
hover (about 1 m of travel) are garbage. The direction of `u_w` is not used as a filter here (near the ground the
magnetic heading is off by tens of degrees, the height is unaffected); it becomes the heading anchor in section 5.

**Vertical odometry (hover).** The same model between frames two apart gives `dz / h = -u_w,z`, integrated in ln h
from an anchor height.

**Car widths (check).** Rays through the bottom corners of parked-car boxes hit the ground at h = 1 m; the 10th
percentile of those widths is the car width (1.9 m), so `h = 1.9 / w10`. A coarse scale check: it reads 0.80 x the
track height here (0.82-0.88 in the first analysis), open until the drone's own altitude log is available.

**Height track.** A per-session plan stitches the sources (config `altitude.track`): a constant for the low hover
(7.9 m from GNSS displacement pairs), vertical odometry for the climb and the car park hover, a ramp where neither
method sees the climb, the smoothed homography heights along the street, odometry continuing the last measured height
in the final hover. Session 2026-10-20 with the ChArUco lens: 8.0 m low hover, 22.7-24.0 m car park, 31-36 m street,
25.7 m final hover. `altitude.telemetry` (the flight controller's height) replaces the plan: the setup for live use.

## 4. Roll and pitch: gyro + Xsens

In hover the Xsens roll and pitch are off by about 0.8 deg over 2 s and 1-2 deg over 10 s against the integrated gyro
(which predicts the image rotation to 0.03 deg). At 50 m that moved a parked car by +-2 m from frame to frame. Fused
attitude: the world up vector seen in the body (from the Xsens roll and pitch) is carried into the gyro's reference
frame, where it should stand still, averaged over 10 s (centred), and brought back to the body with the gyro
orientation. The gyro supplies everything faster than 10 s, the Xsens the gravity reference. Fused minus Xsens in the
car park hover: sd 0.43 deg (roll), 0.14 deg (pitch), up to 4.8 deg. `attitude.tilt_window_s: 0` switches it off.

## 5. Heading (`heading`)

The Xsens magnetic yaw is off by up to 120 deg near the ground and drifts by -50 deg in 150 s of car park hover while
the gyro says the drone did not turn. The yaw used: the gyro's Euler yaw rate
`psi_dot = (w_y sin(roll) + w_z cos(roll)) / cos(pitch)` (bias and scale corrected), integrated, plus a correction
c(t) from the visual motion direction: for every GNSS + homography pair the image gives the direction of the camera's
horizontal displacement in camera axes, the GNSS gives it in the world; their difference is the heading error
(`yaw_true = yaw_xsens + e`, e = image bearing - GNSS bearing). Anchors come in clusters (gaps over 20 s), a cluster
needs 5 anchors, outliers beyond 3 MAD (at least 6 deg) are dropped, a 30 s rolling median follows the slow gyro drift
inside a cluster, c(t) is interpolated between clusters.

Session 2026-10-20: 161 anchors in 3 clusters (low hover 170-194 s, street 453-576 s and 604-658 s); the Xsens heading
error at the anchors is +8 deg median at height and -64..+13 deg (p10..p90) overall; after the correction the
residual is -0.1 +- 1.0 deg (MAD). Independent checks of the corrected heading: parked cars line up along the parking
row on the satellite image (with the Xsens yaw they scatter over the building roofs), and a north-south line projected
into street frames runs along the road edges (with the Xsens yaw it crosses them). Static-car trilateration cannot
check the heading on this flight: from a nearly straight path the mirrored point fits the distances equally well.

## 6. Detection, ranging, tracking (`detect`, `range`, `track`)

YOLO11x + ByteTrack at 2048 px, confidence 0.15. Any detector works: ranging needs only the box. The drone position is
the GNSS position low-passed (5 s, zero phase) plus the receiver velocity track.

Ground-plane tracker (`worldtrack.py`): a constant-velocity Kalman filter [E, N, vE, vN] per track (white-noise
acceleration 0.3 m^2/s^3 for people, 1.5 for vehicles), measurement noise larger along the camera ray (range) than
across it. Duplicate boxes on one object (car + truck, a box nested in another) are merged first. Association in two
stages: (1) where ByteTrack kept its id and the track was seen within 1 s, the track continues if its ground point
stays within 3 m for vehicles, 2 m for people (and a Mahalanobis gate of 9): frame to frame the image is the more
reliable cue, adjacent parked cars are only 2.5 m apart on the ground; (2) the rest by Hungarian assignment per class
group on the Mahalanobis distance (gate chi-square 13.8) with a bonus for overlapping the track's last box. The ground
plane is what stitches ByteTrack's fragments together. Confirmed after 3 hits in 6 frames; ids are small integers in
the order of confirmation. A track outside the camera footprint is remembered (parked: 20 min, moving: 20 s) and found
again by position; an unobserved track coasts with its velocity damped (3 s).

Moving test: a straight line through the last 4 s of measured positions; moving when its speed exceeds the class
threshold (people 0.8 m/s, vehicles 1.5 m/s, growing with distance as 0.01 / 0.03 m/s per metre) and 3 standard
errors, still again below 0.45 / 0.8 m/s (or 1.5 standard errors). Speed and direction are reported only while moving (the raw filter speed stays in `speed_kf`); the video
shows a vehicle as moving after 1 s of motion.

Session 2026-10-20: 252 ids (11 people, 241 vehicles) where ByteTrack had 117 + 1025. Car park hover (2 min): 45 ids;
of the image tracker's long tracks, 81 % of the frames stay on one ground track (98 % in the low hover); 2.2 % of the
parked-vehicle rows show as moving (they wander with the height and attitude errors). Two cars parked side by side
can still swap ids. Real traffic shows 20-45 km/h at 100-125 m. Output `tracks.csv` (one row per frame and track:
status seen / coast / memory, E, N, lat, lon, velocity, direction, Xo / Yo / Zo, range, ground distance, the matched
box) and `tracks.geojson`.

## 7. Video (`render`)

1920 x 1080: the camera view with boxes and labels, a drone-centred radar, a north-up satellite map (Esri World
Imagery tiles, cached in `work_dir/tiles`) and the nearest tracks in a table. Label (VyzAI style): "CLASS id", Xo,
Yo, Zo, Range, Velocity in km/h with an arrow for the direction of motion (up = away from the drone along the camera
heading, as on the radar). `hud.labels: auto` gives people and moving objects the full label and parked vehicles class,
id and range; `full` gives every track the full label. The map shows every remembered track by id: filled while in
view, hollow when out of view, with a velocity arrow while moving; the drone with its 60 s trail and the camera
footprint.

## 8. Checks without a survey (`validate`, `terrain`)

- **Pilot triangle.** The pilot stands at the take-off point (first GNSS fix); while he stands there, the ground
  distance to him must equal the GNSS distance drone -> take-off point. 5 windows in the low hover:
  +6.4 / +1.5 / -0.7 / +0.8 / -7.9 % at 19-29 m (with the self-calibrated lens: +4.4 / +0.1 / -1.2 / +1.4 / -6.2 %).
  He is the only person in the low hover; in his other tracks he walks around, so they do not count.
- **Static-vehicle trilateration.** 33 parked cars seen from the moving drone (baseline >= 15 m): the distances meet
  in one point with 2.6 % per-frame scatter, 0.2 % median bias.
- **Terrain.** USGS 3DEP on a 10 m grid: the site is flat (objects within -0.39..+0.35 m of the take-off ground),
  flat-ground error -3.0..+1.4 %. `range --terrain` intersects the rays with the terrain model.

## 9. Limits

- Height enters 1:1. Live use needs the flight controller's barometric or laser height; the visual height lags during
  fast climbs (parked cars then appear to move for a few seconds).
- Heading: anchored only where the drone translates a few metres; between anchors it is the gyro (6-8 deg between
  clusters here). The flight controller's heading or the GNSS course in forward flight would serve live.
- Past 100 m a 1 deg ray error is about 7 m of position: far objects can show a few km/h of false motion.
- Flat ground is assumed unless the terrain model is used. Occluded contact points (feet behind a car) bias the
  distance long. The stock detector misses some people and dark cars from above.
- The satellite imagery is older than the scene in places: west of the street the video shows a parking lot where the
  imagery still has trees, so cars parked there sit "in the trees" on the map (the camera frame warped onto the
  imagery puts the street exactly on the street).
