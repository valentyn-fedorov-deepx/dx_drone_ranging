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
from an anchor height. Every pair with a homography counts: an earlier version dropped pairs whose plane normal
disagreed with the IMU by more than 8 deg, which during climbs was most of them, and lost a third of the climb. Against
the barometer (10-23, climbs 13.1 -> 30.2 m) the end height came out -27 % with that rejection and -9 % without it.

**Car widths (check).** Rays through the bottom corners of parked-car boxes hit the ground at h = 1 m; the 10th
percentile of those widths is the car width (1.9 m), so `h = 1.9 / w10`. A coarse scale check: it reads 0.80 x the
track height here (0.82-0.88 in the first analysis), open until the drone's own altitude log is available.

**Height track.** A per-session plan stitches the sources (config `altitude.track`): a constant for the low hover
(7.9 m from GNSS displacement pairs), vertical odometry for the climb and the car park hover, a ramp where neither
method sees the climb, the smoothed homography heights along the street, odometry continuing the last measured height
in the final hover. Session 2026-10-20 with the ChArUco lens: 8.0 m low hover, 26-28 m car park, 31-36 m street,
25 m final hover (the car park was 23-24 m before the odometry fix). `altitude.telemetry` (the flight controller's height) replaces the plan: the setup for live use.

**Aircraft telemetry.** With the aircraft's own flight log (a DJI record decoded to CSV, about 10 records a second)
the height above the take-off point comes from its barometer (`altitude.telemetry`), the drone position from its GNSS
(`position_source`) and the heading from its compass-aided attitude (`heading.source: dji`, section 5). The log's clock
is aligned to the unit's by cross-correlating the horizontal speed (10-23: +1.25 s, correlation 0.92). On 10-23 the
unit's own GNSS drifted 18 m east (and its height by -15 m) while the aircraft climbed in place; the aircraft's GNSS held
within 1 m.

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

**Heading from the aircraft (`heading.source: dji`).** The camera unit is fixed to the airframe, so its ENU yaw is
`c0 - (DJI yaw)` with one constant c0 (the mount), taken from the visual-GNSS anchors of the moving stretches. 10-23:
c0 = -176.9 deg from 19 anchors, residual 1.7 deg (MAD); the gyro yaw drifts -1.2 deg per minute against it.

**Camera tilt: measured, not set.** The tilt of the optical axis comes from the IMU in the camera unit every frame, so
no angle has to be configured. The 10-23 flight was set up with a nominal 30 deg; the IMU reads 23-24 deg below the
horizon in the holds and 24.3 deg on the ground before take-off (the airframe 2 deg nose down). Three independent checks
side with the measured angle: the street's near edge in frame 2522 is 86.4 m away with it and 82-85 m away on the
satellite imagery along the same ray (a fixed 30 deg: 54 m); RVs across the street at 130 m land on the RV lot; and the
parked cars' ground distances stay most consistent across the five heights (2.0 % median change, against 2.8 % for a
fixed 30 deg and 4.1 % for 21 deg). `attitude.fixed_depression_deg` forces a fixed tilt (with the measured heading and a
level horizon) for comparisons or for a camera without the IMU; on 10-23 a fixed 30 deg makes every distance about 20 %
short. Tools: `tools/tilt_test.py`, `tools/road_check.py`, `tools/overlay_check.py`.

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
box) and `tracks.geojson`. Every detection is reported as the tracker has it in that frame: a detection whose track is
not confirmed yet (its first two hits) or that starts no track (below the confidence for a new track) is a row with
status `tentative` and id -1. Nothing is filled in from later frames, so the video shows what a live run would have
had. 10-23: 14036 detections; 12814 reported (12619 on confirmed tracks, 195 tentative); the other 1222 are the second
box on an object (car + truck on one vehicle), merged before association.

## 7. Video (`render`) and viewer (`gui`)

1920 x 1080: the camera view with labels, a drone-centred radar, a north-up satellite map (Esri World Imagery tiles,
cached in `work_dir/tiles`) and a table of the objects detected in the frame. Label (VyzAI style): "CLASS id", Xo, Yo,
Zo, Range, Velocity in km/h with an arrow for the direction of motion (up = away from the drone along the camera
heading, as on the radar). The labels show the system's state in that frame and nothing else: every detection of the
frame gets one, a frame where the detector missed an object has no label on it, and a detection the tracker has not
confirmed yet shows the class without an id and without a velocity. `hud.labels: full` (10-23) gives every detection
the full label; `auto` gives parked vehicles class, id and range only; `compact`, `off`. Detection boxes are off
(`hud.boxes: true` draws them); a label sits next to its object (above, below, beside) and moves away with a thin
leader only where it would cover another label or object. Radar and map show the tracker's state: filled = detected in
this frame, hollow = kept by the tracker but not detected in this frame (briefly missed, or out of view), with a
velocity arrow while moving; the map also has the drone with its 60 s trail and the camera footprint.

The camera view can be the video or any polarisation view of the raw frames (section 8): `render --view nxyz`, or
`--view tour` (every view in turn, `tour_s` seconds each; preset `hold30_views`). Only frames with a raw frame are
rendered in those views.

**Stabilised view** (`stabilise`, then `hud.stabilise: true`). The camera hangs on the aircraft without a gimbal and
the image shakes: frame to frame 15 / 38 px (x / y, shake = rms change of the frame-to-frame motion, pixels of the
1280 x 938 view) in the 30 m hold, mostly a 0.36 Hz pitch sway of up to 1.6 deg. The gyro attitude explains it
(correlation with the image motion 0.93 horizontally, 0.99 vertically), so each frame is turned to a smooth virtual
camera: azimuth, elevation and roll of the optical axis follow the smoothest path (least squared second and first
differences) inside a corridor around the real ones, the corridor being 60 % of the margin a 1.25 zoom leaves; what
the turn leaves (the drone's sway of a few tens of centimetres) is measured on the turned frames by phase correlation
and smoothed inside 35 % of the margin. The path breaks at the two recording pauses. Result (10-23, shake x / y):

| part of the flight | raw | stabilised |
|---|---|---|
| 30 m hold, frames 2072-2982 | 15.3 / 37.9 | 0.5 / 1.8 |
| climbs, 1150-2072 | 16.3 / 41.5 | 0.4 / 13.3 |
| flight to the drain, 0-520 | 42.9 / 47.7 | 11.1 / 27.2 |

The climbs and the approach keep more because the drone really moves there and the margin limits the turn. The
frame fills the view in 99.8 % of the frames (the worst frame leaves 0.2 % of the view at the edge). The view shows
80 % of the frame; labels and clicks are mapped through the same turn, the radar footprint follows the virtual camera,
and the measurements are those of the real camera. The viewer has a `stabilise` switch.

**Presentation video** (`demo`, `demo.py`): 1920 x 1080 at 30 fps, laid out for showing rather than for work: the
stabilised camera view at full height (1474 x 1080), labels with the same VyzAI fields drawn as cards next to their
object with a leader to the object's point (feet of a person, middle of a vehicle), and a side panel (altitude,
camera tilt as measured, objects detected, satellite map with the tracks, nearest objects, the view and its colour
scale, the altitude profile of the flight with a playhead). The sections, titles and closing facts come from the
config's `demo` block. Nothing is added to the recording: every camera frame is held for its 0.2 s (the timelapse shows
every second frame), only the titles fade and the views of one frame cross-fade; the polarisation views are computed
at full resolution there.

`python -m dronerange --config <session>.yaml gui` opens the same frames in a window: the views as buttons (keys 1-8),
play / pause, a frame slider, the label mode and the boxes. In a polarisation view the slider and the arrow keys step
over the frames with a raw frame; four worker processes decode them ahead (0.3-0.5 s per frame each). A click on the
picture gives that pixel's ground point (distance from the drone, slant range, bearing, lat / lon: the ray to the
ground at the frame's height), the values behind the polarisation views (S0, Nz, Nxy, the Nxyz components and the
three physical models' vectors) and the detection under the click. Ctrl+S saves the frame as PNG.

## 8. Polarisation views of the raw frames (`polar`)

The unit's sensor (Sony IMX253MYR, 4096 x 3000, 12 bit) has a 4 x 4 cell of colour x polariser angle (0 / 45 / 90 /
135 deg in each 2 x 2 cell, the 2 x 2 cells R / G / G / B), stored upside down; the raw frames come as MIPI RAW12 or,
in the archives, as one JPEG-LS picture per frame with the pixels regrouped into 16 tiles (`polar.RawStore` reads them
in place, archives still downloading up to their last complete frame). The views follow the drone application, as
found by fitting its own decodes of frame 2522 (`frame_2026_10_23_18_24_19/`):

- no black level subtracted; each polariser angle is a 2048 x 1500 Bayer picture (one value per 2 x 2 cell),
  demosaiced edge-aware; the intensity is the mean of R, G, B;
- S0 = (I0 + I45 + I90 + I135) / 2, S1 = I0 - I90, S2 = I45 - I135; degree rho = |(S1, S2)| / S0; angle
  phi = atan2(S2, S1) / 2;
- **Nz**: grey rho / 0.7; **Nxy**: the jet colour map over phi from -90 to 90 deg; **Nxyz**: (|sin t cos(phi + 45)|,
  |sin t sin(phi + 45)|, cos t), t = atan(19 rho);
- physical models: the zenith from rho by the Fresnel equations with n = 1.5, diffuse (**PN PxDiffuse**,
  **PN VyzLut**, saturating at 90 deg from rho 0.38) or specular below the Brewster angle (**PN PxSpecularV2**); the
  azimuth phi with y <= 0 (PxDiffuse, PxSpecularV2) or phi + 90 deg with x <= 0 (VyzLut); colour (v + 1) / 2;
- **S0** in colour: the cell sums demosaiced, a colour matrix fitted to the application's S0 picture of the unit
  (`configs/cameras/drone2244_s0_colour.json`, `polar --fit-colour`), the sRGB curve.

| view (frame 2522, full resolution) | rms against the application, levels of 255 |
|---|---|
| S0 | 1.2 |
| Nz | 1.3 |
| Nxy | 13 (median 0: one step of the jet map is 4 levels, the differences sit on noisy pixels) |
| Nxyz | 5.8 (median 2) |
| PN PxSpecularV2 | 1.8 |
| PN PxDiffuse, PN VyzLut | 7.3 (median 1: the diffuse zenith is steep at small rho) |

`polar` reproduces the table (`polar_verify.json`, `polar_verify.jpg`: ours above, the application's below). The
viewer and the videos use the Stokes values averaged over 2 x 2 cells (1024 x 750), which suppresses the per-cell noise;
the application's own live views are 512 x 375.

## 9. Checks without a survey (`validate`, `terrain`)

- **Pilot triangle.** The pilot stands at the take-off point (first GNSS fix); while he stands there, the ground
  distance to him must equal the GNSS distance drone -> take-off point. 5 windows in the low hover:
  +6.4 / +1.5 / -0.7 / +0.8 / -0.3 % at 19-29 m; the last window sits on the odometry height and was -7.9 % before
  the odometry fix (with the self-calibrated lens: +4.4 / +0.1 / -1.2 / +1.4 / -6.2 %).
  He is the only person in the low hover; in his other tracks he walks around, so they do not count.
- **Static-vehicle trilateration.** 33 parked cars seen from the moving drone (baseline >= 15 m): the distances meet
  in one point with 2.6 % per-frame scatter, 0.2 % median bias.
- **Terrain.** USGS 3DEP on a 10 m grid: the site is flat (objects within -0.39..+0.35 m of the take-off ground),
  flat-ground error -3.0..+1.4 %. `range --terrain` intersects the rays with the terrain model.

## 10. Limits

- Height enters 1:1. Live use needs the flight controller's barometric or laser height; the visual height lags during
  fast climbs (parked cars then appear to move for a few seconds), and the vertical odometry still reads climbs about
  9 % short (barometer check, 10-23).
- The unit's GNSS receiver is unreliable on the drone: its height drifted on both flights and on 10-23 its horizontal
  position drifted 18 m while the aircraft held still. Use the aircraft's position when its log is available.
- Heading: anchored only where the drone translates a few metres; between anchors it is the gyro (6-8 deg between
  clusters here). The flight controller's heading or the GNSS course in forward flight would serve live.
- Past 100 m a 1 deg ray error is about 7 m of position: far objects can show a few km/h of false motion.
- Flat ground is assumed unless the terrain model is used. Occluded contact points (feet behind a car) bias the
  distance long. The stock detector misses some people and dark cars from above.
- What the video and the viewer show is causal for detection and tracking (no label or id from later frames), but the
  geometry still uses centred smoothing windows offline (drone position 1 s, the slow part of roll and pitch 10 s, the
  heading anchors); a live run would use the one-sided versions with a little lag.
- The satellite imagery is older than the scene in places: west of the street the video shows a parking lot where the
  imagery still has trees, so cars parked there sit "in the trees" on the map (the camera frame warped onto the
  imagery puts the street exactly on the street).
