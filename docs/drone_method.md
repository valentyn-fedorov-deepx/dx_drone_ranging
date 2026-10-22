# Drone ranging: method, calibration, height, checks

Distance from a drone camera to people and vehicles on the ground, from geometry only (no learned depth): a ray from
the calibrated camera through the object's ground-contact pixel, turned into world coordinates with the IMU attitude,
meets the ground at the drone's height above ground.

## 1. Ray to the ground

- Contact pixel: the bottom centre of the detection box (feet, tyres).
- Camera ray: `b_c = normalize([x s, y s, 1])`, `x = (u - cx) / f`, `y = (v - cy) / f`, `s = 1 / (1 + k1 r^2)`.
- World ray: `b_w = R_wb(t + toff) R_bc b_c`, with `R_wb = Rz(yaw) Ry(pitch) Rx(roll)` (Xsens Euler, ZYX, ENU; the
  convention was checked against the accelerometer on the ground).
- Ground hit at height h: `lambda = h / (-b_w,z)`, offsets `dE = b_w,x lambda`, `dN = b_w,y lambda`.
  Ground distance `D = hypot(dE, dN)`, slant distance `S = hypot(D, h)`.
- Radar bearing: relative to the camera's ground heading (`atan2` of the optical axis), so it does not depend on
  the compass. Yaw only rotates both directions about the vertical, so `D` and the relative bearing are independent
  of the yaw error.

Error budget (depression angle alpha of the ray, `D = h / tan(alpha)`):

| error source | effect on D |
|---|---|
| height error dh | dD / D = dh / h (1:1) |
| attitude / ray angle error d_alpha | dD / D = 2 d_alpha / sin(2 alpha), about 3-5 % per degree at these angles |
| object ground dz above the reference ground | dD / D = -dz / h |
| heading (yaw) error | none on D and on the radar; only map positions |

## 2. Calibration from the flight (`calib`)

Hand-eye rotation calibration on hover pairs at height: for consecutive frames during hover (ground speed below
0.25 m/s, frames >= 1450, gyro rotation 1-10 deg), the image motion must be the pure rotation measured by the
bias-corrected gyro:

    b2 = R_bc^T dR^T R_bc b1,   dR = R_ref_b(t1)^T R_ref_b(t2)  (continuous Slerp of the integrated gyro)

Unknowns: f, k1, R_bc (rotation vector), IMU time offset. SIFT matches + RANSAC homography inliers (150 pairs,
up to 250 points each); start from the best of the 24 axis-aligned R_bc, then a grid over the time offset
(-200..200 ms) and f, then robust least squares (soft L1). The gyro bias is the mean of the first 12 s on the ground.
Pairs during the low hover are excluded: with a narrow lens the translation parallax there breaks the pure-rotation
model.

Session 2026-10-20: f = 4926.3 px (HFOV 23.5 deg, VFOV 17.3 deg, about a 34 mm lens on this sensor), k1 = +0.058,
IMU time offset -20.7 ms, residual median 2.71 px (p90 6.46); the optical axis is the IMU -Z axis, about 23 deg below
the drone's horizontal.

## 3. Height above ground (`altitude`)

The GNSS height of this unit drifts by -8 to +6 m against the true height, so the height comes from the images.

**GNSS displacement + ground homography (moving drone).** For frames i, j with a few metres of horizontal travel:
ground-plane homography H between the undistorted points, camera rotation R21 from the gyro, ground normal n in
camera-1 coordinates from roll and pitch. The plane-induced homography gives

    R21^T K^-1 H K / sigma_2 = I + u n^T,   u = -C2 / d

(C2 = camera-2 centre in camera-1 coordinates, d = height of camera 1 above the plane). In world axes
`u_w = R_wc1 u = -dC / d`, so `d = |dC_h| / |u_w,h|`, where |dC_h| is the horizontal displacement from the receiver
velocity (integrated). The compass is not needed: only the length of the horizontal displacement enters.
Per-pair checks: the dominant right singular vector of `M - I` agrees with the IMU normal within 4 deg, rank-1
residual below 0.15, more than 60 inliers, at least 2 m of travel. Pairs from hover (about 1 m of travel) are
garbage and must not be used. The motion direction is not used as a filter: near the ground the IMU heading is off
by tens of degrees while the height is unaffected.

**Vertical odometry (hover).** From the same model between frames two apart, `dz / h = -u_w,z`; integrated in
`ln h` from an anchor height. Pairs are used when the plane normal agrees within 8 deg (or the motion is tiny).

**Car widths (check).** For parked cars, the rays through the bottom-left and bottom-right box corners hit the
ground at h = 1 m; the 10th percentile of these widths over many cars is the car width (1.9 m with mirrors), so
`h = 1.9 / w10`. A coarse independent scale check.

**Height track.** A per-session plan in the config stitches these sources: a constant for the low hover (GNSS
displacement pairs, 7.9 m), vertical odometry for the climb and the car park hover, a short ramp where neither
method sees the climb, the smoothed (rolling median of 7) homography heights along the street, and vertical
odometry continuing the last measured height in the final hover. With the drone's own height-above-ground telemetry
(`altitude.telemetry`) the plan is replaced by it; that is the intended setup for live use.

## 4. Detection and ranging (`detect`, `range`)

YOLO11x + ByteTrack at 2048 px, confidence 0.15 (people are drawn from 0.2, vehicles from 0.35). Any detector works:
the ranging only needs the contact pixel.

## 5. Checks without a survey (`validate`, `terrain`)

- **Pilot triangle.** The pilot stands at the take-off point (first GNSS fix). While the camera sees the pilot, the
  ranged ground distance must equal the GNSS distance from the drone to the take-off point.
- **Static-vehicle trilateration.** A parked car seen from a moving drone (baseline >= 15 m) must be at one point;
  the per-frame scatter of the ranged distances about the least-squares point is the ranging noise, and the
  absolute bearing against the trilaterated direction gives the IMU heading error.
- **Terrain.** USGS 3DEP (EPQS) heights on a 10 m grid over the area: the flat-ground error is `dz / h` per object.
  `range --terrain` intersects the rays with the terrain model instead of the plane.

## 6. Limits

- Height is the main error source: it enters 1:1. Live use needs the flight controller's barometric or laser height.
- Xsens magnetic yaw: off by about 8 deg at height and by tens of degrees near the ground. The radar is
  drone-relative and unaffected; map positions and distances to other GPS points need the drone heading.
- Flat ground is assumed unless the terrain model is used.
- Occluded contact points (feet hidden by a car) bias the distance long.
