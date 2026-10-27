"""Multi-object tracker on the ground plane (ENU metres): persistent ids, velocity and direction of motion.

Each detection becomes a ground point: the feet for people, the middle of the body for vehicles (the box-centre ray
meets the plane 0.75 m above the ground, which stays put when the drone flies around a parked car; the box bottom
does not). Every track is a constant-velocity Kalman filter [E, N, vE, vN] with white-noise acceleration; the
measurement noise is larger along the camera ray (range) than across it. Association per frame and class group:
first the image tracker's continuity (ByteTrack kept its id and the track was seen within 1 s: frame to frame the
image is the more reliable cue, adjacent parked cars are only 2.5 m apart on the ground), then a Hungarian assignment
on the Mahalanobis distance for the rest, gated, with a bonus for the box overlapping the track's last box; the ground
plane is what stitches the image tracker's fragments together.
A track is confirmed after 3 hits; its id is a small integer in the order of confirmation. A track that leaves the
camera footprint is remembered (parked cars keep their id when the camera comes back); an unobserved track coasts
with its velocity damped, so a stopped object is found again near where it was.
Every detection is reported as the tracker has it in that frame (`emit_unconfirmed`): a detection whose track is not
confirmed yet, or that starts no track, comes out as status "tentative" with id -1; nothing is filled in from later
frames (duplicates removed by the box suppression are not reported).
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

GROUPS = {"person": "person", "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle", "bicycle": "bicycle"}
REF_HEIGHT = {"person": 0.0, "vehicle": 0.75, "bicycle": 0.5}       # tracked point above the ground (m)

DEFAULTS = dict(
    q={"person": 0.3, "vehicle": 1.5, "bicycle": 1.0},             # white-noise acceleration spectral density (m^2/s^3)
    sigma_along=(0.25, 0.012), sigma_cross=(0.2, 0.006),            # measurement sigma = a + b * ground distance (m)
    ref_sigma={"person": 0.15, "vehicle": 0.5, "bicycle": 0.3},     # wander of the reference point on the object (m)
    gate=13.8,                                                      # chi-square, 2 dof, 99.9 %
    max_jump={"person": 6.0, "vehicle": 12.0, "bicycle": 8.0},     # m, plus speed * dt
    bytetrack_bonus=4.0, iou_bonus=8.0,                            # cost bonuses: same image-tracker id, box overlap with the last frame
    confirm_hits=3, confirm_window=6,
    min_conf_new={"person": 0.25, "vehicle": 0.4, "bicycle": 0.3},
    damp_s=3.0,                                                     # velocity decay time while unobserved
    max_pos_sigma={"person": 8.0, "vehicle": 12.0, "bicycle": 10.0},
    max_miss_in_view_s=6.0, memory_static_s=1200.0, memory_moving_s=20.0,
    moving_window_s=4.0, moving_fast=3.0,                           # straight-line fit of the last 4 s (m/s), hysteresis:
    moving_on={"person": 0.8, "vehicle": 1.5, "bicycle": 1.0}, moving_off={"person": 0.45, "vehicle": 0.8, "bicycle": 0.6},
    moving_per_m={"person": 0.01, "vehicle": 0.03, "bicycle": 0.015},   # far objects need more: on = max(moving_on, k * distance)
    nms_iou=0.6, nms_ios=0.75,
    bt_link_s=1.0, bt_gate=9.0,                                     # the image tracker's id continues a track seen this recently,
    bt_jump={"person": 2.0, "vehicle": 3.0, "bicycle": 2.5},        # if the ground point stays this close (m, plus speed * dt)
    emit_unconfirmed=True,                                          # report the detections of tracks not confirmed yet (id -1)
)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _ios(a, b):
    """intersection over the smaller box: a box nested in another (car inside truck) is the same object"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / (min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) + 1e-9)


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]; x2, y2 = poly[(k + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


class Track:
    def __init__(self, tid, grp, z, R, t, det, P):
        self.tmp_id = tid
        self.id = None
        self.grp = grp
        self.x = np.r_[z, 0.0, 0.0]
        self.P = np.diag([R[0, 0], R[1, 1], 4.0 if grp == "person" else 25.0, 4.0 if grp == "person" else 25.0])
        self.P[:2, :2] = R
        self.t = t                      # time of the state
        self.t_seen = t
        self.t_born = t
        self.hits = 1
        self.frames = [det["frame"]]
        self.cls_count = {det["cls"]: 1}
        self.bt_id = det["track"]
        self.det = det
        self.speeds = []
        self.misses_in_view = 0.0
        self.P_ = P
        self.hist = [(t, float(z[0]), float(z[1]))]                  # measured positions of the last moving_window_s
        self.moving = False
        self.moving_since = None
        self.v_ls, self.sv_ls = 0.0, float("inf")

    @property
    def cls(self):
        return max(self.cls_count, key=self.cls_count.get)

    @property
    def confirmed(self):
        return self.id is not None

    def predict(self, t, observed_recently=True):
        dt = t - self.t
        if dt <= 0:
            return
        q = self.P_["q"][self.grp]
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        Q = q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2], [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        unseen = t - self.t_seen
        if unseen > 0.5:                                             # coasting: the velocity dies out, a stopped object stays put
            k = math.exp(-dt / self.P_["damp_s"])
            D = np.diag([1.0, 1.0, k, k])
            self.x = D @ self.x
            self.P = D @ self.P @ D
        smax = self.P_["max_pos_sigma"][self.grp]
        for j in (0, 1):
            if self.P[j, j] > smax ** 2:
                f = smax / math.sqrt(self.P[j, j])
                self.P[j, :] *= f; self.P[:, j] *= f
        self.t = t

    def innovation(self, z, R):
        S = self.P[:2, :2] + R
        v = z - self.x[:2]
        return v, S, float(v @ np.linalg.solve(S, v))

    def update(self, z, R, t, det):
        v, S, _ = self.innovation(z, R)
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ v
        IKH = np.eye(4) - K @ np.eye(2, 4)
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T                    # Joseph form: stays symmetric positive definite
        self.P = 0.5 * (self.P + self.P.T)
        self.t_seen = t
        self.hits += 1
        self.frames.append(det["frame"])
        self.cls_count[det["cls"]] = self.cls_count.get(det["cls"], 0) + 1
        self.bt_id = det["track"]
        self.det = det
        self.misses_in_view = 0.0
        self.speeds.append(math.hypot(self.x[2], self.x[3]))
        self.speeds = self.speeds[-15:]
        self.hist.append((t, float(z[0]), float(z[1])))              # raw measured positions
        W = self.P_["moving_window_s"]
        while self.hist and self.hist[0][0] < t - W:
            self.hist.pop(0)
        self.v_ls, self.sv_ls = motion_fit(self.hist)
        if not self.moving:
            if len(self.hist) >= 8 and self.hist[-1][0] - self.hist[0][0] >= 0.6 * W:
                self.moving = self.v_ls > max(self.P_["moving_on"][self.grp], self.P_["moving_per_m"][self.grp] * det["D"], 3 * self.sv_ls)
            elif len(self.hist) >= 5 and self.hist[-1][0] - self.hist[0][0] >= 1.0:
                self.moving = self.v_ls > max(self.P_["moving_fast"], 3 * self.sv_ls)
        elif len(self.hist) >= 5 and (self.v_ls < self.P_["moving_off"][self.grp] or self.v_ls < 1.5 * self.sv_ls):
            self.moving = False
        if self.moving and self.moving_since is None:
            self.moving_since = t
        elif not self.moving:
            self.moving_since = None

    @property
    def static(self):
        return not self.moving and self.hits >= 8


def motion_fit(hist):
    """least-squares straight line through the recent measured positions -> (speed, its standard error)"""
    if len(hist) < 3:
        return 0.0, float("inf")
    a = np.asarray(hist)
    t = a[:, 0] - a[:, 0].mean()
    stt = float(t @ t)
    if stt < 1e-6:
        return 0.0, float("inf")
    ve = float(t @ (a[:, 1] - a[:, 1].mean())) / stt
    vn = float(t @ (a[:, 2] - a[:, 2].mean())) / stt
    res = np.r_[a[:, 1] - a[:, 1].mean() - ve * t, a[:, 2] - a[:, 2].mean() - vn * t]
    s2 = float(res @ res) / max(2 * len(t) - 4, 1)
    return math.hypot(ve, vn), math.sqrt(s2 / stt)


def measurement_cov(ref_e, ref_n, cam_e, cam_n, grp, P):
    d = math.hypot(ref_e - cam_e, ref_n - cam_n)
    sa = P["sigma_along"][0] + P["sigma_along"][1] * d
    sc = P["sigma_cross"][0] + P["sigma_cross"][1] * d
    rs = P["ref_sigma"][grp]
    u = np.array([ref_e - cam_e, ref_n - cam_n]) / max(d, 1e-6)
    w = np.array([-u[1], u[0]])
    return (sa ** 2 + rs ** 2) * np.outer(u, u) + (sc ** 2 + rs ** 2) * np.outer(w, w)


def _track_row(tr, i, t, t0, status):
    seen = tr.t_seen == t
    return dict(frame=i, t=t - t0, id=tr.id if tr.confirmed else -1, cls=tr.cls, grp=tr.grp, status=status,
                E=tr.x[0], N=tr.x[1], vE=tr.x[2], vN=tr.x[3], speed=math.hypot(tr.x[2], tr.x[3]) if tr.moving else 0.0,
                speed_kf=math.hypot(tr.x[2], tr.x[3]), static=tr.static, moving=tr.moving,
                moving_s=0.0 if tr.moving_since is None else t - tr.moving_since,
                sigma_pos=math.sqrt(max(tr.P[0, 0], tr.P[1, 1])), hits=tr.hits, det_row=tr.det["row"] if seen else -1,
                conf=tr.det["conf"] if seen else np.nan, bt_id=tr.bt_id)


def _det_row(d, i, t, t0):
    """a detection that starts no track (below min_conf_new): reported where it was measured, without an id"""
    return dict(frame=i, t=t - t0, id=-1, cls=d["cls"], grp=d["grp"], status="tentative", E=d["z"][0], N=d["z"][1],
                vE=0.0, vN=0.0, speed=0.0, speed_kf=0.0, static=False, moving=False, moving_s=0.0, sigma_pos=np.nan,
                hits=1, det_row=d["row"], conf=d["conf"], bt_id=d["track"])


class WorldTracker:
    """the tracker frame by frame (step), for a live run; run() feeds it a whole session"""

    def __init__(self, sess, params=None):
        P = {**DEFAULTS, **(params or {})}
        for k in ("q", "ref_sigma", "max_jump", "min_conf_new", "max_pos_sigma", "moving_on", "moving_off", "moving_per_m", "bt_jump"):
            P[k] = {**DEFAULTS[k], **(params or {}).get(k, {})}
        self.P, self.emit, self.t0 = P, P["emit_unconfirmed"], float(sess.ft[0])
        self.tracks, self.next_tmp, self.next_id, self.prev_t = [], 0, 1, None

    def step(self, i, t, g, poly):
        """frame i at time t: g = its ranged detections (DataFrame with Er, Nr, ...; index = detection row) or None,
        poly = the camera footprint (ENU) -> the rows of this frame"""
        P, emit, t0 = self.P, self.emit, self.t0
        out = []
        dets = []
        if g is not None:
            for o in g.itertuples():
                grp = GROUPS.get(o.cls)
                if grp is None or not np.isfinite(o.Er):
                    continue
                dets.append(dict(frame=i, cls=o.cls, grp=grp, conf=o.conf, track=int(o.track), box=(o.x1, o.y1, o.x2, o.y2),
                                 z=np.array([o.Er, o.Nr]), cam=(o.E_drone, o.N_drone), D=o.D, row=o.Index))
            dets.sort(key=lambda d: -d["conf"])                     # duplicates (car + truck on one object): keep the stronger
            kept = []
            for d in dets:
                if not any(k["grp"] == d["grp"] and (_iou(k["box"], d["box"]) > P["nms_iou"] or _ios(k["box"], d["box"]) > P["nms_ios"]) for k in kept):
                    kept.append(d)
            dets = kept
        tracks = self.tracks
        for tr in tracks:
            tr.predict(t)
        matched_t, matched_d = set(), set()
        for k, d in enumerate(dets):                                 # 1) the image tracker kept its id: continue that track
            if d["track"] < 0:
                continue
            Rk = measurement_cov(d["z"][0], d["z"][1], d["cam"][0], d["cam"][1], d["grp"], P)
            cand = [(tr.t_seen, j) for j, tr in enumerate(tracks) if j not in matched_t and tr.grp == d["grp"] and tr.bt_id == d["track"]
                    and t - tr.t_seen <= P["bt_link_s"] and tr.innovation(d["z"], Rk)[2] < P["bt_gate"]
                    and math.hypot(*(d["z"] - tr.x[:2])) <= P["bt_jump"][d["grp"]] + math.hypot(tr.x[2], tr.x[3]) * (t - tr.t_seen)]
            if cand:
                j = max(cand)[1]
                tracks[j].update(d["z"], Rk, t, d)
                matched_t.add(j); matched_d.add(k)
        for grp in ("person", "vehicle", "bicycle"):                  # 2) the rest: ground-plane association
            ti_ = [k for k, tr in enumerate(tracks) if tr.grp == grp and k not in matched_t]
            di_ = [k for k, d in enumerate(dets) if d["grp"] == grp and k not in matched_d]
            if not ti_ or not di_:
                continue
            C = np.full((len(ti_), len(di_)), 1e6)
            Rs = {}
            for b, dk in enumerate(di_):
                d = dets[dk]
                Rs[dk] = measurement_cov(d["z"][0], d["z"][1], d["cam"][0], d["cam"][1], grp, P)
                for a, tk in enumerate(ti_):
                    tr = tracks[tk]
                    v, S, m2 = tr.innovation(d["z"], Rs[dk])
                    jump = math.hypot(*v)
                    if m2 > P["gate"] or jump > P["max_jump"][grp] + math.hypot(tr.x[2], tr.x[3]) * (t - tr.t_seen):
                        continue
                    iou = _iou(tr.det["box"], d["box"]) if t - tr.t_seen < 0.45 else 0.0
                    C[a, b] = (m2 - (P["bytetrack_bonus"] if d["track"] >= 0 and d["track"] == tr.bt_id else 0.0)
                               - P["iou_bonus"] * iou - (2.0 if tr.confirmed else 0.0))
            ra, cb = linear_sum_assignment(C)
            for a, b in zip(ra, cb):
                if C[a, b] >= 1e5:
                    continue
                tr, d = tracks[ti_[a]], dets[di_[b]]
                tr.update(d["z"], Rs[di_[b]], t, d)
                matched_t.add(ti_[a]); matched_d.add(di_[b])
        for k, d in enumerate(dets):
            if k in matched_d:
                continue
            Rm = measurement_cov(d["z"][0], d["z"][1], d["cam"][0], d["cam"][1], d["grp"], P)
            if d["conf"] < P["min_conf_new"][d["grp"]]:
                if emit:
                    out.append(_det_row(d, i, t, t0))
                continue
            tracks.append(Track(self.next_tmp, d["grp"], d["z"], Rm, t, d, P)); self.next_tmp += 1
            matched_t.add(len(tracks) - 1)
        dt = t - self.prev_t if self.prev_t is not None else 0.2
        self.prev_t = t
        keep = []
        for k, tr in enumerate(tracks):
            if not tr.confirmed:
                recent = [f for f in tr.frames if f > i - P["confirm_window"]]
                if len(recent) >= P["confirm_hits"]:
                    tr.id = self.next_id; self.next_id += 1
                elif i - tr.frames[0] >= P["confirm_window"]:
                    continue                                         # never confirmed: drop
            if k not in matched_t:
                in_view = poly is not None and point_in_poly(tr.x[0], tr.x[1], poly)
                if in_view:
                    tr.misses_in_view += dt
                unseen = t - tr.t_seen
                limit = P["memory_static_s"] if tr.static else P["memory_moving_s"]
                if tr.misses_in_view > P["max_miss_in_view_s"] or unseen > limit:
                    continue
            keep.append(tr)
        self.tracks = keep
        for tr in keep:
            if not tr.confirmed:
                if emit and tr.t_seen == t:
                    out.append(_track_row(tr, i, t, t0, "tentative"))   # detected, no id yet
                continue
            unseen = t - tr.t_seen
            out.append(_track_row(tr, i, t, t0, "seen" if unseen == 0 else ("coast" if unseen <= 1.0 else "memory")))
        return out


def run(sess, R, footprints, params=None, log=print):
    """R: ranges with reference points (Er, Nr) and the smoothed drone position; footprints: frame -> ENU polygon.
    Returns one row per frame and track that is confirmed and not forgotten (status seen / coast / memory) and, with
    emit_unconfirmed, one per detection without a confirmed track (status tentative, id -1)."""
    wt = WorldTracker(sess, params)
    by = {f: g for f, g in R.groupby("frame")}
    rows = []
    for i in range(int(R.frame.min()), int(R.frame.max()) + 1):
        rows += wt.step(i, float(sess.ft[i]), by.get(i), footprints.get(i))
    T = pd.DataFrame(rows)
    if len(T):
        T = T.sort_values(["frame", "id"], kind="stable").reset_index(drop=True)
        C = T[T.id >= 0].drop_duplicates("id")
        log(f"world tracks: {len(C)} confirmed ids ({(C.grp == 'person').sum()} person, {(C.grp == 'vehicle').sum()} vehicle), "
            f"{int((T.status == 'tentative').sum())} unconfirmed detections; image tracker ids for comparison: "
            f"{R[R.track >= 0].groupby(R.cls.map(GROUPS)).track.nunique().to_dict()}")
    return T
