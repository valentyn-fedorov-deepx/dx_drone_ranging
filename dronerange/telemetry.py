"""The aircraft's own log: a DJI flight record decoded to CSV (dji-log-parser, about 10 records a second) with the
height above the take-off point from the barometer, the aircraft's attitude and heading, and its GNSS position.

Times are UTC on the DJI app's clock, within about a second of the camera unit's; `offset_s` is added to them.
DJI yaw: degrees clockwise from north (a compass bearing of the aircraft's nose).
"""
import numpy as np
import pandas as pd


def load(path, t_col="utc", offset_s=0.0):
    d = pd.read_csv(path)
    t = pd.to_datetime(d[t_col], utc=True)
    d["t_unix"] = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() + offset_s
    return d.sort_values("t_unix").drop_duplicates("t_unix").reset_index(drop=True)


def height_at(d, t, h_col="height_above_takeoff_m"):
    return np.interp(t, d.t_unix.values, d[h_col].values)


def yaw_at(d, t, col="yaw_deg"):
    """aircraft heading, compass degrees, unwrapped"""
    y = np.degrees(np.unwrap(np.radians(d[col].values)))
    return np.interp(t, d.t_unix.values, y)


def standstill_gyro_bias(path, t0, t1, power_on=None):
    """mean gyro (rad/s) of the IMU polled before the recording, over a standstill window [t0, t1] (UTC strings)"""
    d = pd.read_csv(path)
    t = pd.to_datetime(d.utc, utc=True)
    m = (t >= pd.Timestamp(t0, tz="UTC")) & (t <= pd.Timestamp(t1, tz="UTC"))
    if power_on is not None:
        m &= d.imu_power_on == power_on
    g = d.loc[m, ["gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]].values
    return g.mean(0), g.std(0) / np.sqrt(max(len(g), 1)), int(len(g))
