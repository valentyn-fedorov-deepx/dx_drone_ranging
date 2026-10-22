"""Reader for the VyzAI recorder's Xsens MTi log (`xsens_<unix_seconds>.bin`).

Wire format (little-endian, fixed 115-byte records, no padding), as written by qcom_dev's recorder:
    int64 timestamp_ns | u8 has_accel, f32 acc[3] (m/s^2) | u8 has_gyro, f32 gyro[3] (rad/s) |
    u8 has_mag, f32 mag[3] (Gauss) | u8 has_euler, f32 roll, pitch, yaw (deg) | u8 has_time, u32 sample_time_fine |
    u8 has_status, u32 status_word | u8 has_gnss, f64 lat, lon, alt_m | f32 velN, velE, velD |
    u8 fix_type, num_sats, gnss_flags | f32 ground_speed | u8 gnss_valid
The column names match the session's decoded CSV (`gnss/imu_gnss_*.csv`), so either source can be used.
"""
import numpy as np
import pandas as pd

RECORD = np.dtype([
    ("t_ns", "<i8"), ("has_accel", "u1"), ("ax", "<f4"), ("ay", "<f4"), ("az", "<f4"),
    ("has_gyro", "u1"), ("gx", "<f4"), ("gy", "<f4"), ("gz", "<f4"),
    ("has_mag", "u1"), ("mx", "<f4"), ("my", "<f4"), ("mz", "<f4"),
    ("has_euler", "u1"), ("roll", "<f4"), ("pitch", "<f4"), ("yaw", "<f4"),
    ("has_time", "u1"), ("sample_time_fine", "<u4"), ("has_status", "u1"), ("status_word", "<u4"),
    ("has_gnss", "u1"), ("lat", "<f8"), ("lon", "<f8"), ("alt_m", "<f8"),
    ("velN", "<f4"), ("velE", "<f4"), ("velD", "<f4"),
    ("fix_type", "u1"), ("num_sats", "u1"), ("gnss_flags", "u1"), ("ground_speed", "<f4"), ("gnss_valid", "u1"),
])
assert RECORD.itemsize == 115


def read_bin(path):
    raw = open(path, "rb").read()
    n = len(raw) // RECORD.itemsize
    a = np.frombuffer(raw[: n * RECORD.itemsize], dtype=RECORD)
    return pd.DataFrame({name: a[name].astype(np.float64) if a[name].dtype.kind == "f" else a[name] for name in RECORD.names})


def read_imu(path):
    """Decoded CSV or raw .bin -> DataFrame sorted by time, duplicate / out-of-order timestamps removed
    (the recorder's log has a few backwards jumps when a write stalls)."""
    df = read_bin(path) if str(path).lower().endswith(".bin") else pd.read_csv(path)
    return df.sort_values("t_ns").drop_duplicates("t_ns").reset_index(drop=True)
