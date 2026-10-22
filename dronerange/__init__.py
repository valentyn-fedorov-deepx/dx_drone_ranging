"""dronerange: distance to people and vehicles seen from a drone camera, from the camera's own geometry.

Pipeline (each step is a CLI command, see `python -m dronerange --help`):
    calib     camera intrinsics (f, k1) + camera-IMU rotation + IMU time offset, from the flight itself
    altitude  height above ground per frame, from the images (GNSS height is not trusted)
    detect    people / vehicles with YOLO + ByteTrack
    range     ground distance, slant distance and bearing of every detection (camera ray to the ground)
    validate  pilot-at-take-off triangle check and static-vehicle trilateration
    terrain   USGS 3DEP terrain over the area, flat-ground error, optional terrain-aware ranging
    render    video with distance labels and a drone-centred radar
"""
__version__ = "0.1.0"
