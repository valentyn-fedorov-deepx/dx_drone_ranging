"""YAML session config: data paths are relative to `data_root`, every product goes to `work_dir`."""
import datetime
import os
from pathlib import Path

import yaml


def load(path, work_dir=None):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    root = Path(cfg.get("data_root", "."))
    cfg["data"] = {k: str(v if Path(v).is_absolute() else root / v) for k, v in cfg["data"].items()}
    cfg["work_dir"] = str(work_dir or cfg.get("work_dir") or Path(path).with_suffix("").name + "_work")
    os.makedirs(cfg["work_dir"], exist_ok=True)
    os.makedirs(Path(cfg["work_dir"]) / "out", exist_ok=True)
    return cfg


def work(cfg, name):
    return str(Path(cfg["work_dir"]) / name)


class Log:
    """print + append to <work_dir>/log.txt"""

    def __init__(self, cfg):
        self.path = work(cfg, "log.txt")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for line in s.splitlines() or [""]:
                fh.write(f"{stamp}  {line}\n")
