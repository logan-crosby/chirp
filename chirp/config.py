"""Configuration loading for chirp. Merges a YAML config file with CLI arguments.

Priority (highest to lowest): CLI arguments → YAML config file → built-in defaults.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import yaml


@dataclass
class ZoneConfig:
    name: str
    polygon: List[Tuple[int, int]]  # list of [x, y] pairs


@dataclass
class HighlightsConfig:
    enabled: bool = True
    pre_roll: int = 5    # seconds of footage before bird arrives
    post_roll: int = 3   # seconds after bird leaves


@dataclass
class ScheduleConfig:
    start_time: Optional[str] = None  # "HH:MM" 24-hour; None = always run
    stop_time: Optional[str] = None


@dataclass
class ChirpConfig:
    # Input
    video_source: str = "/dev/video0"

    # Model
    model: str = "yolo26n.pt"
    confidence: float = 0.35
    iou: float = 0.5
    classes: Optional[List[int]] = None  # None = auto-detect; empty list = all

    # Output
    output_directory: Path = field(default_factory=lambda: Path("."))
    database: Optional[Path] = None  # None = <output_directory>/chirp.db

    # Tracking
    track_buffer: int = 8       # seconds a lost track is held
    in_threshold: Optional[int] = None  # frames; None = fps/2

    # Display
    show_display: bool = True

    # Highlights
    highlights: HighlightsConfig = field(default_factory=HighlightsConfig)

    # Zones (empty = single full-frame zone)
    zones: List[ZoneConfig] = field(default_factory=list)

    # Scheduler
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)

    # Location metadata
    location: Optional[str] = None

    def effective_database(self) -> Path:
        if self.database:
            return Path(self.database)
        return Path(self.output_directory) / "chirp.db"


def load_config(args: argparse.Namespace | None = None,
                config_file: Path | None = None) -> ChirpConfig:
    """Build a ChirpConfig by merging YAML file + CLI args."""
    cfg = ChirpConfig()

    # 1. Try to load YAML
    yaml_data: dict = {}
    if config_file and config_file.exists():
        with open(config_file) as f:
            yaml_data = yaml.safe_load(f) or {}

    _apply_yaml(cfg, yaml_data)

    # 2. CLI overrides
    if args:
        _apply_args(cfg, args)

    # Expand ~ in paths
    cfg.output_directory = Path(cfg.output_directory).expanduser()
    if cfg.database:
        cfg.database = Path(cfg.database).expanduser()

    return cfg


def _apply_yaml(cfg: ChirpConfig, data: dict) -> None:
    if "video_source" in data:
        cfg.video_source = str(data["video_source"])
    if "model" in data:
        cfg.model = str(data["model"])
    if "confidence" in data:
        cfg.confidence = float(data["confidence"])
    if "iou" in data:
        cfg.iou = float(data["iou"])
    if "classes" in data and data["classes"] is not None:
        cfg.classes = [int(c) for c in data["classes"]]
    if "output_directory" in data:
        cfg.output_directory = Path(data["output_directory"])
    if "database" in data:
        cfg.database = Path(data["database"])
    if "track_buffer" in data:
        cfg.track_buffer = int(data["track_buffer"])
    if "in_threshold" in data and data["in_threshold"] is not None:
        cfg.in_threshold = int(data["in_threshold"])
    if "show_display" in data:
        cfg.show_display = bool(data["show_display"])
    if "location" in data:
        cfg.location = str(data["location"])

    if "highlights" in data:
        h = data["highlights"] or {}
        cfg.highlights = HighlightsConfig(
            enabled=bool(h.get("enabled", True)),
            pre_roll=int(h.get("pre_roll", 5)),
            post_roll=int(h.get("post_roll", 3)),
        )

    if "zones" in data:
        cfg.zones = [
            ZoneConfig(
                name=str(z["name"]),
                polygon=[(int(p[0]), int(p[1])) for p in z["polygon"]],
            )
            for z in (data["zones"] or [])
        ]

    if "schedule" in data and data["schedule"]:
        s = data["schedule"]
        cfg.schedule = ScheduleConfig(
            start_time=s.get("start_time"),
            stop_time=s.get("stop_time"),
        )


def _apply_args(cfg: ChirpConfig, args: argparse.Namespace) -> None:
    if getattr(args, "video_source", None):
        cfg.video_source = args.video_source
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "confidence", None) is not None:
        cfg.confidence = args.confidence
    if getattr(args, "iou", None) is not None:
        cfg.iou = args.iou
    if getattr(args, "classes", None) is not None:
        if args.classes.lower() == "all":
            cfg.classes = []
        else:
            cfg.classes = [int(c.strip()) for c in args.classes.split(",")]
    if getattr(args, "output_directory", None):
        cfg.output_directory = Path(args.output_directory)
    if getattr(args, "location", None):
        cfg.location = args.location
    if getattr(args, "track_buffer", None) is not None:
        cfg.track_buffer = args.track_buffer
    if getattr(args, "in_threshold", None) is not None:
        cfg.in_threshold = args.in_threshold
    if getattr(args, "no_display", False):
        cfg.show_display = False
    if getattr(args, "no_highlights", False):
        cfg.highlights.enabled = False
