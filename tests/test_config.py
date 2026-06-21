"""Unit tests for ChirpConfig loading and CLI/YAML merging."""
import argparse
import textwrap
from pathlib import Path

import pytest
import yaml

from chirp.config import ChirpConfig, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "chirp_config.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def fake_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        video_source=None, model=None, confidence=None, iou=None,
        classes=None, output_directory=None, location=None,
        track_buffer=None, in_threshold=None, no_display=False,
        no_highlights=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

def test_defaults():
    cfg = load_config()
    assert cfg.video_source == "/dev/video0"
    assert cfg.model == "yolo26n.pt"
    assert cfg.confidence == pytest.approx(0.35)
    assert cfg.iou == pytest.approx(0.5)
    assert cfg.track_buffer == 8
    assert cfg.show_display is True
    assert cfg.highlights.enabled is True
    assert cfg.highlights.pre_roll == 5
    assert cfg.highlights.post_roll == 3
    assert cfg.zones == []
    assert cfg.schedule.start_time is None
    assert cfg.schedule.stop_time is None


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def test_yaml_overrides_defaults(tmp_path):
    p = write_yaml(tmp_path, """
        video_source: /dev/video1
        model: yolo26s.pt
        confidence: 0.55
        location: "51.5,-0.1"
    """)
    cfg = load_config(config_file=p)
    assert cfg.video_source == "/dev/video1"
    assert cfg.model == "yolo26s.pt"
    assert cfg.confidence == pytest.approx(0.55)
    assert cfg.location == "51.5,-0.1"


def test_yaml_highlights(tmp_path):
    p = write_yaml(tmp_path, """
        highlights:
          enabled: false
          pre_roll: 10
          post_roll: 5
    """)
    cfg = load_config(config_file=p)
    assert cfg.highlights.enabled is False
    assert cfg.highlights.pre_roll == 10
    assert cfg.highlights.post_roll == 5


def test_yaml_schedule(tmp_path):
    p = write_yaml(tmp_path, """
        schedule:
          start_time: "06:30"
          stop_time: "20:00"
    """)
    cfg = load_config(config_file=p)
    assert cfg.schedule.start_time == "06:30"
    assert cfg.schedule.stop_time == "20:00"


def test_yaml_zones(tmp_path):
    p = write_yaml(tmp_path, """
        zones:
          - name: feeder
            polygon:
              - [100, 150]
              - [500, 150]
              - [500, 400]
              - [100, 400]
          - name: perch
            polygon:
              - [200, 80]
              - [400, 80]
              - [400, 140]
              - [200, 140]
    """)
    cfg = load_config(config_file=p)
    assert len(cfg.zones) == 2
    assert cfg.zones[0].name == "feeder"
    assert cfg.zones[1].name == "perch"
    assert cfg.zones[0].polygon[0] == (100, 150)


def test_yaml_class_filter(tmp_path):
    p = write_yaml(tmp_path, """
        classes: [0, 1, 5]
    """)
    cfg = load_config(config_file=p)
    assert cfg.classes == [0, 1, 5]


def test_missing_yaml_file_uses_defaults():
    cfg = load_config(config_file=Path("/does/not/exist.yaml"))
    assert cfg.model == "yolo26n.pt"  # Default unchanged


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------

def test_cli_overrides_defaults():
    args = fake_args(video_source="/dev/video2", confidence=0.7)
    cfg = load_config(args=args)
    assert cfg.video_source == "/dev/video2"
    assert cfg.confidence == pytest.approx(0.7)


def test_cli_overrides_yaml(tmp_path):
    p = write_yaml(tmp_path, "model: yolo26s.pt\nconfidence: 0.4")
    args = fake_args(model="yolo26m.pt", confidence=0.8)
    cfg = load_config(args=args, config_file=p)
    assert cfg.model == "yolo26m.pt"   # CLI wins
    assert cfg.confidence == pytest.approx(0.8)


def test_cli_no_display_flag():
    args = fake_args(no_display=True)
    cfg = load_config(args=args)
    assert cfg.show_display is False


def test_cli_no_highlights_flag():
    args = fake_args(no_highlights=True)
    cfg = load_config(args=args)
    assert cfg.highlights.enabled is False


def test_cli_classes_parsed():
    args = fake_args(classes="0,2,14")
    cfg = load_config(args=args)
    assert cfg.classes == [0, 2, 14]


def test_cli_classes_all():
    args = fake_args(classes="all")
    cfg = load_config(args=args)
    assert cfg.classes == []


# ---------------------------------------------------------------------------
# effective_database
# ---------------------------------------------------------------------------

def test_effective_database_default(tmp_path):
    cfg = load_config(args=fake_args(output_directory=str(tmp_path)))
    assert cfg.effective_database() == tmp_path / "chirp.db"


def test_effective_database_explicit(tmp_path):
    p = write_yaml(tmp_path, f"database: {tmp_path}/custom.db")
    cfg = load_config(config_file=p)
    assert cfg.effective_database() == tmp_path / "custom.db"
