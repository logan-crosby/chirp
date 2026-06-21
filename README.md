# chirp

Real-time bird detection, tracking, and species counting for backyard feeders.
Point a camera at your feeder, run `chirp`, and get a live count of arrivals
by species, visit-duration stats, automatic highlight clips, and a persistent
SQLite database you can query across days and weeks.

## Demo

[Bird detection, tracking, and counting](https://youtu.be/nCKLM8BG1l4)

---

## Features

| Feature | Details |
|---|---|
| **Detection** | YOLO11 (or any Ultralytics model) with confidence + NMS filtering |
| **Tracking** | ByteTrack multi-object tracker — handles occlusion and re-identification |
| **Species ID** | Majority-vote over full track lifetime; plug in a fine-tuned model for per-species accuracy |
| **Counting** | Zone-based arrival / departure events with configurable sensitivity |
| **Multi-zone** | Define multiple polygon zones in `chirp_config.yaml` (feeder, perch, bath, …) |
| **Database** | SQLite — query history across sessions, species summaries, hourly activity charts |
| **Highlights** | Auto-saves MP4 clips of each visit with configurable pre- and post-roll |
| **Schedule** | Optional time window (e.g. 06:00–20:00) — sleeps outside hours, wakes automatically |
| **Headless** | `show_display: false` for unattended Raspberry Pi / SSH deployments |
| **Analytics** | `chirp --stats` prints species summary + hourly chart from the database |

---

## Requirements

- Python 3.11+
- A camera (`/dev/video0`) or a video file / RTSP stream
- NVIDIA GPU strongly recommended; CPU works but will be slower

---

## Installation

```bash
# Clone
git clone https://github.com/logan-crosby/chirp.git
cd chirp

# Install (Poetry)
pip install poetry
poetry install

# Or install directly with pip
pip install -e .
```

Install CUDA-enabled PyTorch for GPU acceleration (optional but recommended):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Quick start

```bash
# 1. Copy and edit the example config
cp chirp_config.yaml.example chirp_config.yaml
$EDITOR chirp_config.yaml   # set video_source, location, zones, …

# 2. Run
chirp --config chirp_config.yaml

# Or without a config file (all defaults, COCO bird detection)
chirp -v /dev/video0 -l 40.71,-74.00

# Headless (no window) — for unattended yard use
chirp --config chirp_config.yaml --no-display
```

Press **Q** in the preview window to stop.

---

## Configuration

All options can be set in `chirp_config.yaml` (see `chirp_config.yaml.example`)
or passed as CLI flags (CLI overrides the file).

### Key options

```yaml
video_source: /dev/video0       # camera device or video file path / RTSP URL
model: yolo11n.pt               # model name (auto-downloaded) or path to .pt
confidence: 0.35                # detection confidence threshold
location: "40.7128,-74.0060"    # feeder GPS (LAT,LON) — stored in metadata

output_directory: ~/chirp_data  # root for database + highlight clips
show_display: true              # set false for headless operation

highlights:
  enabled: true
  pre_roll: 5    # seconds before bird arrives
  post_roll: 3   # seconds after bird leaves

schedule:
  start_time: "06:00"
  stop_time:  "20:30"

zones:
  - name: feeder
    polygon:
      - [120, 180]
      - [520, 180]
      - [520, 420]
      - [120, 420]
```

### Finding polygon coordinates

Capture a still frame from your camera, then:

```bash
python -c "
import cv2
img = cv2.imread('frame.jpg')
cv2.namedWindow('pick')
cv2.setMouseCallback('pick', lambda e,x,y,_,__: print(x,y) if e==1 else None)
cv2.imshow('pick', img)
cv2.waitKey(0)
"
```

Click the corners of your feeder area and note the printed `x y` coordinates.

---

## Choosing a model

### Generic bird detection (counts only)

The default `yolo11n.pt` is trained on COCO and detects the generic "bird"
class. It gives you arrival counts but can't tell species apart.

| Model | Speed | Accuracy | Notes |
|---|---|---|---|
| `yolo11n.pt` | Fastest | Lower | Good for RPi / CPU |
| `yolo11s.pt` | Fast | Moderate | Balanced |
| `yolo11m.pt` | Moderate | Higher | Needs GPU |

### Per-species identification

Fine-tune YOLO11 on a bird-species dataset, or download a community model:

| Resource | Notes |
|---|---|
| [NABirds dataset](https://dl.allaboutbirds.org/nabirds) | 400+ North American species, 48,000 images |
| [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) | 200 species, widely used for fine-grained classification |
| [Roboflow Universe](https://universe.roboflow.com/search?q=class:bird+trained+model) | Pre-trained community models, ready to download |

Train your own with Ultralytics:

```bash
yolo train data=nabirds.yaml model=yolo11s.pt epochs=100 imgsz=640
```

Then pass the result:

```yaml
model: runs/detect/train/weights/best.pt
classes: [0, 1, 2, …]   # all your bird class IDs
```

---

## Viewing statistics

```bash
# Species summary + hourly activity chart (last 7 days)
chirp --stats

# Last 30 days
chirp --stats --stats-days 30

# Query the database directly
sqlite3 ~/chirp_data/chirp.db \
  "SELECT class_name, COUNT(*) visits FROM bird_events
   WHERE event_type='enter' GROUP BY class_name ORDER BY visits DESC;"
```

---

## Output files

```
~/chirp_data/
├── chirp.db                        # SQLite database (all sessions)
└── chirp_sessions/
    └── highlights/
        ├── highlight_20240601T120301_chickadee_t42.mp4
        └── …
```

The database schema:

```sql
sessions      -- one row per run (model, location, settings)
zone_defs     -- polygon zones registered per session
bird_events   -- enter/exit events (class_name, tracker_id, occurred_at, …)
```

---

## Running tests

```bash
pytest
```

---

## Architecture

```
chirp/
├── birdcounter.py   BirdCounter class + CLI entry-point
├── zone_monitor.py  Arrival/departure logic with majority-vote species ID
├── config.py        YAML + CLI config merging (ChirpConfig)
├── database.py      SQLite persistence (ChirpDatabase)
├── metrics.py       Live SessionMetrics + historical DB queries
├── highlights.py    Pre-roll video clip recorder (HighlightRecorder)
└── scheduler.py     Time-window guard (is_within_window)
```

---

## License

MIT
