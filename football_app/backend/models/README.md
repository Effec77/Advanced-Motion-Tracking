# Motion Analysis Pipeline - Models Module

## Overview

This module implements a **confidential, multi-view, sport-agnostic human motion analysis pipeline** consisting of three core stages: Detection & Tracking, 2D Pose Estimation, and 3D Pose Lifting.

**Key Principles:**
- Multi-view aware by design
- Sport-agnostic core with sport-specific adapters
- Confidentiality-safe (no raw video persistence)
- Scientifically defensible (physics-based metrics)
- ActionInstance as canonical unit of analysis

---

## Pipeline Architecture

```
ActionInstance (one physical event, multiple camera views)
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Detection & Tracking                           │
│ Input:  ActionInstance                                  │
│ Output: PlayerTrackSet (per CameraView)                 │
│ - YOLO detection + BoT-SORT tracking                    │
│ - All players tracked (no selection)                    │
│ - Bounding boxes per frame                              │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 2: 2D Pose Estimation                             │
│ Input:  PlayerTrackSet                                  │
│ Output: PoseTrackSet (per CameraView)                   │
│ - RTMPose for 17-keypoint COCO format                   │
│ - Per-keypoint confidence scores                        │
│ - Frame-aligned with detections                         │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 3: 3D Pose Lifting                                │
│ Input:  PoseTrackSet                                    │
│ Output: Pose3DTrackSet (per CameraView)                 │
│ - MotionAGFormer temporal lifting                       │
│ - Camera-centric relative 3D poses                      │
│ - 243-frame temporal windows                            │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Motion Metrics & Stability Engine              │
│ Input:  Pose3DTrackSet                                  │
│ Output: MetricTrackSet (per CameraView)                 │
│ - Kinematic metrics (velocity, acceleration, jerk)      │
│ - Stability index and symmetry metrics                  │
│ - Frame-level and track-level aggregations              │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 5: Multi-View Aggregation (PLANNED)               │
│ - Aggregate metrics across views                        │
│ - ActionInstance Motion Profile                         │
└─────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
football_app/backend/models/
├── README.md                           # This file
├── FUTURE_ENHANCEMENTS_ROADMAP.md      # Planned enhancements
│
├── action_instance.py                  # ActionInstance & CameraView data structures
├── detection_tracking.py               # Stage 1: Detection & Tracking
├── pose_estimation_2d.py               # Stage 2: 2D Pose Estimation
├── pose_lifting_3d.py                  # Stage 3: 3D Pose Lifting
├── motion_metrics.py                   # Stage 4: Motion Metrics & Stability
│
├── test_action_instance.py             # Tests for ActionInstance
├── test_detection_tracking.py          # Tests for Stage 1
├── test_pose_estimation_2d.py          # Tests for Stage 2
├── test_pose_lifting_3d.py             # Tests for Stage 3
└── test_motion_metrics.py              # Tests for Stage 4
```

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch opencv-python numpy ultralytics

# Ensure model weights are available
# - YOLO: 3dsp_utils/bot_sort/yolov8_player/best.pt
# - RTMPose: Auto-downloaded by rtmlib
# - MotionAGFormer: 3dsp_utils/MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr
```

### Basic Usage

```python
from pathlib import Path
from action_instance import ActionInstanceLoader
from detection_tracking import DetectionTrackingOrchestrator
from pose_estimation_2d import PoseEstimator2D
from pose_lifting_3d import PoseLifter3D

# Load an ActionInstance (one folder = one physical event)
action = ActionInstanceLoader.load_from_folder(Path("3dsp/test/00001"))

# Stage 1: Detection & Tracking
orchestrator = DetectionTrackingOrchestrator(
    yolo_model_path="3dsp_utils/bot_sort/yolov8_player/best.pt",
    device="cuda"
)
track_sets = orchestrator.process_action_instance(action)

# Stage 2: 2D Pose Estimation
pose_estimator = PoseEstimator2D(device="cuda")
pose_track_sets = {}
for view_id, player_track_set in track_sets.items():
    camera_view = action.get_view_by_id(view_id)
    pose_track_set = pose_estimator.process_player_track_set(
        player_track_set, camera_view
    )
    pose_track_sets[view_id] = pose_track_set

# Stage 3: 3D Pose Lifting
pose_lifter = PoseLifter3D(
    model_path="3dsp_utils/MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr",
    device="cuda"
)
pose_3d_track_sets = {}
for view_id, pose_track_set in pose_track_sets.items():
    pose_3d_track_set = pose_lifter.process_pose_track_set(pose_track_set)
    pose_3d_track_sets[view_id] = pose_3d_track_set

# Stage 4: Motion Metrics & Stability
from motion_metrics import MotionMetricsEngine

metrics_engine = MotionMetricsEngine()
metric_track_sets = {}
for view_id, pose_3d_track_set in pose_3d_track_sets.items():
    metric_track_set = metrics_engine.process_pose_3d_track_set(pose_3d_track_set)
    metric_track_sets[view_id] = metric_track_set

# Access results
for view_id, metric_track_set in metric_track_sets.items():
    complete_tracks = metric_track_set.get_complete_tracks()
    print(f"View {view_id}: {len(complete_tracks)} complete metric tracks")
    
    # Access metrics for first track
    if complete_tracks:
        mt = complete_tracks[0]
        sm = mt.summary_metrics
        print(f"  V_avg: {sm.V_avg:.3f}, Stability: {sm.Stability:.3f}")
```

---

## Core Data Structures

### ActionInstance & CameraView

**ActionInstance**: One real-world physical movement event, possibly observed by multiple cameras.

```python
from action_instance import ActionInstance, CameraView, MediaType

# ActionInstance contains multiple CameraViews
action = ActionInstance(
    instance_id="kick_001",
    camera_views=[view1, view2, view3],
    metadata={"sport": "football"}
)
```

**Key Principle**: Multiple videos ≠ multiple actions. Multiple videos = multiple observations of the same action.

---

### Stage 1: Detection & Tracking

**PlayerTrack**: Temporal sequence of bounding box detections for one player.

```python
from detection_tracking import PlayerTrack, BoundingBox, PlayerDetection

# PlayerTrack contains detections across frames
track = PlayerTrack(track_id=1)
track.add_detection(PlayerDetection(
    frame_id=1,
    bbox=BoundingBox(x=100, y=200, w=50, h=100, confidence=0.95),
    confidence=0.95
))
```

**PlayerTrackSet**: Collection of all player tracks for one camera view.

```python
track_set = PlayerTrackSet(view_id="cam_1")
track_set.add_track(track)
```

---

### Stage 2: 2D Pose Estimation

**Pose2D**: 17-keypoint 2D pose in COCO format with confidence scores.

```python
from pose_estimation_2d import Pose2D, Keypoint2D

# COCO 17-keypoint format
# 0: nose, 1: left_eye, 2: right_eye, ..., 16: right_ankle
pose_2d = Pose2D(
    frame_id=1,
    keypoints=[Keypoint2D(x=100, y=200, confidence=0.9) for _ in range(17)],
    bbox=bbox
)
```

**PoseTrack**: Temporal sequence of 2D poses for one player.

```python
from pose_estimation_2d import PoseTrack

pose_track = PoseTrack(track_id=1, player_track=player_track)
pose_track.add_pose(pose_2d)
```

---

### Stage 3: 3D Pose Lifting

**Pose3D**: 17-joint 3D pose in camera-centric coordinates, relative to root joint.

```python
from pose_lifting_3d import Pose3D, Joint3D

# Root joint = midpoint of left_hip (11) and right_hip (12)
pose_3d = Pose3D(
    frame_id=1,
    joints=[Joint3D(x=0.1, y=0.2, z=0.3) for _ in range(17)]
)
```

**Pose3DTrack**: Temporal sequence of 3D poses with status tracking.

```python
from pose_lifting_3d import Pose3DTrack, Pose3DStatus

pose_3d_track = Pose3DTrack(
    track_id=1,
    status=Pose3DStatus.VALID,  # or INSUFFICIENT_LENGTH, FAILED
    pose_track=pose_track
)
```

---

### Stage 4: Motion Metrics & Stability

**MetricTrack**: Complete metric set for a single Pose3DTrack.

```python
from motion_metrics import MetricTrack, MetricStatus, SummaryMetrics

metric_track = MetricTrack(
    track_id=1,
    status=MetricStatus.COMPLETE,  # or PARTIAL, FAILED
    summary_metrics=SummaryMetrics(
        V_avg=0.073,      # Average velocity
        V_peak=0.561,     # Peak velocity
        A_var=0.002,      # Acceleration variance
        J_avg=0.025,      # Mean jerk
        Stability=0.975,  # Stability index (0 < S ≤ 1)
        Symmetry=0.818    # Symmetry metric (0 ≤ Sym ≤ 1)
    )
)
```

**MetricTrackSet**: Collection of all metric tracks for one camera view.

```python
from motion_metrics import MetricTrackSet

metric_track_set = MetricTrackSet(view_id="cam_1")
metric_track_set.add_metric_track(metric_track)

# Get complete tracks only
complete_tracks = metric_track_set.get_complete_tracks()
```

---

## Key Features

### Multi-View Support

- Each camera view processed independently
- No cross-view fusion at Stages 1-3
- Multi-view aggregation planned for Stage 5

### Multi-Player Support

- All detected players tracked simultaneously
- No player selection or prioritization
- Compatible with football multi-player scenarios

### Confidentiality-Safe

- No raw video persistence
- No frame or image saving
- No identity reconstruction
- Only derived metrics stored (future stages)

### Sport-Agnostic Core

- Core pipeline works for any sport
- Football-specific logic in separate adapters
- Easy to extend to badminton, tennis, etc.

### Robust Error Handling

- Tracks with insufficient length marked as INSUFFICIENT_LENGTH
- Model failures marked as FAILED
- Pipeline never crashes due to one bad track

---

## Testing

Run tests for each stage:

```bash
# Test ActionInstance
python test_action_instance.py

# Test Stage 1: Detection & Tracking
python test_detection_tracking.py

# Test Stage 2: 2D Pose Estimation
python test_pose_estimation_2d.py

# Test Stage 3: 3D Pose Lifting
python test_pose_lifting_3d.py

# Test Stage 4: Motion Metrics & Stability
python test_motion_metrics.py
```

**Note**: Full integration tests require model weights and video data.

---

## Configuration

### Stage 1: Detection & Tracking

```python
orchestrator = DetectionTrackingOrchestrator(
    yolo_model_path="path/to/yolo.pt",
    device="cuda",                    # or "cpu"
    track_high_thresh=0.6,            # High confidence threshold
    track_low_thresh=0.1,             # Low confidence threshold
    new_track_thresh=0.7,             # New track threshold
    track_buffer=30,                  # Frames to keep lost tracks
    match_thresh=0.8                  # Matching threshold
)
```

### Stage 2: 2D Pose Estimation

```python
pose_estimator = PoseEstimator2D(
    pose_model_path=None,             # Auto-download if None
    device="cuda",                    # or "cpu"
    backend="onnxruntime",            # or "openvino"
    pose_input_size=(288, 384)        # Model input size
)
```

### Stage 3: 3D Pose Lifting

```python
pose_lifter = PoseLifter3D(
    model_path="path/to/motionagformer.pth.tr",
    device="cuda",                    # or "cpu"
    temporal_window=243,              # Temporal window size (matches checkpoint)
    confidence_threshold=0.2          # Mask joints below this
)
```

### Stage 4: Motion Metrics & Stability

```python
from motion_metrics import MotionMetricsEngine

metrics_engine = MotionMetricsEngine(
    epsilon=1e-8                      # Small constant for numerical stability
)
```

**Note**: Stage 4 has no configurable parameters beyond epsilon. All computations follow fixed mathematical formulas.

---

## Performance Considerations

### GPU Acceleration

- **Recommended**: CUDA-enabled GPU for real-time processing
- **Fallback**: CPU inference supported (slower)

### Memory Usage

- Frames processed sequentially (memory-efficient)
- No intermediate frame storage
- Lazy-loading of models

### Processing Time (Approximate)

- **Stage 1**: ~30 FPS (GPU), ~5 FPS (CPU)
- **Stage 2**: ~20 FPS (GPU), ~3 FPS (CPU)
- **Stage 3**: ~15 FPS (GPU), ~2 FPS (CPU)
- **Stage 4**: ~12,000 FPS (CPU) - negligible overhead

**Note**: Times vary based on hardware, video resolution, and number of players. Stage 4 is extremely fast as it only performs mathematical computations on existing 3D poses.

---

## Limitations & Future Work

### Current Limitations

1. **No temporal alignment**: Views processed independently
2. **No ball tracking**: Only tracks players (persons)
3. **No cross-view correspondence**: Track IDs are view-specific
4. **Fixed temporal window**: 243 frames for 3D lifting
5. **No cross-camera metric fusion**: Stage 5 not yet implemented
6. **No metric interpretation**: Raw metrics only, no semantic analysis

### Planned Enhancements

See `FUTURE_ENHANCEMENTS_ROADMAP.md` for detailed roadmap including:

- **Stage 1**: Temporal alignment, ball tracking, team identification
- **Stage 2**: Temporal smoothing, occlusion handling
- **Stage 3**: Adaptive windows, multi-view fusion
- **Stage 4**: Temporal smoothing, biomechanical metrics (angles, limb lengths)
- **Stage 5**: Multi-view aggregation, motion profile, cross-camera fusion

---

## Troubleshooting

### Common Issues

**Error: "Cannot open video"**
- Verify video file path is correct
- Check video file is not corrupted
- Ensure OpenCV can read the video format

**Error: "YOLO model not found"**
- Download YOLO weights: `3dsp_utils/bot_sort/yolov8_player/best.pt`
- Or use default YOLOv8: Will auto-download

**Error: "MotionAGFormer dependencies not available"**
- Ensure `3dsp_utils/MotionAGFormer/` directory exists
- Check model weights are present

**Low tracking quality**
- Adjust `track_high_thresh` and `track_low_thresh`
- Ensure good lighting and clear view of subjects
- Check video resolution and quality

**Many tracks marked as INSUFFICIENT_LENGTH**
- Tracks need ≥27 frames for 3D lifting
- Check Stage 1 tracking quality
- Verify video has sufficient frames

---

## Architecture Compliance

This implementation strictly adheres to:

- **Core Architecture**: `docs/confidential_multi_view_human_motion_analysis_architecture_change_documentation.md`
- **Football Hackathon Mode**: `docs/football_hackathon_mode_system_redesign_implementation_documentation.md`
- **Stage 3 Specification**: `Stage 3: 3D Pose Lifting (Design Specification).pdf`

### Key Compliance Points

✓ ActionInstance as canonical unit of analysis  
✓ Multi-view aware (independent per-view processing)  
✓ Sport-agnostic core architecture  
✓ Confidentiality-safe (no raw data persistence)  
✓ No player selection or prioritization  
✓ Physics-based, scientifically defensible  
✓ Robust error handling  

---

## Contributing

When adding new features or enhancements:

1. **Follow existing architectural style**
2. **Maintain stage isolation** (no cross-stage dependencies)
3. **Preserve data contracts** between stages
4. **Add comprehensive tests**
5. **Update documentation**
6. **Ensure confidentiality guarantees**

---

## References

### Documentation
- **Core Architecture**: `docs/confidential_multi_view_human_motion_analysis_architecture_change_documentation.md`
- **Football Hackathon Mode**: `docs/football_hackathon_mode_system_redesign_implementation_documentation.md`
- **Future Enhancements**: `FUTURE_ENHANCEMENTS_ROADMAP.md`
- **Model Weights Setup**: `MODEL_WEIGHTS_SETUP.md` (root directory)

### Research Papers
- **BoT-SORT**: [Byte-Track + Optical Flow + ReID](https://arxiv.org/abs/2206.14651)
- **RTMPose**: [Real-Time Multi-Person Pose Estimation](https://arxiv.org/abs/2303.07399)
- **MotionAGFormer**: [Enhancing 3D Human Pose Estimation](https://arxiv.org/abs/2405.11429)
- **COCO Keypoints**: [COCO Keypoint Detection](https://cocodataset.org/#keypoints-2020)

---

## License

See `LICENSE.md` in the root directory.

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub.
