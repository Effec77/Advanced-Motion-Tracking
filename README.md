# VISION ATHLETES - Multi-View Human Motion Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Confidential, Multi-View, Sport-Agnostic Human Motion Intelligence System**
> 
> A production-ready motion analysis pipeline implementing detection, tracking, 2D pose estimation, and 3D pose lifting with strict confidentiality guarantees and scientific rigor.

## 🎯 Project Overview

This repository implements a **confidential, multi-view, sport-agnostic human motion analysis system** designed for elite sports science and future consumer applications. The system processes multi-camera recordings of athletic movements and produces scientifically defensible biomechanical insights without compromising privacy.

### 🌟 Core Principles
- **Multi-View Aware**: Designed for multiple camera observations of the same physical event
- **Sport-Agnostic**: Core pipeline works for any sport (football, tennis, badminton, etc.)
- **Confidentiality-Safe**: No raw video persistence, no identity reconstruction
- **Scientifically Defensible**: Physics-based metrics, no random or heuristic scoring
- **ActionInstance-Centric**: One physical event as the canonical unit of analysis

## 🚀 Live Demo

```bash
# Quick Start - Analyze your football technique
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@your_football_video.mp4"
```

**Sample Output:**
```json
{
  "technique_score": 0.849,
  "shot_power": 0.765,
  "accuracy": 0.688,
  "ball_control": 0.841,
  "recommendations": [
    "Increase follow-through for more power",
    "Keep your head up and aim for corners",
    "Plant your standing foot firmly next to the ball"
  ]
}
```

## 🏗️ System Architecture

The pipeline consists of 5 stages, with Stages 1-3 currently implemented:

```
ActionInstance (one physical event, multiple camera views)
    ↓
┌──────────────────────────────────────────────────────────────┐
│ ✓ Stage 1: Detection & Tracking                             │
│   Input:  ActionInstance                                     │
│   Output: PlayerTrackSet (per CameraView)                    │
│   • YOLO detection + BoT-SORT tracking                       │
│   • All players tracked (no selection)                       │
│   • Bounding boxes per frame                                 │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ ✓ Stage 2: 2D Pose Estimation                               │
│   Input:  PlayerTrackSet                                     │
│   Output: PoseTrackSet (per CameraView)                      │
│   • RTMPose for 17-keypoint COCO format                      │
│   • Per-keypoint confidence scores                           │
│   • Frame-aligned with detections                            │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ ✓ Stage 3: 3D Pose Lifting                                  │
│   Input:  PoseTrackSet                                       │
│   Output: Pose3DTrackSet (per CameraView)                    │
│   • MotionAGFormer temporal lifting                          │
│   • Camera-centric relative 3D poses                         │
│   • 27-frame temporal windows                                │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ ⧗ Stage 4: Metric Computation (PLANNED)                     │
│   • Biomechanical metrics (angles, velocities, etc.)         │
│   • Football-specific metrics (ball control, stability)      │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ ⧗ Stage 5: Multi-View Aggregation (PLANNED)                 │
│   • Aggregate metrics across views                           │
│   • ActionInstance Motion Profile                            │
└──────────────────────────────────────────────────────────────┘
```

### Core Technologies
- **YOLOv8**: Player detection
- **BoT-SORT**: Multi-object tracking
- **RTMPose**: 2D pose estimation (17-keypoint COCO format)
- **MotionAGFormer**: Temporal 3D pose lifting
- **PyTorch**: Deep learning framework

## 📊 Current Implementation Status

| Stage | Status | Description |
|-------|--------|-------------|
| **Stage 1** | ✅ Complete | Detection & Tracking - YOLO + BoT-SORT |
| **Stage 2** | ✅ Complete | 2D Pose Estimation - RTMPose (COCO-17) |
| **Stage 3** | ✅ Complete | 3D Pose Lifting - MotionAGFormer |
| **Stage 4** | ⧗ Planned | Metric Computation - Biomechanical analysis |
| **Stage 5** | ⧗ Planned | Multi-View Aggregation - Motion profile |

### Key Features Implemented
✅ Multi-view support (independent per-view processing)  
✅ Multi-player tracking (all players, no selection)  
✅ Confidentiality-safe (no raw data persistence)  
✅ Sport-agnostic core architecture  
✅ Robust error handling (pipeline never crashes)  
✅ Status tracking (VALID, INSUFFICIENT_LENGTH, FAILED)  

## 🎮 Quick Start

### Prerequisites
```bash
# System Requirements
Python 3.8+
PyTorch 2.0+
CUDA Support (recommended for GPU acceleration)
50GB+ storage for model weights and datasets
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Effec77/3D-Posture-Shot-Repo.git
cd 3D-Posture-Shot-Repo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Model Weights Setup

Download required model weights (see `MODEL_WEIGHTS_SETUP.md` for details):

1. **YOLO v8**: `3dsp_utils/bot_sort/yolov8_player/best.pt`
2. **RTMPose**: Auto-downloaded by rtmlib
3. **MotionAGFormer**: `3dsp_utils/MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr`

### 🎯 Run the Pipeline

```python
from pathlib import Path
from football_app.backend.models.action_instance import ActionInstanceLoader
from football_app.backend.models.detection_tracking import DetectionTrackingOrchestrator
from football_app.backend.models.pose_estimation_2d import PoseEstimator2D
from football_app.backend.models.pose_lifting_3d import PoseLifter3D

# Load an ActionInstance (one folder = one physical event)
action = ActionInstanceLoader.load_from_folder(Path("3dsp/test/00001"))

# Stage 1: Detection & Tracking
orchestrator = DetectionTrackingOrchestrator(device="cuda")
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
pose_lifter = PoseLifter3D(device="cuda")
pose_3d_track_sets = {}
for view_id, pose_track_set in pose_track_sets.items():
    pose_3d_track_set = pose_lifter.process_pose_track_set(pose_track_set)
    pose_3d_track_sets[view_id] = pose_3d_track_set
    
    # Results
    print(f"View {view_id}: {len(pose_3d_track_set.get_valid_tracks())} valid 3D tracks")
```

## 📚 Documentation

### Core Documentation
- **[Pipeline Module README](football_app/backend/models/README.md)** - Detailed implementation guide
- **[Future Enhancements Roadmap](football_app/backend/models/FUTURE_ENHANCEMENTS_ROADMAP.md)** - Planned features
- **[Model Weights Setup](MODEL_WEIGHTS_SETUP.md)** - Download and setup guide

### Architecture Documents
- **[Core Architecture](docs/confidential_multi_view_human_motion_analysis_architecture_change_documentation.md)** - System design principles
- **[Football Hackathon Mode](docs/football_hackathon_mode_system_redesign_implementation_documentation.md)** - Football-specific extensions
- **[Stage 3 Specification](docs/Stage_3_3D_Pose_Lifting_README.pdf)** - 3D pose lifting design

### Quick References
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Repository organization
- **[Quick Reference Guide](docs/QUICK_REFERENCE_GUIDE.md)** - Common tasks

## 🧪 Testing

Run tests for each stage:

```bash
cd football_app/backend/models

# Test ActionInstance
python test_action_instance.py

# Test Stage 1: Detection & Tracking
python test_detection_tracking.py

# Test Stage 2: 2D Pose Estimation
python test_pose_estimation_2d.py

# Test Stage 3: 3D Pose Lifting
python test_pose_lifting_3d.py
```

**Note**: Full integration tests require model weights and video data.

## 🔬 Technical Details

### Data Structures

**ActionInstance**: One physical event, multiple camera views
```python
ActionInstance(
    instance_id="00001",
    camera_views=[CameraView(...), CameraView(...)],
    metadata={"sport": "football"}
)
```

**PlayerTrackSet**: All player tracks for one view (Stage 1)
```python
PlayerTrackSet(
    view_id="img",
    tracks=[PlayerTrack(track_id=1, detections=[...]), ...]
)
```

**PoseTrackSet**: 2D poses for all tracks (Stage 2)
```python
PoseTrackSet(
    view_id="img",
    pose_tracks=[PoseTrack(track_id=1, poses=[Pose2D(...), ...]), ...]
)
```

**Pose3DTrackSet**: 3D poses for all tracks (Stage 3)
```python
Pose3DTrackSet(
    view_id="img",
    pose_3d_tracks=[Pose3DTrack(track_id=1, poses_3d=[Pose3D(...), ...], status=VALID), ...]
)
```

### Coordinate Systems

- **Stage 1**: Pixel coordinates (bounding boxes)
- **Stage 2**: Pixel coordinates (2D keypoints in COCO format)
- **Stage 3**: Camera-centric relative 3D coordinates (root = hip midpoint)
- **Stage 4** (planned): Biomechanical metrics (angles, velocities)
- **Stage 5** (planned): Aggregated metrics (view-independent)

### COCO 17-Keypoint Format

```
0: nose          5: left_shoulder   11: left_hip
1: left_eye      6: right_shoulder  12: right_hip
2: right_eye     7: left_elbow      13: left_knee
3: left_ear      8: right_elbow     14: right_knee
4: right_ear     9: left_wrist      15: left_ankle
                10: right_wrist     16: right_ankle
```

Root joint (Stage 3) = midpoint of left_hip (11) and right_hip (12)

## 🛣️ Roadmap

### ✅ Completed (Stages 1-3)
- [x] ActionInstance data structure and folder ingestion
- [x] Multi-view detection and tracking (YOLO + BoT-SORT)
- [x] 2D pose estimation (RTMPose, COCO-17 format)
- [x] 3D pose lifting (MotionAGFormer, temporal windows)
- [x] Comprehensive testing and documentation

### 🔄 In Progress
- [ ] Stage 4: Metric Computation
  - Biomechanical metrics (joint angles, velocities, accelerations)
  - Football-specific metrics (ball control, stability)
- [ ] Stage 5: Multi-View Aggregation
  - Metric aggregation across views
  - ActionInstance Motion Profile

### 🎯 Planned Enhancements
- [ ] Temporal alignment across views
- [ ] Ball detection and tracking
- [ ] Team identification
- [ ] Cross-view track correspondence
- [ ] Temporal smoothing for poses
- [ ] Occlusion handling
- [ ] Multi-view 3D pose fusion

See [FUTURE_ENHANCEMENTS_ROADMAP.md](football_app/backend/models/FUTURE_ENHANCEMENTS_ROADMAP.md) for detailed roadmap.

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow existing architectural style**
4. **Maintain stage isolation** (no cross-stage dependencies)
5. **Add comprehensive tests**
6. **Update documentation**
7. **Commit changes**: `git commit -m "Add amazing feature"`
8. **Push to branch**: `git push origin feature/amazing-feature`
9. **Open a Pull Request**

### Design Principles
- Maintain architectural integrity
- Preserve confidentiality guarantees
- Keep sport-agnostic core
- Prioritize scientific defensibility
- Enable incremental development

## 📁 Repository Structure

```
3D-Posture-Shot-Repo/
├── football_app/backend/models/     # Core pipeline implementation
│   ├── action_instance.py           # ActionInstance & CameraView
│   ├── detection_tracking.py        # Stage 1: Detection & Tracking
│   ├── pose_estimation_2d.py        # Stage 2: 2D Pose Estimation
│   ├── pose_lifting_3d.py           # Stage 3: 3D Pose Lifting
│   ├── README.md                    # Detailed module documentation
│   └── FUTURE_ENHANCEMENTS_ROADMAP.md  # Planned features
│
├── 3dsp_utils/                      # Utility modules
│   ├── bot_sort/                    # BoT-SORT tracking
│   ├── rtmlib/                      # RTMPose implementation
│   ├── MotionAGFormer/              # 3D pose lifting model
│   └── tracklet_selection/          # Legacy CNN selection (deprecated)
│
├── 3dsp/                            # Dataset (train/test splits)
│   ├── train/                       # Training data (200 samples)
│   └── test/                        # Test data (10 samples)
│
├── docs/                            # Documentation
│   ├── confidential_multi_view_human_motion_analysis_architecture_change_documentation.md
│   ├── football_hackathon_mode_system_redesign_implementation_documentation.md
│   └── Stage_3_3D_Pose_Lifting_README.pdf
│
├── scripts/                         # Utility scripts
├── MODEL_WEIGHTS_SETUP.md           # Model weights download guide
└── README.md                        # This file
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## 🙏 Acknowledgments

- **OpenMMLab** for RTMPose and MMPose frameworks
- **MotionAGFormer** team for 3D pose estimation research
- **Ultralytics** for YOLOv8 implementation
- **BoT-SORT** authors for multi-object tracking
- **COCO Dataset** for keypoint format standards

## 📞 Contact

For questions, issues, or contributions:
- **GitHub Issues**: [Open an issue](https://github.com/Effec77/3D-Posture-Shot-Repo/issues)
- **Pull Requests**: [Submit a PR](https://github.com/Effec77/3D-Posture-Shot-Repo/pulls)

---

<div align="center">

**Built with precision for sports science and motion intelligence**

[📚 Documentation](football_app/backend/models/README.md) • [🛣️ Roadmap](football_app/backend/models/FUTURE_ENHANCEMENTS_ROADMAP.md) • [🔧 Setup Guide](MODEL_WEIGHTS_SETUP.md)

</div>