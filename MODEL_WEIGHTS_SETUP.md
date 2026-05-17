# Model Weights Setup Guide

## Overview

Due to GitHub's file size limitations (100MB), large model weight files are not included in this repository. This guide explains which weights are needed and how to obtain them.

## Required Model Weights

### 1. YOLO v3 Weights (236.52 MB)
**Location**: `3dsp_utils/MotionAGFormer/checkpoint/yolov3.weights`

**Purpose**: Object detection for player/person detection

**Download**:
```bash
cd 3dsp_utils/MotionAGFormer/checkpoint/
wget https://pjreddie.com/media/files/yolov3.weights
```

Or download manually from: https://pjreddie.com/darknet/yolo/

### 2. MotionAGFormer Weights (135.36 MB)
**Location**: `3dsp_utils/MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr`

**Purpose**: 3D pose estimation from 2D keypoints

**Download**:
Follow the instructions in the MotionAGFormer repository:
https://github.com/TaatiTeam/MotionAGFormer

Or check the checkpoint directory for download links.

### 3. Tracklet Selection CNN Weights (28.8 MB) - OPTIONAL
**Location**: `3dsp_utils/tracklet_selection/params/best.pth`

**Purpose**: CNN-based player selection (DEPRECATED in Hackathon Mode)

**Status**: ⚠️ **NOT REQUIRED** for the new architecture

According to the Football Hackathon Mode documentation:
> "Major Change 2: Remove CNN-Based Best Player Selection"
> "The CNN-based selector is **disabled** in Hackathon Mode."

This weight file is only needed if you're running the legacy `demo.py` script.

## Current Status

✓ **Local weights present**:
- `yolov3.weights` - Present locally
- `motionagformer-b-h36m.pth.tr` - Present locally

⚠️ **Missing locally**:
- `best.pth` - Not present (but not needed for new architecture)

## Setup Instructions

### For New Users Cloning This Repository

1. Clone the repository:
```bash
git clone https://github.com/Effec77/3D-Posture-Shot-Repo.git
cd 3D-Posture-Shot-Repo
```

2. Download YOLO v3 weights:
```bash
cd 3dsp_utils/MotionAGFormer/checkpoint/
wget https://pjreddie.com/media/files/yolov3.weights
cd ../../..
```

3. Download MotionAGFormer weights:
```bash
# Follow instructions from MotionAGFormer repository
# Place the downloaded file in: 3dsp_utils/MotionAGFormer/checkpoint/
```

4. (Optional) If using legacy demo.py, download tracklet selection weights:
```bash
mkdir -p 3dsp_utils/tracklet_selection/params/
# Contact repository maintainer for best.pth file
```

## Why Are These Files Not in Git?

GitHub has a strict 100MB file size limit. Large model weights should be:
- Downloaded separately
- Stored using Git LFS (Large File Storage)
- Hosted on model repositories (Hugging Face, Google Drive, etc.)

## .gitignore Configuration

The following patterns are configured to prevent accidental commits of large files:

```gitignore
# AI Models & Weights
*.pth
*.pt
*.onnx
*.pkl
*.h5
*.pb
*.weights
checkpoints/
models/weights/
3dsp_utils/MotionAGFormer/checkpoint/
3dsp_utils/bot_sort/yolov8_player/best.pt
```

## Architecture Compliance

According to the **Confidential Multi-View Human Motion Analysis Architecture**:
- Model weights are **temporary processing artifacts**
- Only **derived metrics** should be stored long-term
- No raw video or model outputs should be persisted

The new ActionInstance-based architecture minimizes dependency on specific model weights by focusing on metric aggregation rather than raw predictions.

## Troubleshooting

### Error: "FileNotFoundError: yolov3.weights"
Download the YOLO v3 weights as described above.

### Error: "FileNotFoundError: motionagformer-b-h36m.pth.tr"
Download the MotionAGFormer weights from the official repository.

### Error: "FileNotFoundError: best.pth"
This is expected if you're using the new architecture. The CNN-based player selection is deprecated. If you need to run legacy code, contact the repository maintainer.

## Contact

For access to model weights or questions about setup, please open an issue on GitHub.
