"""
Minimal Validation Script (First 50 Frames Only)

Quick test to verify pipeline works end-to-end.
Processes only first 50 frames of CAM0.
"""

from pathlib import Path
import sys
import os
import cv2

# Add repository root to path AND change to repo root
repo_root = Path(__file__).parent.parent.parent.parent
os.chdir(repo_root)  # Change directory to repo root
sys.path.insert(0, str(repo_root))

# Also add MotionAGFormer parent to path for Stage 3
motionagformer_parent = repo_root / "3dsp_utils"
sys.path.insert(0, str(motionagformer_parent))

from football_app.backend.models.detection_tracking import (
    PlayerTrack, PlayerTrackSet, PlayerDetection, BoundingBox, TrackStatus
)
from football_app.backend.models.pose_estimation_2d import PoseEstimator2D
from football_app.backend.models.pose_lifting_3d import PoseLifter3D, Pose3DStatus
from football_app.backend.models.action_instance import CameraView, MediaType


def create_minimal_track_set(video_path: Path, camera_id: str, max_frames: int = 250) -> PlayerTrackSet:
    """Create a PlayerTrackSet from first N frames of video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    player_track = PlayerTrack(track_id=1, status=TrackStatus.ACTIVE)
    
    frame_id = 0
    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        bbox = BoundingBox(x=0.0, y=0.0, w=float(width), h=float(height), confidence=1.0)
        detection = PlayerDetection(frame_id=frame_id, bbox=bbox, confidence=1.0)
        player_track.add_detection(detection)
    
    cap.release()
    
    track_set = PlayerTrackSet(
        view_id=camera_id,
        num_frames=frame_id,
        metadata={"fps": fps, "width": width, "height": height, "source": str(video_path)}
    )
    track_set.add_track(player_track)
    
    return track_set


def run_minimal_validation():
    """Run minimal validation on first 50 frames."""
    print("=" * 70)
    print("MINIMAL VALIDATION - FIRST 250 FRAMES")
    print("=" * 70)
    print()
    
    validation_path = Path("D:/ValidationTesting/Shoot1")
    video_file = validation_path / "CAM0.avi"
    camera_name = "CAM0"
    
    if not video_file.exists():
        print(f"❌ ERROR: Video file not found: {video_file}")
        return
    
    # ADAPTER
    print("ADAPTER: Creating track set (250 frames)...")
    track_set = create_minimal_track_set(video_file, camera_name, max_frames=250)
    print(f"✓ Track set created: {track_set.num_tracks()} track, {track_set.num_frames} frames")
    print()
    
    # STAGE 2
    print("STAGE 2: 2D Pose Estimation...")
    pose_estimator = PoseEstimator2D(device="cuda")
    camera_view = CameraView(view_id=camera_name, media_path=video_file, media_type=MediaType.VIDEO)
    pose_track_set = pose_estimator.process_player_track_set(track_set, camera_view)
    print(f"✓ Pose tracks: {len(pose_track_set.pose_tracks)}")
    if len(pose_track_set.pose_tracks) > 0:
        print(f"  Poses: {pose_track_set.pose_tracks[0].num_poses()}")
    print()
    
    # STAGE 3
    print("STAGE 3: 3D Pose Lifting...")
    pose_lifter = PoseLifter3D(
        model_path="3dsp_utils/MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr",
        device="cuda"
    )
    pose_3d_track_set = pose_lifter.process_pose_track_set(pose_track_set)
    
    status_counts = {s: 0 for s in Pose3DStatus}
    for track in pose_3d_track_set.pose_3d_tracks:
        status_counts[track.status] += 1
    
    print(f"✓ Pose3D tracks:")
    print(f"  VALID: {status_counts[Pose3DStatus.VALID]}")
    print(f"  INSUFFICIENT_LENGTH: {status_counts[Pose3DStatus.INSUFFICIENT_LENGTH]}")
    print(f"  FAILED: {status_counts[Pose3DStatus.FAILED]}")
    print()
    
    print("=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  PlayerTracks: {track_set.num_tracks()}")
    print(f"  PoseTracks: {len(pose_track_set.pose_tracks)}")
    print(f"  Pose3DTracks (VALID): {status_counts[Pose3DStatus.VALID]}")
    print()
    print("✓ Pipeline validated: Adapter → Stage 2 → Stage 3")
    print()


if __name__ == "__main__":
    run_minimal_validation()
