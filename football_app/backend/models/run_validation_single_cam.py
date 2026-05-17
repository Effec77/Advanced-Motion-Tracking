"""
Validation Script for Single Camera (Quick Test)

Runs: Adapter → Stage 2 (2D Pose) → Stage 3 (3D Pose)
Processes only CAM0 for quick validation.
"""

from pathlib import Path
import sys
import os

# Change to repository root directory
repo_root = Path(__file__).parent.parent.parent.parent
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))

from football_app.backend.models.validation_adapter import ValidationAdapter
from football_app.backend.models.pose_estimation_2d import PoseEstimator2D
from football_app.backend.models.pose_lifting_3d import PoseLifter3D, Pose3DStatus
from football_app.backend.models.action_instance import CameraView, MediaType


def run_single_camera_validation(validation_folder: str, camera_name: str = "CAM0"):
    """
    Run validation pipeline on a single camera.
    
    Args:
        validation_folder: Path to folder containing camera video files
        camera_name: Name of camera to process (default: CAM0)
    """
    print("=" * 70)
    print(f"VALIDATION PIPELINE - SINGLE CAMERA ({camera_name})")
    print("=" * 70)
    print()
    
    validation_path = Path(validation_folder)
    
    if not validation_path.exists():
        print(f"❌ ERROR: Validation folder does not exist: {validation_path}")
        return
    
    print(f"📂 Validation folder: {validation_path}")
    print(f"📹 Processing camera: {camera_name}")
    print()
    
    # Find the video file
    video_file = None
    for ext in ['.avi', '.mp4']:
        candidate = validation_path / f"{camera_name}{ext}"
        if candidate.exists():
            video_file = candidate
            break
    
    if video_file is None:
        print(f"❌ ERROR: Video file not found for {camera_name}")
        return
    
    # ADAPTER: Load single video
    print("-" * 70)
    print("ADAPTER: Loading Pre-Cropped Video")
    print("-" * 70)
    
    try:
        adapter = ValidationAdapter()
        track_set = adapter._process_single_player_video(video_file, camera_name)
        
        print(f"✓ Loaded {camera_name}")
        print(f"  - Tracks: {track_set.num_tracks()}")
        print(f"  - Frames: {track_set.num_frames}")
        print()
        
    except Exception as e:
        print(f"❌ ADAPTER FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STAGE 2: 2D Pose Estimation
    print("-" * 70)
    print("STAGE 2: 2D Pose Estimation")
    print("-" * 70)
    
    try:
        pose_estimator = PoseEstimator2D(device="cuda")
        print("✓ Pose estimator initialized")
        
        camera_view = CameraView(
            view_id=camera_name,
            media_path=video_file,
            media_type=MediaType.VIDEO
        )
        
        print(f"Processing {camera_name}...")
        pose_track_set = pose_estimator.process_player_track_set(track_set, camera_view)
        
        print(f"✓ 2D Pose Estimation complete")
        print(f"  - Pose tracks: {len(pose_track_set.pose_tracks)}")
        if len(pose_track_set.pose_tracks) > 0:
            print(f"  - Poses in track: {pose_track_set.pose_tracks[0].num_poses()}")
        print()
        
    except Exception as e:
        print(f"❌ STAGE 2 FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STAGE 3: 3D Pose Lifting
    print("-" * 70)
    print("STAGE 3: 3D Pose Lifting")
    print("-" * 70)
    
    try:
        pose_lifter = PoseLifter3D(
            model_path="3dsp_utils/MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr",
            device="cuda"
        )
        print("✓ Pose lifter initialized")
        
        print(f"Processing {camera_name}...")
        pose_3d_track_set = pose_lifter.process_pose_track_set(pose_track_set)
        
        print(f"✓ 3D Pose Lifting complete")
        print()
        
    except Exception as e:
        print(f"❌ STAGE 3 FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # SUMMARY
    print("=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)
    print()
    
    print(f"📹 {camera_name} Results:")
    print(f"  - PlayerTracks: {track_set.num_tracks()}")
    print(f"  - PoseTracks: {len(pose_track_set.pose_tracks)}")
    
    # Pose3DTracks with status breakdown
    status_counts = {
        Pose3DStatus.VALID: 0,
        Pose3DStatus.INSUFFICIENT_LENGTH: 0,
        Pose3DStatus.FAILED: 0
    }
    
    for pose_3d_track in pose_3d_track_set.pose_3d_tracks:
        status_counts[pose_3d_track.status] += 1
    
    print(f"  - Pose3DTracks:")
    print(f"    • VALID: {status_counts[Pose3DStatus.VALID]}")
    print(f"    • INSUFFICIENT_LENGTH: {status_counts[Pose3DStatus.INSUFFICIENT_LENGTH]}")
    print(f"    • FAILED: {status_counts[Pose3DStatus.FAILED]}")
    
    if status_counts[Pose3DStatus.VALID] > 0:
        valid_track = [t for t in pose_3d_track_set.pose_3d_tracks if t.status == Pose3DStatus.VALID][0]
        print(f"    • Valid track has {valid_track.num_poses()} 3D poses")
    
    print()
    print("✓ Pipeline executed successfully")
    print("✓ Data flow validated: Adapter → Stage 2 → Stage 3")
    print()


if __name__ == "__main__":
    validation_folder = "D:/ValidationTesting/Shoot1"
    camera_name = "CAM0"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        validation_folder = sys.argv[1]
    if len(sys.argv) > 2:
        camera_name = sys.argv[2]
    
    run_single_camera_validation(validation_folder, camera_name)
