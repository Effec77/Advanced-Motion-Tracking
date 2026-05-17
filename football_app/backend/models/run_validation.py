"""
Validation Script for Pre-Cropped Single-Player Videos

Runs: Adapter → Stage 2 (2D Pose) → Stage 3 (3D Pose)

Input: D:/ValidationTesting/Shoot1/ with cam_X.mp4 files
Output: Summary statistics only (no files saved)

STRICT CONSTRAINTS:
- Does NOT modify Stage 1, 2, or 3 code
- Does NOT save frames, images, or videos
- Does NOT add football logic or metrics
- Validation-only, NOT production code
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


def run_validation(validation_folder: str):
    """
    Run validation pipeline on pre-cropped videos.
    
    Args:
        validation_folder: Path to folder containing cam_X.mp4 files
    """
    print("=" * 70)
    print("VALIDATION PIPELINE - PRE-CROPPED VIDEOS")
    print("=" * 70)
    print()
    
    validation_path = Path(validation_folder)
    
    if not validation_path.exists():
        print(f"❌ ERROR: Validation folder does not exist: {validation_path}")
        return
    
    print(f"📂 Validation folder: {validation_path}")
    print()
    
    # ADAPTER: Load pre-cropped videos (bypasses Stage 1)
    print("-" * 70)
    print("ADAPTER: Loading Pre-Cropped Videos")
    print("-" * 70)
    
    try:
        adapter = ValidationAdapter()
        track_sets = adapter.load_validation_folder(validation_path)
        
        print(f"✓ Loaded {len(track_sets)} camera views")
        for camera_id, track_set in track_sets.items():
            print(f"  - {camera_id}: {track_set.num_tracks()} track(s), {track_set.num_frames} frames")
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
        
        pose_track_sets = {}
        
        for camera_id, player_track_set in track_sets.items():
            # Create a minimal CameraView for Stage 2
            # Stage 2 needs CameraView to read frames from the video
            # Try to find the video file (handle both cam_X.mp4 and CAMX.avi patterns)
            video_path = None
            for pattern in [f"{camera_id}.mp4", f"{camera_id}.avi"]:
                candidate = validation_path / pattern
                if candidate.exists():
                    video_path = candidate
                    break
            
            if video_path is None:
                print(f"⚠️  WARNING: Could not find video file for {camera_id}, skipping...")
                continue
            
            camera_view = CameraView(
                view_id=camera_id,
                media_path=video_path,
                media_type=MediaType.VIDEO
            )
            
            print(f"Processing {camera_id}...")
            pose_track_set = pose_estimator.process_player_track_set(
                player_track_set, camera_view
            )
            pose_track_sets[camera_id] = pose_track_set
        
        print(f"✓ 2D Pose Estimation complete")
        print(f"  - Processed {len(pose_track_sets)} camera views")
        for camera_id, pose_track_set in pose_track_sets.items():
            print(f"  - {camera_id}: {len(pose_track_set.pose_tracks)} pose track(s)")
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
        
        pose_3d_track_sets = {}
        
        for camera_id, pose_track_set in pose_track_sets.items():
            print(f"Processing {camera_id}...")
            pose_3d_track_set = pose_lifter.process_pose_track_set(pose_track_set)
            pose_3d_track_sets[camera_id] = pose_3d_track_set
        
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
    
    print("Summary by Camera:")
    print()
    
    for camera_id in sorted(track_sets.keys()):
        print(f"📹 {camera_id}:")
        
        # PlayerTracks
        player_track_set = track_sets[camera_id]
        print(f"  - PlayerTracks: {player_track_set.num_tracks()}")
        
        # PoseTracks
        pose_track_set = pose_track_sets[camera_id]
        print(f"  - PoseTracks: {len(pose_track_set.pose_tracks)}")
        
        # Pose3DTracks with status breakdown
        pose_3d_track_set = pose_3d_track_sets[camera_id]
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
        print()
    
    print("✓ Pipeline executed successfully")
    print("✓ Data flow validated: Adapter → Stage 2 → Stage 3")
    print()


if __name__ == "__main__":
    # Default validation folder
    validation_folder = "D:/ValidationTesting/Shoot1"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        validation_folder = sys.argv[1]
    
    run_validation(validation_folder)
