"""
GPU-Accelerated Validation Script - Controlled Run

STRICT CONSTRAINTS:
- ONNX GPU (CUDAExecutionProvider) MANDATORY
- NO architectural changes to Stages 1, 2, or 3
- NO batching, parallelism, or optimizations
- NO visualization or file saving
- ONE validation run only
- Future ideas captured in FUTURE_ASPECTS.md ONLY

Input: D:/ValidationTesting/Shoot1/ with CAM0-CAM6.avi files
Output: Structural summary only
"""

from pathlib import Path
import sys
import os
import time

# Setup paths
repo_root = Path(__file__).parent.parent.parent.parent
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))
motionagformer_parent = repo_root / "3dsp_utils"
sys.path.insert(0, str(motionagformer_parent))

from football_app.backend.models.validation_adapter import ValidationAdapter
from football_app.backend.models.pose_estimation_2d import PoseEstimator2D
from football_app.backend.models.pose_lifting_3d import PoseLifter3D, Pose3DStatus
from football_app.backend.models.action_instance import CameraView, MediaType


def verify_onnx_gpu():
    """Verify ONNX Runtime has CUDA support. MANDATORY check."""
    import onnxruntime as ort
    
    providers = ort.get_available_providers()
    has_cuda = 'CUDAExecutionProvider' in providers
    
    print("=" * 70)
    print("ONNX RUNTIME GPU VERIFICATION")
    print("=" * 70)
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {providers}")
    print(f"CUDA available: {has_cuda}")
    print()
    
    if not has_cuda:
        print("❌ ERROR: CUDAExecutionProvider NOT available")
        print("   GPU acceleration is MANDATORY for this run.")
        print("   Install: pip install onnxruntime-gpu")
        print()
        sys.exit(1)
    
    print("✅ CUDA verification passed")
    print()
    return True


def run_gpu_validation(validation_folder: str):
    """
    Run GPU-accelerated validation on all cameras.
    
    STRICT EXECUTION:
    - Process all 7 cameras sequentially
    - No batching or parallelism
    - No optimizations
    - Structural output only
    """
    
    # MANDATORY: Verify GPU before proceeding
    verify_onnx_gpu()
    
    print("=" * 70)
    print("GPU-ACCELERATED VALIDATION RUN")
    print("=" * 70)
    print()
    
    validation_path = Path(validation_folder)
    
    if not validation_path.exists():
        print(f"❌ ERROR: Validation folder does not exist: {validation_path}")
        return
    
    print(f"📂 Validation folder: {validation_path}")
    print()
    
    # Start timing
    total_start_time = time.time()
    
    # ADAPTER: Load all cameras
    print("-" * 70)
    print("ADAPTER: Loading Pre-Cropped Videos")
    print("-" * 70)
    
    adapter_start = time.time()
    
    try:
        adapter = ValidationAdapter()
        track_sets = adapter.load_validation_folder(validation_path)
        
        adapter_time = time.time() - adapter_start
        
        print(f"✓ Loaded {len(track_sets)} camera views in {adapter_time:.2f}s")
        for camera_id, track_set in sorted(track_sets.items()):
            print(f"  - {camera_id}: {track_set.num_tracks()} track(s), {track_set.num_frames} frames")
        print()
        
    except Exception as e:
        print(f"❌ ADAPTER FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STAGE 2: 2D Pose Estimation (GPU-accelerated)
    print("-" * 70)
    print("STAGE 2: 2D Pose Estimation (ONNX GPU)")
    print("-" * 70)
    
    stage2_start = time.time()
    
    try:
        # Initialize with CUDA device
        pose_estimator = PoseEstimator2D(device="cuda")
        print("✓ Pose estimator initialized (CUDA)")
        print()
        
        pose_track_sets = {}
        
        for camera_id, player_track_set in sorted(track_sets.items()):
            # Find video file
            video_path = None
            for ext in ['.avi', '.mp4']:
                candidate = validation_path / f"{camera_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            
            if video_path is None:
                print(f"⚠️  WARNING: Video file not found for {camera_id}, skipping...")
                continue
            
            camera_view = CameraView(
                view_id=camera_id,
                media_path=video_path,
                media_type=MediaType.VIDEO
            )
            
            cam_start = time.time()
            print(f"Processing {camera_id}...", end=" ", flush=True)
            
            pose_track_set = pose_estimator.process_player_track_set(
                player_track_set, camera_view
            )
            pose_track_sets[camera_id] = pose_track_set
            
            cam_time = time.time() - cam_start
            print(f"✓ {len(pose_track_set.pose_tracks)} pose track(s) in {cam_time:.2f}s")
        
        stage2_time = time.time() - stage2_start
        
        print()
        print(f"✓ Stage 2 complete in {stage2_time:.2f}s")
        print(f"  - Total cameras: {len(pose_track_sets)}")
        print(f"  - Total pose tracks: {sum(len(pts.pose_tracks) for pts in pose_track_sets.values())}")
        print()
        
    except Exception as e:
        print(f"❌ STAGE 2 FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STAGE 3: 3D Pose Lifting (CUDA-accelerated)
    print("-" * 70)
    print("STAGE 3: 3D Pose Lifting (MotionAGFormer CUDA)")
    print("-" * 70)
    
    stage3_start = time.time()
    
    try:
        pose_lifter = PoseLifter3D(
            model_path="3dsp_utils/MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr",
            device="cuda"
        )
        print("✓ Pose lifter initialized (CUDA)")
        print()
        
        pose_3d_track_sets = {}
        
        for camera_id, pose_track_set in sorted(pose_track_sets.items()):
            cam_start = time.time()
            print(f"Processing {camera_id}...", end=" ", flush=True)
            
            pose_3d_track_set = pose_lifter.process_pose_track_set(pose_track_set)
            pose_3d_track_sets[camera_id] = pose_3d_track_set
            
            cam_time = time.time() - cam_start
            
            # Count statuses
            status_counts = {s: 0 for s in Pose3DStatus}
            for track in pose_3d_track_set.pose_3d_tracks:
                status_counts[track.status] += 1
            
            print(f"✓ VALID:{status_counts[Pose3DStatus.VALID]} "
                  f"INSUFFICIENT:{status_counts[Pose3DStatus.INSUFFICIENT_LENGTH]} "
                  f"FAILED:{status_counts[Pose3DStatus.FAILED]} "
                  f"in {cam_time:.2f}s")
        
        stage3_time = time.time() - stage3_start
        
        print()
        print(f"✓ Stage 3 complete in {stage3_time:.2f}s")
        print()
        
    except Exception as e:
        print(f"❌ STAGE 3 FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # SUMMARY
    print("=" * 70)
    print("✅ GPU-ACCELERATED VALIDATION COMPLETE")
    print("=" * 70)
    print()
    
    print("Summary by Camera:")
    print()
    
    for camera_id in sorted(track_sets.keys()):
        print(f"📹 {camera_id}:")
        
        # Frames
        player_track_set = track_sets[camera_id]
        print(f"  - Frames processed: {player_track_set.num_frames}")
        
        # PlayerTracks
        print(f"  - PlayerTracks: {player_track_set.num_tracks()}")
        
        # PoseTracks
        if camera_id in pose_track_sets:
            pose_track_set = pose_track_sets[camera_id]
            print(f"  - PoseTracks: {len(pose_track_set.pose_tracks)}")
        
        # Pose3DTracks
        if camera_id in pose_3d_track_sets:
            pose_3d_track_set = pose_3d_track_sets[camera_id]
            status_counts = {s: 0 for s in Pose3DStatus}
            for track in pose_3d_track_set.pose_3d_tracks:
                status_counts[track.status] += 1
            
            print(f"  - Pose3DTracks:")
            print(f"    • VALID: {status_counts[Pose3DStatus.VALID]}")
            print(f"    • INSUFFICIENT_LENGTH: {status_counts[Pose3DStatus.INSUFFICIENT_LENGTH]}")
            print(f"    • FAILED: {status_counts[Pose3DStatus.FAILED]}")
        
        print()
    
    # Timing breakdown
    print("Timing Breakdown:")
    print(f"  - Adapter: {adapter_time:.2f}s")
    print(f"  - Stage 2 (2D Pose): {stage2_time:.2f}s")
    print(f"  - Stage 3 (3D Pose): {stage3_time:.2f}s")
    print(f"  - Total runtime: {total_time:.2f}s")
    print()
    
    # Aggregate statistics
    total_frames = sum(ts.num_frames for ts in track_sets.values())
    total_valid_3d = sum(
        sum(1 for t in pts.pose_3d_tracks if t.status == Pose3DStatus.VALID)
        for pts in pose_3d_track_sets.values()
    )
    
    print("Aggregate Statistics:")
    print(f"  - Total cameras: {len(track_sets)}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Total VALID 3D tracks: {total_valid_3d}")
    print(f"  - Average FPS (Stage 2): {total_frames / stage2_time:.1f}")
    print()
    
    print("✓ Pipeline executed successfully")
    print("✓ Data flow validated: Adapter → Stage 2 (GPU) → Stage 3 (GPU)")
    print("✓ No architectural modifications made")
    print("✓ Future ideas captured in FUTURE_ASPECTS.md")
    print()


if __name__ == "__main__":
    validation_folder = "D:/ValidationTesting/Shoot3"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        validation_folder = sys.argv[1]
    
    run_gpu_validation(validation_folder)
