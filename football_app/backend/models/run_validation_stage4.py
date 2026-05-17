"""
Stage 4 Validation Script - Motion Metrics & Stability Engine

Runs: Adapter → Stage 2 (2D Pose) → Stage 3 (3D Pose) → Stage 4 (Motion Metrics)

STRICT CONSTRAINTS:
- Follows exact workflow from GPU validation
- Adds Stage 4 processing only
- No modifications to Stages 1-3
- Structural output only
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
from football_app.backend.models.motion_metrics import MotionMetricsEngine, MetricStatus
from football_app.backend.models.action_instance import CameraView, MediaType


def verify_onnx_gpu():
    """Verify ONNX Runtime has CUDA support."""
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
        sys.exit(1)
    
    print("✅ CUDA verification passed")
    print()
    return True


def run_stage4_validation(validation_folder: str):
    """
    Run full pipeline validation including Stage 4.
    """
    
    # Verify GPU
    verify_onnx_gpu()
    
    print("=" * 70)
    print("STAGE 4 VALIDATION - MOTION METRICS & STABILITY")
    print("=" * 70)
    print()
    
    validation_path = Path(validation_folder)
    
    if not validation_path.exists():
        print(f"❌ ERROR: Validation folder does not exist: {validation_path}")
        return
    
    print(f"📂 Validation folder: {validation_path}")
    print()
    
    total_start_time = time.time()
    
    # ADAPTER
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
        return
    
    # STAGE 2
    print("-" * 70)
    print("STAGE 2: 2D Pose Estimation (ONNX GPU)")
    print("-" * 70)
    
    stage2_start = time.time()
    
    try:
        pose_estimator = PoseEstimator2D(device="cuda")
        print("✓ Pose estimator initialized (CUDA)")
        print()
        
        pose_track_sets = {}
        
        for camera_id, player_track_set in sorted(track_sets.items()):
            video_path = None
            for ext in ['.avi', '.mp4']:
                candidate = validation_path / f"{camera_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            
            if video_path is None:
                continue
            
            camera_view = CameraView(
                view_id=camera_id,
                media_path=video_path,
                media_type=MediaType.VIDEO
            )
            
            print(f"Processing {camera_id}...", end=" ", flush=True)
            
            pose_track_set = pose_estimator.process_player_track_set(
                player_track_set, camera_view
            )
            pose_track_sets[camera_id] = pose_track_set
            
            print(f"✓ {len(pose_track_set.pose_tracks)} pose track(s)")
        
        stage2_time = time.time() - stage2_start
        print()
        print(f"✓ Stage 2 complete in {stage2_time:.2f}s")
        print()
        
    except Exception as e:
        print(f"❌ STAGE 2 FAILED: {type(e).__name__}: {e}")
        return
    
    # STAGE 3
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
            print(f"Processing {camera_id}...", end=" ", flush=True)
            
            pose_3d_track_set = pose_lifter.process_pose_track_set(pose_track_set)
            pose_3d_track_sets[camera_id] = pose_3d_track_set
            
            status_counts = {s: 0 for s in Pose3DStatus}
            for track in pose_3d_track_set.pose_3d_tracks:
                status_counts[track.status] += 1
            
            print(f"✓ VALID:{status_counts[Pose3DStatus.VALID]}")
        
        stage3_time = time.time() - stage3_start
        print()
        print(f"✓ Stage 3 complete in {stage3_time:.2f}s")
        print()
        
    except Exception as e:
        print(f"❌ STAGE 3 FAILED: {type(e).__name__}: {e}")
        return
    
    # STAGE 4: Motion Metrics & Stability
    print("-" * 70)
    print("STAGE 4: Motion Metrics & Stability Engine")
    print("-" * 70)
    
    stage4_start = time.time()
    
    try:
        metrics_engine = MotionMetricsEngine()
        print("✓ Motion metrics engine initialized")
        print()
        
        metric_track_sets = {}
        
        for camera_id, pose_3d_track_set in sorted(pose_3d_track_sets.items()):
            print(f"Processing {camera_id}...", end=" ", flush=True)
            
            metric_track_set = metrics_engine.process_pose_3d_track_set(pose_3d_track_set)
            metric_track_sets[camera_id] = metric_track_set
            
            status_counts = {s: 0 for s in MetricStatus}
            for track in metric_track_set.metric_tracks:
                status_counts[track.status] += 1
            
            print(f"✓ COMPLETE:{status_counts[MetricStatus.COMPLETE]} "
                  f"PARTIAL:{status_counts[MetricStatus.PARTIAL]} "
                  f"FAILED:{status_counts[MetricStatus.FAILED]}")
        
        stage4_time = time.time() - stage4_start
        print()
        print(f"✓ Stage 4 complete in {stage4_time:.2f}s")
        print()
        
    except Exception as e:
        print(f"❌ STAGE 4 FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    total_time = time.time() - total_start_time
    
    # SUMMARY
    print("=" * 70)
    print("✅ STAGE 4 VALIDATION COMPLETE")
    print("=" * 70)
    print()
    
    print("Summary by Camera:")
    print()
    
    for camera_id in sorted(track_sets.keys()):
        print(f"📹 {camera_id}:")
        
        # Frames
        player_track_set = track_sets[camera_id]
        print(f"  - Frames processed: {player_track_set.num_frames}")
        
        # Metrics
        if camera_id in metric_track_sets:
            metric_track_set = metric_track_sets[camera_id]
            print(f"  - MetricTracks: {metric_track_set.num_tracks()}")
            
            # Show summary metrics for first track
            if len(metric_track_set.metric_tracks) > 0:
                mt = metric_track_set.metric_tracks[0]
                if mt.summary_metrics:
                    sm = mt.summary_metrics
                    print(f"  - Metrics:")
                    print(f"    • V_avg: {sm.V_avg:.3f}")
                    print(f"    • V_peak: {sm.V_peak:.3f}")
                    print(f"    • Stability: {sm.Stability:.3f}")
                    print(f"    • Symmetry: {sm.Symmetry:.3f}")
        
        print()
    
    # Timing breakdown
    print("Timing Breakdown:")
    print(f"  - Adapter: {adapter_time:.2f}s")
    print(f"  - Stage 2 (2D Pose): {stage2_time:.2f}s")
    print(f"  - Stage 3 (3D Pose): {stage3_time:.2f}s")
    print(f"  - Stage 4 (Metrics): {stage4_time:.2f}s")
    print(f"  - Total runtime: {total_time:.2f}s")
    print()
    
    print("✓ Pipeline executed successfully")
    print("✓ Data flow validated: Adapter → Stage 2 → Stage 3 → Stage 4")
    print()


if __name__ == "__main__":
    validation_folder = "D:/ValidationTesting/Shoot1"
    
    if len(sys.argv) > 1:
        validation_folder = sys.argv[1]
    
    run_stage4_validation(validation_folder)
