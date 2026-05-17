"""
Quick determinism test for Stage 4 - Run twice and compare results
"""

from pathlib import Path
import sys
import os

# Setup paths
repo_root = Path(__file__).parent.parent.parent.parent
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))
motionagformer_parent = repo_root / "3dsp_utils"
sys.path.insert(0, str(motionagformer_parent))

from football_app.backend.models.validation_adapter import ValidationAdapter
from football_app.backend.models.pose_estimation_2d import PoseEstimator2D
from football_app.backend.models.pose_lifting_3d import PoseLifter3D
from football_app.backend.models.motion_metrics import MotionMetricsEngine
from football_app.backend.models.action_instance import CameraView, MediaType


def run_pipeline_once(validation_path: Path, camera_id: str):
    """Run pipeline once and return metrics"""
    
    # Adapter
    adapter = ValidationAdapter()
    track_sets = adapter.load_validation_folder(validation_path)
    player_track_set = track_sets[camera_id]
    
    # Stage 2
    video_path = None
    for ext in ['.avi', '.mp4']:
        candidate = validation_path / f"{camera_id}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    
    camera_view = CameraView(
        view_id=camera_id,
        media_path=video_path,
        media_type=MediaType.VIDEO
    )
    
    pose_estimator = PoseEstimator2D(device="cuda")
    pose_track_set = pose_estimator.process_player_track_set(
        player_track_set, camera_view
    )
    
    # Stage 3
    pose_lifter = PoseLifter3D(
        model_path="3dsp_utils/MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr",
        device="cuda"
    )
    pose_3d_track_set = pose_lifter.process_pose_track_set(pose_track_set)
    
    # Stage 4
    metrics_engine = MotionMetricsEngine()
    metric_track_set = metrics_engine.process_pose_3d_track_set(pose_3d_track_set)
    
    # Extract metrics
    mt = metric_track_set.metric_tracks[0]
    sm = mt.summary_metrics
    
    return {
        'V_avg': sm.V_avg,
        'V_peak': sm.V_peak,
        'A_var': sm.A_var,
        'J_avg': sm.J_avg,
        'Stability': sm.Stability,
        'Symmetry': sm.Symmetry
    }


if __name__ == "__main__":
    validation_path = Path("D:/ValidationTesting/Shoot2")
    camera_id = "CAM0"
    
    print("=" * 70)
    print("STAGE 4 DETERMINISM TEST")
    print("=" * 70)
    print(f"Camera: {camera_id}")
    print()
    
    print("Run 1...")
    metrics_1 = run_pipeline_once(validation_path, camera_id)
    
    print("Run 2...")
    metrics_2 = run_pipeline_once(validation_path, camera_id)
    
    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()
    
    all_match = True
    for key in metrics_1.keys():
        val1 = metrics_1[key]
        val2 = metrics_2[key]
        match = abs(val1 - val2) < 1e-9
        status = "✓" if match else "✗"
        print(f"{status} {key:12s}: {val1:.9f} vs {val2:.9f}")
        if not match:
            all_match = False
    
    print()
    if all_match:
        print("✅ DETERMINISM VERIFIED: All metrics identical")
    else:
        print("❌ DETERMINISM FAILED: Metrics differ between runs")
    print()
