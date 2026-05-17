"""
Pose2D Stabilizer Test Script

This script tests the Pose2DStabilizer on real football video data to measure
the effectiveness of temporal stabilization.

Usage:
    python scripts/test_pose_stabilizer.py [video_path]
    
Default video: D:/ValidationTesting/Shoot3/CAM0.avi
"""

import sys
from pathlib import Path
import time

# Dynamic project root resolution
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root))

# Add 3dsp_utils for rtmlib
motionagformer_parent = project_root / "3dsp_utils"
sys.path.insert(0, str(motionagformer_parent))

from football_app.backend.models.pose_2d_stabilizer import Pose2DStabilizer
from football_app.backend.models.pose_2d_temporal_analyzer import Pose2DTemporalAnalyzer
from football_app.backend.models.pose_estimation_2d import PoseEstimator2D
from football_app.backend.models.validation_adapter import ValidationAdapter
from football_app.backend.models.action_instance import CameraView, MediaType


def print_metrics_comparison(title: str, original_metrics: dict, stabilized_metrics: dict):
    """Print side-by-side comparison of metrics"""
    print(f"\n{title}:")
    print("=" * 80)
    
    if "jitter" in original_metrics:
        print("\n📊 High-Frequency Jitter Comparison (Target Joints Only):")
        print("Joint | Original  | Stabilized | Improvement")
        print("-" * 50)
        
        target_joints = [11, 12, 13, 14, 15, 16]  # Hips, knees, ankles
        joint_names = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        
        for i, joint_id in enumerate(target_joints):
            orig_val = original_metrics["jitter"][joint_id]
            stab_val = stabilized_metrics["jitter"][joint_id]
            improvement = ((orig_val - stab_val) / orig_val * 100) if orig_val > 0 else 0
            
            print(f"{joint_id:2d}    | {orig_val:8.2f}  | {stab_val:9.2f}  | {improvement:+6.1f}%")
    
    if "spike_ratio" in original_metrics:
        print("\n⚡ Velocity Spike Ratio Comparison (Target Joints Only):")
        print("Joint | Original  | Stabilized | Improvement")
        print("-" * 50)
        
        for i, joint_id in enumerate(target_joints):
            orig_val = original_metrics["spike_ratio"][joint_id]
            stab_val = stabilized_metrics["spike_ratio"][joint_id]
            improvement = ((orig_val - stab_val) / orig_val * 100) if orig_val > 0 else 0
            
            print(f"{joint_id:2d}    | {orig_val:8.4f}  | {stab_val:9.4f}  | {improvement:+6.1f}%")
    
    if "limb_variance" in original_metrics:
        print("\n🦵 Limb Length Variance Comparison:")
        print("Limb           | Original  | Stabilized | Improvement")
        print("-" * 55)
        
        for limb in ["left_thigh", "right_thigh", "left_shank", "right_shank"]:
            orig_val = original_metrics["limb_variance"][limb]
            stab_val = stabilized_metrics["limb_variance"][limb]
            improvement = ((orig_val - stab_val) / orig_val * 100) if orig_val > 0 else 0
            
            print(f"{limb:14s} | {orig_val:8.4f}  | {stab_val:9.4f}  | {improvement:+6.1f}%")
    
    if "root_accel_std" in original_metrics:
        orig_accel = original_metrics["root_accel_std"]
        stab_accel = stabilized_metrics["root_accel_std"]
        accel_improvement = ((orig_accel - stab_accel) / orig_accel * 100) if orig_accel > 0 else 0
        
        print(f"\n📈 Root Acceleration Std:")
        print(f"  Original:   {orig_accel:8.2f}")
        print(f"  Stabilized: {stab_accel:8.2f}")
        print(f"  Improvement: {accel_improvement:+6.1f}%")
    
    if "jerk_std" in original_metrics:
        orig_jerk = original_metrics["jerk_std"]
        stab_jerk = stabilized_metrics["jerk_std"]
        jerk_improvement = ((orig_jerk - stab_jerk) / orig_jerk * 100) if orig_jerk > 0 else 0
        
        print(f"\n📉 Jerk Std:")
        print(f"  Original:   {orig_jerk:8.2f}")
        print(f"  Stabilized: {stab_jerk:8.2f}")
        print(f"  Improvement: {jerk_improvement:+6.1f}%")


def test_pose_stabilizer(video_path: str):
    """
    Test pose stabilizer on a football video and compare metrics.
    
    Args:
        video_path: Path to the football video file
    """
    
    print("=" * 80)
    print("POSE2D STABILIZER TEST - EFFECTIVENESS MEASUREMENT")
    print("=" * 80)
    print()
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"❌ ERROR: Video file does not exist: {video_path}")
        return
    
    print(f"📹 Video: {video_path}")
    print()
    
    # Step 1: Load video and run 2D pose estimation
    print("-" * 80)
    print("STEP 1: Loading Video & 2D Pose Estimation")
    print("-" * 80)
    
    try:
        # Load validation data for SINGLE camera only
        validation_folder = video_path.parent
        camera_id = video_path.stem
        
        # Check if the specific video file exists
        if not video_path.exists():
            print(f"❌ ERROR: Video file does not exist: {video_path}")
            return
        
        # Process ONLY the specific camera video
        adapter = ValidationAdapter()
        player_track_set = adapter._process_single_player_video(video_path, camera_id)
        
        print(f"✓ Loaded {camera_id}: {player_track_set.num_frames} frames")
        
        # Run 2D pose estimation
        pose_estimator = PoseEstimator2D(device="cuda")
        camera_view = CameraView(
            view_id=camera_id,
            media_path=video_path,
            media_type=MediaType.VIDEO
        )
        
        start_time = time.time()
        pose_track_set = pose_estimator.process_player_track_set(
            player_track_set, camera_view
        )
        pose_time = time.time() - start_time
        
        if not pose_track_set.pose_tracks:
            print("❌ ERROR: No pose tracks generated")
            return
        
        original_track = pose_track_set.pose_tracks[0]
        print(f"✓ Generated pose track: {original_track.num_poses()} poses in {pose_time:.2f}s")
        print()
        
    except Exception as e:
        print(f"❌ STEP 1 FAILED: {type(e).__name__}: {e}")
        return
    
    # Step 2: Apply stabilization
    print("-" * 80)
    print("STEP 2: Applying Pose Stabilization")
    print("-" * 80)
    
    try:
        stabilizer = Pose2DStabilizer(
            low_velocity_alpha=0.85,
            medium_velocity_alpha=0.60,
            high_velocity_alpha=0.25,
            max_limb_deviation=0.15
        )
        print("✓ Stabilizer initialized")
        
        start_time = time.time()
        stabilized_track = stabilizer.stabilize_pose_track(original_track)
        stabilization_time = time.time() - start_time
        
        print(f"✓ Stabilization complete: {stabilized_track.num_poses()} poses in {stabilization_time:.3f}s")
        print()
        
    except Exception as e:
        print(f"❌ STEP 2 FAILED: {type(e).__name__}: {e}")
        return
    
    # Step 3: Temporal analysis comparison
    print("-" * 80)
    print("STEP 3: Temporal Analysis Comparison")
    print("-" * 80)
    
    try:
        analyzer = Pose2DTemporalAnalyzer()
        print("✓ Temporal analyzer initialized")
        
        # Analyze original track
        print("Analyzing original track...")
        original_metrics = analyzer.analyze_pose_track(original_track)
        
        # Analyze stabilized track
        print("Analyzing stabilized track...")
        stabilized_metrics = analyzer.analyze_pose_track(stabilized_track)
        
        print("✓ Analysis complete")
        print()
        
    except Exception as e:
        print(f"❌ STEP 3 FAILED: {type(e).__name__}: {e}")
        return
    
    # Step 4: Results comparison
    print_metrics_comparison("FULL TRACK COMPARISON", original_metrics, stabilized_metrics)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ POSE2D STABILIZER TEST COMPLETE")
    print("=" * 80)
    print()
    
    # Calculate overall improvements
    target_joints = [11, 12, 13, 14, 15, 16]
    
    # Average jitter improvement
    orig_jitter_avg = sum(original_metrics["jitter"][j] for j in target_joints) / len(target_joints)
    stab_jitter_avg = sum(stabilized_metrics["jitter"][j] for j in target_joints) / len(target_joints)
    jitter_improvement = ((orig_jitter_avg - stab_jitter_avg) / orig_jitter_avg * 100) if orig_jitter_avg > 0 else 0
    
    # Average spike ratio improvement
    orig_spike_avg = sum(original_metrics["spike_ratio"][j] for j in target_joints) / len(target_joints)
    stab_spike_avg = sum(stabilized_metrics["spike_ratio"][j] for j in target_joints) / len(target_joints)
    spike_improvement = ((orig_spike_avg - stab_spike_avg) / orig_spike_avg * 100) if orig_spike_avg > 0 else 0
    
    # Root motion improvements
    accel_improvement = ((original_metrics["root_accel_std"] - stabilized_metrics["root_accel_std"]) / 
                        original_metrics["root_accel_std"] * 100) if original_metrics["root_accel_std"] > 0 else 0
    jerk_improvement = ((original_metrics["jerk_std"] - stabilized_metrics["jerk_std"]) / 
                       original_metrics["jerk_std"] * 100) if original_metrics["jerk_std"] > 0 else 0
    
    print("📊 OVERALL EFFECTIVENESS:")
    print(f"  - Average jitter reduction:     {jitter_improvement:+6.1f}%")
    print(f"  - Average spike ratio reduction: {spike_improvement:+6.1f}%")
    print(f"  - Root acceleration reduction:   {accel_improvement:+6.1f}%")
    print(f"  - Jerk reduction:               {jerk_improvement:+6.1f}%")
    print()
    
    print("🎯 NEXT STEPS:")
    if jitter_improvement > 10:
        print("  ✅ Stabilizer is effective - consider integration")
    else:
        print("  ⚠️  Stabilizer needs tuning - adjust parameters")
    
    print("  1. Fine-tune alpha parameters based on results")
    print("  2. Test on additional football videos")
    print("  3. Integrate into production pipeline if effective")
    print("  4. Compare Stage 3 (3D lifting) results with/without stabilization")
    print()


if __name__ == "__main__":
    # Default video path
    default_video = "D:/ValidationTesting/Shoot3/CAM0.avi"
    
    # Allow command-line override
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video
    
    test_pose_stabilizer(video_path)