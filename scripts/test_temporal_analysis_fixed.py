"""
Temporal Analysis Test Script - FIXED Single Camera Version

This script tests the Pose2DTemporalAnalyzer on a football video to establish
baseline measurements for 2D pose stability metrics.

Key Features:
- SINGLE CAMERA PROCESSING ONLY
- Dynamic project root resolution using sys.path
- Load video using existing PoseEstimator2D
- Confirm frame count (abort if < 100)
- Manual segmentation into 3 phases: Pre-kick (0-80), Swing (80-170), Follow-through (170-end)
- Run Pose2DTemporalAnalyzer on full track and each segment
- Print structured metrics output

Usage:
    python scripts/test_temporal_analysis_fixed.py [video_path]
    
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

from football_app.backend.models.pose_2d_temporal_analyzer import Pose2DTemporalAnalyzer
from football_app.backend.models.pose_estimation_2d import PoseEstimator2D
from football_app.backend.models.validation_adapter import ValidationAdapter
from football_app.backend.models.action_instance import CameraView, MediaType


def print_metrics_section(title: str, metrics: dict, indent: str = "  "):
    """Print a formatted metrics section"""
    print(f"\n{title}:")
    
    if "jitter" in metrics:
        print(f"{indent}📊 High-Frequency Jitter (per joint):")
        for joint_id, value in sorted(metrics["jitter"].items()):
            joint_name = get_joint_name(joint_id)
            print(f"{indent}  Joint {joint_id:2d} ({joint_name:15s}): {value:.4f}")
    
    if "spike_ratio" in metrics:
        print(f"{indent}⚡ Velocity Spike Ratio (per joint):")
        for joint_id, value in sorted(metrics["spike_ratio"].items()):
            joint_name = get_joint_name(joint_id)
            print(f"{indent}  Joint {joint_id:2d} ({joint_name:15s}): {value:.4f}")
    
    if "limb_variance" in metrics:
        print(f"{indent}🦵 Limb Length Variance (normalized):")
        for limb, value in metrics["limb_variance"].items():
            print(f"{indent}  {limb:15s}: {value:.4f}")
    
    if "root_accel_std" in metrics:
        print(f"{indent}📈 Root Acceleration Std: {metrics['root_accel_std']:.4f}")
    
    if "jerk_std" in metrics:
        print(f"{indent}📉 Jerk Std: {metrics['jerk_std']:.4f}")


def get_joint_name(joint_id: int) -> str:
    """Get COCO joint name by ID"""
    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    if 0 <= joint_id < len(joint_names):
        return joint_names[joint_id]
    return f"joint_{joint_id}"


def create_phase_track(full_track, start_frame: int, end_frame: int, phase_name: str):
    """Create a PoseTrack for a specific phase"""
    from football_app.backend.models.pose_estimation_2d import PoseTrack
    
    # Filter poses within the frame range
    phase_poses = []
    for pose in full_track.poses:
        if start_frame <= pose.frame_id <= end_frame:
            phase_poses.append(pose)
    
    # Create new track
    phase_track = PoseTrack(
        track_id=full_track.track_id,
        poses=phase_poses,
        player_track=full_track.player_track
    )
    
    return phase_track


def test_temporal_analysis_single_camera(video_path: str):
    """
    Test temporal analysis on a SINGLE football video camera.
    
    Args:
        video_path: Path to the football video file
    """
    
    print("=" * 80)
    print("TEMPORAL ANALYSIS TEST - SINGLE CAMERA MODE")
    print("=" * 80)
    print()
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"❌ ERROR: Video file does not exist: {video_path}")
        return
    
    print(f"📹 Video: {video_path}")
    print()
    
    # Step 1: Load SINGLE camera video ONLY
    print("-" * 80)
    print("STEP 1: Loading Single Camera Video")
    print("-" * 80)
    
    try:
        camera_id = video_path.stem  # e.g., "CAM0"
        
        # Process ONLY the specific camera video (NO multi-camera processing)
        adapter = ValidationAdapter()
        track_set = adapter._process_single_player_video(video_path, camera_id)
        
        print(f"✅ Loaded {camera_id}: {track_set.num_tracks()} track(s), {track_set.num_frames} frames")
        
        # Check frame count
        if track_set.num_frames < 100:
            print(f"❌ ERROR: Insufficient frames ({track_set.num_frames} < 100)")
            print("   Temporal analysis requires at least 100 frames")
            return
        
        print(f"✅ Frame count check passed: {track_set.num_frames} frames")
        print()
        
    except Exception as e:
        print(f"❌ STEP 1 FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Run 2D Pose Estimation
    print("-" * 80)
    print("STEP 2: 2D Pose Estimation")
    print("-" * 80)
    
    try:
        # Initialize pose estimator (GPU for speed)
        pose_estimator = PoseEstimator2D(device="cuda")
        print("✅ Pose estimator initialized (CUDA)")
        
        # Create camera view
        camera_view = CameraView(
            view_id=camera_id,
            media_path=video_path,
            media_type=MediaType.VIDEO
        )
        
        # Process pose estimation
        start_time = time.time()
        print("Processing 2D poses...", end=" ", flush=True)
        
        pose_track_set = pose_estimator.process_player_track_set(
            track_set, camera_view
        )
        
        pose_time = time.time() - start_time
        print(f"✅ {len(pose_track_set.pose_tracks)} pose track(s) in {pose_time:.2f}s")
        
        # Get the first (and likely only) pose track
        if not pose_track_set.pose_tracks:
            print("❌ ERROR: No pose tracks generated")
            return
        
        main_track = pose_track_set.pose_tracks[0]
        print(f"✅ Main track: {main_track.num_poses()} poses")
        print()
        
    except Exception as e:
        print(f"❌ STEP 2 FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Manual Phase Segmentation
    print("-" * 80)
    print("STEP 3: Manual Phase Segmentation")
    print("-" * 80)
    
    total_frames = main_track.num_poses()
    
    # Define phases (frame-based)
    phases = {
        "Pre-kick": (1, min(80, total_frames)),
        "Swing": (81, min(170, total_frames)),
        "Follow-through": (171, total_frames)
    }
    
    print("Phase definitions:")
    for phase_name, (start, end) in phases.items():
        frame_count = max(0, end - start + 1)
        print(f"  - {phase_name:15s}: frames {start:3d}-{end:3d} ({frame_count:3d} frames)")
    print()
    
    # Create phase tracks
    phase_tracks = {}
    for phase_name, (start_frame, end_frame) in phases.items():
        if end_frame >= start_frame:
            phase_track = create_phase_track(main_track, start_frame, end_frame, phase_name)
            phase_tracks[phase_name] = phase_track
            print(f"✅ {phase_name:15s}: {phase_track.num_poses()} poses extracted")
        else:
            print(f"⚠️  {phase_name:15s}: skipped (invalid range)")
    
    print()
    
    # Step 4: Temporal Analysis
    print("-" * 80)
    print("STEP 4: Temporal Analysis")
    print("-" * 80)
    
    try:
        # Initialize analyzer
        analyzer = Pose2DTemporalAnalyzer(
            moving_avg_window=5,
            spike_threshold_multiplier=3.0
        )
        print("✅ Temporal analyzer initialized")
        print()
        
        # Analyze full track
        print("🔍 Analyzing FULL TRACK...")
        full_results = analyzer.analyze_pose_track(main_track)
        print_metrics_section("FULL TRACK RESULTS", full_results)
        
        # Analyze each phase
        for phase_name, phase_track in phase_tracks.items():
            if phase_track.num_poses() >= 5:  # Minimum for meaningful analysis
                print(f"\n🔍 Analyzing {phase_name.upper()} PHASE...")
                phase_results = analyzer.analyze_pose_track(phase_track)
                print_metrics_section(f"{phase_name.upper()} PHASE RESULTS", phase_results)
            else:
                print(f"\n⚠️  Skipping {phase_name} phase: insufficient poses ({phase_track.num_poses()} < 5)")
        
    except Exception as e:
        print(f"❌ STEP 4 FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ SINGLE CAMERA TEMPORAL ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    
    print("Summary:")
    print(f"  - Video: {video_path.name}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Total poses: {main_track.num_poses()}")
    print(f"  - Phases analyzed: {len([p for p in phase_tracks.values() if p.num_poses() >= 5])}")
    print(f"  - Processing time: {pose_time:.2f}s")
    print()
    
    print("Next Steps:")
    print("  1. Review jitter and spike ratios for unstable joints")
    print("  2. Identify phase-specific stability patterns")
    print("  3. Compare results across different football videos")
    print("  4. Use findings to guide 2D pose improvement strategy")
    print()


if __name__ == "__main__":
    # Default video path
    default_video = "D:/ValidationTesting/Shoot2/CAM6.avi"
    
    # Allow command-line override
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video
    
    print(f"Starting SINGLE CAMERA temporal analysis test with video: {video_path}")
    test_temporal_analysis_single_camera(video_path)