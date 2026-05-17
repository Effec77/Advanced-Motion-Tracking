"""
Camera Difficulty Scoring Pipeline

This script ranks all camera videos across Shoot1-3 based on 2D pose temporal
instability metrics to prioritize training data for pose improvement.

CONSTRAINTS:
- Diagnostic only (no pipeline modifications)
- No stabilizer, no smoothing
- Uses existing Pose2DTemporalAnalyzer
- CUDA inference for speed
- Fully standalone

INPUT:
- D:/ValidationTesting/Shoot{1,2,3}/CAM{0-6}.avi
- 21 videos total (3 shoots × 7 cameras)

OUTPUT:
- camera_difficulty_scores.csv (ranked by difficulty)
- Console summary with top/bottom 5 cameras
"""

import sys
from pathlib import Path
import time
import csv
from typing import Dict, List, Tuple
import numpy as np

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


# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory containing all shoots
BASE_DIR = Path("D:/ValidationTesting")

# Shoots and cameras to process
SHOOTS = ["Shoot1", "Shoot2", "Shoot3"]
CAMERAS = ["CAM0", "CAM1", "CAM2", "CAM3", "CAM4", "CAM5", "CAM6"]

# Target joints for instability measurement (lower body only)
TARGET_JOINTS = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles

# Difficulty score weights (must sum to 1.0)
WEIGHTS = {
    "mean_jitter": 0.30,
    "mean_spike": 0.20,
    "mean_limb_var": 0.20,
    "root_accel": 0.15,
    "jerk": 0.15
}

# Output CSV file
OUTPUT_CSV = "camera_difficulty_scores.csv"


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def process_single_video(
    video_path: Path,
    pose_estimator: PoseEstimator2D,
    analyzer: Pose2DTemporalAnalyzer
) -> Dict:
    """
    Process a single video and extract temporal instability metrics.
    
    Args:
        video_path: Path to video file
        pose_estimator: Initialized PoseEstimator2D
        analyzer: Initialized Pose2DTemporalAnalyzer
        
    Returns:
        Dictionary with aggregated metrics or None if processing fails
    """
    try:
        camera_id = video_path.stem
        
        # Step 1: Load video (single camera only)
        adapter = ValidationAdapter()
        track_set = adapter._process_single_player_video(video_path, camera_id)
        
        if track_set.num_frames < 10:
            print(f"  [WARN] Insufficient frames: {track_set.num_frames}")
            return None
        
        # Step 2: Run 2D pose estimation
        camera_view = CameraView(
            view_id=camera_id,
            media_path=video_path,
            media_type=MediaType.VIDEO
        )
        
        pose_track_set = pose_estimator.process_player_track_set(
            track_set, camera_view
        )
        
        if not pose_track_set.pose_tracks:
            print(f"  [WARN] No pose tracks generated")
            return None
        
        main_track = pose_track_set.pose_tracks[0]
        
        if main_track.num_poses() < 10:
            print(f"  [WARN] Insufficient poses: {main_track.num_poses()}")
            return None
        
        # Step 3: Run temporal analysis
        metrics = analyzer.analyze_pose_track(main_track)
        
        # Step 4: Aggregate metrics
        aggregated = aggregate_metrics(metrics)
        
        return aggregated
        
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return None


def aggregate_metrics(metrics: Dict) -> Dict:
    """
    Aggregate temporal metrics for target joints.
    
    Args:
        metrics: Raw metrics from Pose2DTemporalAnalyzer
        
    Returns:
        Dictionary with aggregated metrics
    """
    # Extract jitter for target joints
    jitter_values = [metrics["jitter"][j] for j in TARGET_JOINTS]
    mean_jitter = float(np.mean(jitter_values))
    
    # Extract spike ratio for target joints
    spike_values = [metrics["spike_ratio"][j] for j in TARGET_JOINTS]
    mean_spike = float(np.mean(spike_values))
    
    # Extract limb variance (all limbs)
    limb_var_values = list(metrics["limb_variance"].values())
    mean_limb_var = float(np.mean(limb_var_values))
    
    # Extract root motion metrics
    root_accel = float(metrics["root_accel_std"])
    jerk = float(metrics["jerk_std"])
    
    return {
        "mean_jitter": mean_jitter,
        "mean_spike": mean_spike,
        "mean_limb_var": mean_limb_var,
        "root_accel": root_accel,
        "jerk": jerk
    }


def normalize_metrics(results: List[Dict]) -> List[Dict]:
    """
    Apply min-max normalization to all metrics across the dataset.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        List of result dictionaries with normalized metrics
    """
    # Extract metric values for normalization
    metric_names = ["mean_jitter", "mean_spike", "mean_limb_var", "root_accel", "jerk"]
    
    # Compute min/max for each metric
    min_max = {}
    for metric in metric_names:
        values = [r[metric] for r in results]
        min_val = min(values)
        max_val = max(values)
        min_max[metric] = (min_val, max_val)
    
    # Normalize each result
    normalized_results = []
    for result in results:
        normalized = result.copy()
        
        for metric in metric_names:
            min_val, max_val = min_max[metric]
            value = result[metric]
            
            # Handle edge case: all values are the same
            if max_val - min_val < 1e-10:
                normalized[f"norm_{metric}"] = 0.5  # Neutral value
            else:
                normalized[f"norm_{metric}"] = (value - min_val) / (max_val - min_val)
        
        normalized_results.append(normalized)
    
    return normalized_results


def compute_difficulty_score(result: Dict) -> float:
    """
    Compute camera difficulty score from normalized metrics.
    
    Args:
        result: Result dictionary with normalized metrics
        
    Returns:
        Difficulty score (0-1, higher = more difficult)
    """
    score = (
        WEIGHTS["mean_jitter"] * result["norm_mean_jitter"] +
        WEIGHTS["mean_spike"] * result["norm_mean_spike"] +
        WEIGHTS["mean_limb_var"] * result["norm_mean_limb_var"] +
        WEIGHTS["root_accel"] * result["norm_root_accel"] +
        WEIGHTS["jerk"] * result["norm_jerk"]
    )
    
    return float(score)


def save_results_to_csv(results: List[Dict], output_path: Path):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = [
            "Shoot", "Camera", "VideoPath",
            "MeanJitter", "MeanSpike", "MeanLimbVar", "RootAccel", "Jerk",
            "Score"
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                "Shoot": result["shoot"],
                "Camera": result["camera"],
                "VideoPath": str(result["video_path"]),
                "MeanJitter": f"{result['mean_jitter']:.4f}",
                "MeanSpike": f"{result['mean_spike']:.4f}",
                "MeanLimbVar": f"{result['mean_limb_var']:.4f}",
                "RootAccel": f"{result['root_accel']:.4f}",
                "Jerk": f"{result['jerk']:.4f}",
                "Score": f"{result['score']:.4f}"
            })


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main pipeline: process all videos and generate difficulty ranking.
    """
    # Open log file for output
    log_file = open("camera_ranking_log.txt", "w", buffering=1)
    
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    
    log("=" * 80)
    log("CAMERA DIFFICULTY SCORING PIPELINE")
    log("=" * 80)
    log("")
    
    log("Configuration:")
    log(f"  - Base directory: {BASE_DIR}")
    log(f"  - Shoots: {SHOOTS}")
    log(f"  - Cameras per shoot: {len(CAMERAS)}")
    log(f"  - Expected videos: {len(SHOOTS) * len(CAMERAS)}")
    log(f"  - Target joints: {TARGET_JOINTS}")
    log(f"  - Score weights: {WEIGHTS}")
    log("")
    
    # Verify base directory exists
    if not BASE_DIR.exists():
        log(f"[ERROR] Base directory does not exist: {BASE_DIR}")
        log_file.close()
        return
    
    # Initialize pose estimator and analyzer (reuse across all videos)
    log("Initializing models...")
    pose_estimator = PoseEstimator2D(device="cuda")
    analyzer = Pose2DTemporalAnalyzer()
    log("[OK] Models initialized (CUDA)")
    log("")
    
    # Process all videos
    log("-" * 80)
    log("Processing Videos")
    log("-" * 80)
    
    results = []
    total_start_time = time.time()
    
    for shoot in SHOOTS:
        shoot_dir = BASE_DIR / shoot
        
        if not shoot_dir.exists():
            log(f"[WARN] Shoot directory not found: {shoot_dir}")
            continue
        
        log(f"\n{shoot}:")
        
        for camera in CAMERAS:
            # Try both .avi and .mp4 extensions
            video_path = None
            for ext in ['.avi', '.mp4']:
                candidate = shoot_dir / f"{camera}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            
            if video_path is None:
                log(f"  [WARN] {camera}: Video file not found")
                continue
            
            log(f"  {camera}: Processing...")
            
            video_start_time = time.time()
            metrics = process_single_video(video_path, pose_estimator, analyzer)
            video_time = time.time() - video_start_time
            
            if metrics is None:
                log(f"    FAILED ({video_time:.2f}s)")
                continue
            
            # Store result
            result = {
                "shoot": shoot,
                "camera": camera,
                "video_path": video_path,
                **metrics
            }
            results.append(result)
            
            log(f"    OK ({video_time:.2f}s)")
    
    total_time = time.time() - total_start_time
    
    log("")
    log("-" * 80)
    log(f"[OK] Processed {len(results)} videos in {total_time:.2f}s")
    log(f"     Average: {total_time / len(results):.2f}s per video")
    log("")
    
    # Validation
    if len(results) == 0:
        log("[ERROR] No videos processed successfully")
        log_file.close()
        return
    
    if len(results) < 21:
        log(f"[WARN] Expected 21 videos, got {len(results)}")
    
    # Check for NaN values
    has_nan = False
    for result in results:
        for key in ["mean_jitter", "mean_spike", "mean_limb_var", "root_accel", "jerk"]:
            if np.isnan(result[key]):
                log(f"[WARN] NaN value in {result['shoot']}/{result['camera']} - {key}")
                has_nan = True
    
    if has_nan:
        log("[ERROR] NaN values detected, cannot proceed")
        log_file.close()
        return
    
    # Normalize metrics
    log("Normalizing metrics...")
    normalized_results = normalize_metrics(results)
    log("[OK] Normalization complete")
    log("")
    
    # Compute difficulty scores
    log("Computing difficulty scores...")
    for result in normalized_results:
        result["score"] = compute_difficulty_score(result)
    log("[OK] Scores computed")
    log("")
    
    # Sort by score (descending - hardest first)
    normalized_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Save to CSV
    output_path = Path(OUTPUT_CSV)
    log(f"Saving results to {output_path}...")
    save_results_to_csv(normalized_results, output_path)
    log(f"[OK] Results saved to {output_path}")
    
    # Print summary
    print_summary_to_log(normalized_results, log)
    
    log("=" * 80)
    log("[OK] CAMERA DIFFICULTY RANKING COMPLETE")
    log("=" * 80)
    log("")
    log(f"Results saved to: {output_path.resolve()}")
    log("")
    
    log_file.close()


def print_summary_to_log(results: List[Dict], log_func):
    """
    Print summary statistics and top/bottom cameras to log.
    
    Args:
        results: List of result dictionaries sorted by score
        log_func: Logging function
    """
    scores = [r["score"] for r in results]
    
    log_func("\n" + "=" * 80)
    log_func("CAMERA DIFFICULTY RANKING SUMMARY")
    log_func("=" * 80)
    log_func("")
    
    log_func("Score Statistics:")
    log_func(f"  - Mean:   {np.mean(scores):.4f}")
    log_func(f"  - Std:    {np.std(scores):.4f}")
    log_func(f"  - Min:    {np.min(scores):.4f}")
    log_func(f"  - Max:    {np.max(scores):.4f}")
    log_func(f"  - Median: {np.median(scores):.4f}")
    log_func("")
    
    log_func("TOP 5 HARDEST CAMERAS (Highest Difficulty):")
    log_func("-" * 80)
    log_func(f"{'Rank':<6} {'Shoot':<10} {'Camera':<10} {'Score':<10} {'Path'}")
    log_func("-" * 80)
    for i, result in enumerate(results[:5], 1):
        log_func(f"{i:<6} {result['shoot']:<10} {result['camera']:<10} "
              f"{result['score']:<10.4f} {result['video_path'].name}")
    log_func("")
    
    log_func("BOTTOM 5 EASIEST CAMERAS (Lowest Difficulty):")
    log_func("-" * 80)
    log_func(f"{'Rank':<6} {'Shoot':<10} {'Camera':<10} {'Score':<10} {'Path'}")
    log_func("-" * 80)
    for i, result in enumerate(results[-5:], len(results) - 4):
        log_func(f"{i:<6} {result['shoot']:<10} {result['camera']:<10} "
              f"{result['score']:<10.4f} {result['video_path'].name}")
    log_func("")


if __name__ == "__main__":
    print("Starting camera difficulty ranking pipeline...")
    try:
        main()
    except Exception as e:
        print(f"[FATAL ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
