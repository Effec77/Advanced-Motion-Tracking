"""
Stage 2: 2D Pose Temporal Analyzer (Diagnostic Module)

This module provides diagnostic-only temporal stability analysis for 2D pose sequences.
It does NOT modify any data structures or alter pipeline behavior.

Key Principles:
- Diagnostic-only (no modifications to poses)
- Computes temporal stability metrics
- Uses numpy only (no external dependencies)
- Handles edge cases safely
- No smoothing or filtering applied
"""

from typing import Dict, Optional
import numpy as np
from football_app.backend.models.pose_estimation_2d import PoseTrack


class Pose2DTemporalAnalyzer:
    """
    Diagnostic analyzer for temporal stability of 2D pose sequences.
    
    This class computes stability metrics from PoseTrack objects without
    modifying the original data or affecting pipeline behavior.
    """
    
    # COCO joint indices
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    def __init__(self, moving_avg_window: int = 5, spike_threshold_multiplier: float = 3.0):
        """
        Initialize the temporal analyzer.
        
        Args:
            moving_avg_window: Window size for moving average baseline (default: 5)
            spike_threshold_multiplier: Multiplier for velocity spike detection (default: 3.0)
        """
        self.moving_avg_window = moving_avg_window
        self.spike_threshold_multiplier = spike_threshold_multiplier
    
    def analyze_pose_track(self, pose_track: PoseTrack) -> Dict:
        """
        Analyze temporal stability of a PoseTrack.
        
        Args:
            pose_track: PoseTrack object to analyze
            
        Returns:
            Dictionary containing temporal stability metrics:
            {
                "jitter": {joint_id: value},
                "spike_ratio": {joint_id: value},
                "limb_variance": {
                    "left_thigh": value,
                    "right_thigh": value,
                    "left_shank": value,
                    "right_shank": value
                },
                "root_accel_std": value,
                "jerk_std": value
            }
        """
        # Extract array: (T, 17, 3) where 3 = [x, y, confidence]
        poses_array = pose_track.to_array()
        
        T = poses_array.shape[0]
        
        # Handle edge case: insufficient frames
        if T < 5:
            return self._empty_results()
        
        # Extract positions and confidences
        positions = poses_array[:, :, :2]  # (T, 17, 2) - x, y coordinates
        confidences = poses_array[:, :, 2]  # (T, 17) - confidence scores
        
        # Compute metrics
        jitter = self._compute_jitter(positions)
        spike_ratio = self._compute_spike_ratio(positions)
        limb_variance = self._compute_limb_variance(positions, confidences)
        root_accel_std = self._compute_root_accel_std(positions)
        jerk_std = self._compute_jerk_std(positions)
        
        return {
            "jitter": jitter,
            "spike_ratio": spike_ratio,
            "limb_variance": limb_variance,
            "root_accel_std": root_accel_std,
            "jerk_std": jerk_std
        }
    
    def _compute_jitter(self, positions: np.ndarray) -> Dict[int, float]:
        """
        Compute high-frequency jitter for each joint.
        
        Formula:
        - Velocity: v_t = ||p_t - p_{t-1}||
        - Moving average baseline: v_bar_t = moving_average(v_t, window=5)
        - High-frequency component: delta_t = v_t - v_bar_t
        - Jitter score: J_j = std(delta_t)
        
        Args:
            positions: Array of shape (T, 17, 2)
            
        Returns:
            Dictionary mapping joint_id to jitter score
        """
        T = positions.shape[0]
        num_joints = positions.shape[1]
        
        jitter_dict = {}
        
        for j in range(num_joints):
            # Extract joint positions: (T, 2)
            joint_pos = positions[:, j, :]
            
            # Compute velocity: ||p_t - p_{t-1}||
            # diff[i] = p[i] - p[i-1] for i >= 1
            diff = np.diff(joint_pos, axis=0)  # (T-1, 2)
            velocity = np.linalg.norm(diff, axis=1)  # (T-1,)
            
            if len(velocity) < self.moving_avg_window:
                jitter_dict[j] = 0.0
                continue
            
            # Compute moving average baseline
            v_bar = self._moving_average(velocity, window=self.moving_avg_window)
            
            # Compute high-frequency component
            delta = velocity - v_bar
            
            # Jitter score: standard deviation of high-frequency component
            jitter_dict[j] = float(np.std(delta))
        
        return jitter_dict
    
    def _compute_spike_ratio(self, positions: np.ndarray) -> Dict[int, float]:
        """
        Compute velocity spike ratio for each joint.
        
        Formula:
        - Velocity: v_t = ||p_t - p_{t-1}||
        - Median velocity: median_v = median(v_t)
        - Spike threshold: threshold = 3 * median_v
        - Spike ratio: count(v_t > threshold) / T
        
        Args:
            positions: Array of shape (T, 17, 2)
            
        Returns:
            Dictionary mapping joint_id to spike ratio
        """
        T = positions.shape[0]
        num_joints = positions.shape[1]
        
        spike_ratio_dict = {}
        
        for j in range(num_joints):
            # Extract joint positions: (T, 2)
            joint_pos = positions[:, j, :]
            
            # Compute velocity: ||p_t - p_{t-1}||
            diff = np.diff(joint_pos, axis=0)  # (T-1, 2)
            velocity = np.linalg.norm(diff, axis=1)  # (T-1,)
            
            if len(velocity) == 0:
                spike_ratio_dict[j] = 0.0
                continue
            
            # Compute median velocity
            median_v = np.median(velocity)
            
            if median_v == 0:
                spike_ratio_dict[j] = 0.0
                continue
            
            # Spike threshold
            threshold = self.spike_threshold_multiplier * median_v
            
            # Count spikes
            spike_count = np.sum(velocity > threshold)
            
            # Spike ratio: count / len(velocity) (mathematically correct)
            spike_ratio_dict[j] = float(spike_count / len(velocity))
        
        return spike_ratio_dict
    
    def _compute_limb_variance(self, positions: np.ndarray, confidences: np.ndarray) -> Dict[str, float]:
        """
        Compute normalized limb length variance.
        
        Limbs:
        - Left thigh: left_hip → left_knee
        - Right thigh: right_hip → right_knee
        - Left shank: left_knee → left_ankle
        - Right shank: right_knee → right_ankle
        
        Normalization: Divide by torso length (shoulder_mid - hip_mid)
        
        Formula:
        - Limb length: L_t = ||joint1_t - joint2_t||
        - Normalized variance: LimbVar = std(L_t) / mean(L_t)
        
        Args:
            positions: Array of shape (T, 17, 2)
            confidences: Array of shape (T, 17)
            
        Returns:
            Dictionary with limb variance values
        """
        T = positions.shape[0]
        
        # Compute torso length for normalization (per frame)
        shoulder_mid = (positions[:, self.LEFT_SHOULDER, :] + positions[:, self.RIGHT_SHOULDER, :]) / 2.0
        hip_mid = (positions[:, self.LEFT_HIP, :] + positions[:, self.RIGHT_HIP, :]) / 2.0
        torso_length = np.linalg.norm(shoulder_mid - hip_mid, axis=1)  # (T,)
        
        # Filter frames where torso is invalid (confidence check)
        shoulder_conf = (confidences[:, self.LEFT_SHOULDER] + confidences[:, self.RIGHT_SHOULDER]) / 2.0
        hip_conf = (confidences[:, self.LEFT_HIP] + confidences[:, self.RIGHT_HIP]) / 2.0
        valid_torso_mask = (shoulder_conf >= 0.5) & (hip_conf >= 0.5) & (torso_length > 1e-6)
        
        if np.sum(valid_torso_mask) == 0:
            return {
                "left_thigh": 0.0,
                "right_thigh": 0.0,
                "left_shank": 0.0,
                "right_shank": 0.0
            }
        
        # Normalize torso length (use mean for stability)
        mean_torso = np.mean(torso_length[valid_torso_mask])
        
        if mean_torso < 1e-6:
            return {
                "left_thigh": 0.0,
                "right_thigh": 0.0,
                "left_shank": 0.0,
                "right_shank": 0.0
            }
        
        limb_variance = {}
        
        # Left thigh: left_hip → left_knee
        left_thigh_var = self._compute_limb_variance_single(
            positions[:, self.LEFT_HIP, :],
            positions[:, self.LEFT_KNEE, :],
            confidences[:, self.LEFT_HIP],
            confidences[:, self.LEFT_KNEE],
            mean_torso
        )
        limb_variance["left_thigh"] = left_thigh_var
        
        # Right thigh: right_hip → right_knee
        right_thigh_var = self._compute_limb_variance_single(
            positions[:, self.RIGHT_HIP, :],
            positions[:, self.RIGHT_KNEE, :],
            confidences[:, self.RIGHT_HIP],
            confidences[:, self.RIGHT_KNEE],
            mean_torso
        )
        limb_variance["right_thigh"] = right_thigh_var
        
        # Left shank: left_knee → left_ankle
        left_shank_var = self._compute_limb_variance_single(
            positions[:, self.LEFT_KNEE, :],
            positions[:, self.LEFT_ANKLE, :],
            confidences[:, self.LEFT_KNEE],
            confidences[:, self.LEFT_ANKLE],
            mean_torso
        )
        limb_variance["left_shank"] = left_shank_var
        
        # Right shank: right_knee → right_ankle
        right_shank_var = self._compute_limb_variance_single(
            positions[:, self.RIGHT_KNEE, :],
            positions[:, self.RIGHT_ANKLE, :],
            confidences[:, self.RIGHT_KNEE],
            confidences[:, self.RIGHT_ANKLE],
            mean_torso
        )
        limb_variance["right_shank"] = right_shank_var
        
        return limb_variance
    
    def _compute_limb_variance_single(
        self,
        joint1_pos: np.ndarray,
        joint2_pos: np.ndarray,
        joint1_conf: np.ndarray,
        joint2_conf: np.ndarray,
        normalization_factor: float
    ) -> float:
        """
        Compute normalized variance for a single limb.
        
        Args:
            joint1_pos: Array of shape (T, 2)
            joint2_pos: Array of shape (T, 2)
            joint1_conf: Array of shape (T,)
            joint2_conf: Array of shape (T,)
            normalization_factor: Torso length for normalization
            
        Returns:
            Normalized limb variance
        """
        # Filter frames where either joint confidence < 0.5
        valid_mask = (joint1_conf >= 0.5) & (joint2_conf >= 0.5)
        
        if np.sum(valid_mask) < 2:
            return 0.0
        
        # Compute limb length per frame
        limb_length = np.linalg.norm(joint1_pos - joint2_pos, axis=1)  # (T,)
        
        # Normalize by torso length
        normalized_length = limb_length / normalization_factor
        
        # Extract valid frames
        valid_lengths = normalized_length[valid_mask]
        
        if len(valid_lengths) < 2:
            return 0.0
        
        # Compute variance: std(L_t) / mean(L_t)
        mean_length = np.mean(valid_lengths)
        
        if mean_length < 1e-6:
            return 0.0
        
        std_length = np.std(valid_lengths)
        variance = std_length / mean_length
        
        return float(variance)
    
    def _compute_root_accel_std(self, positions: np.ndarray) -> float:
        """
        Compute root acceleration standard deviation.
        
        Formula:
        - Root midpoint: root_t = (hip_L + hip_R) / 2
        - Acceleration: a_t = r_t - 2*r_{t-1} + r_{t-2}
        - Magnitude: a_mag = ||a_t||
        - Return: std(a_mag)
        
        Args:
            positions: Array of shape (T, 17, 2)
            
        Returns:
            Standard deviation of root acceleration magnitude
        """
        T = positions.shape[0]
        
        if T < 3:
            return 0.0
        
        # Compute root midpoint: (hip_L + hip_R) / 2
        hip_left = positions[:, self.LEFT_HIP, :]  # (T, 2)
        hip_right = positions[:, self.RIGHT_HIP, :]  # (T, 2)
        root = (hip_left + hip_right) / 2.0  # (T, 2)
        
        # Compute acceleration: a_t = r_t - 2*r_{t-1} + r_{t-2}
        # Vectorized: root[2:] - 2*root[1:-1] + root[:-2]
        accel = root[2:] - 2.0 * root[1:-1] + root[:-2]  # (T-2, 2)
        
        # Compute magnitude
        accel_mag = np.linalg.norm(accel, axis=1)  # (T-2,)
        
        if len(accel_mag) == 0:
            return 0.0
        
        # Return standard deviation
        return float(np.std(accel_mag))
    
    def _compute_jerk_std(self, positions: np.ndarray) -> float:
        """
        Compute jerk (third derivative) standard deviation.
        
        Formula:
        - Root midpoint: root_t = (hip_L + hip_R) / 2
        - Acceleration: a_t = r_t - 2*r_{t-1} + r_{t-2}
        - Jerk: j_t = a_t - a_{t-1}
        - Magnitude: jerk_mag = ||j_t||
        - Return: std(jerk_mag)
        
        Args:
            positions: Array of shape (T, 17, 2)
            
        Returns:
            Standard deviation of jerk magnitude
        """
        T = positions.shape[0]
        
        if T < 4:
            return 0.0
        
        # Compute root midpoint: (hip_L + hip_R) / 2
        hip_left = positions[:, self.LEFT_HIP, :]  # (T, 2)
        hip_right = positions[:, self.RIGHT_HIP, :]  # (T, 2)
        root = (hip_left + hip_right) / 2.0  # (T, 2)
        
        # Compute acceleration: a_t = r_t - 2*r_{t-1} + r_{t-2}
        # Vectorized: root[2:] - 2*root[1:-1] + root[:-2]
        accel = root[2:] - 2.0 * root[1:-1] + root[:-2]  # (T-2, 2)
        
        # Compute jerk: j_t = a_t - a_{t-1}
        # Vectorized: accel[1:] - accel[:-1]
        if len(accel) < 2:
            return 0.0
        
        jerk = accel[1:] - accel[:-1]  # (T-3, 2)
        
        # Compute magnitude
        jerk_mag = np.linalg.norm(jerk, axis=1)  # (T-3,)
        
        if len(jerk_mag) == 0:
            return 0.0
        
        # Return standard deviation
        return float(np.std(jerk_mag))
    
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """
        Compute moving average of 1D array using cumulative method.
        
        This method produces cleaner center-aligned smoothing without
        edge distortion from convolution.
        
        Args:
            data: 1D numpy array
            window: Window size
            
        Returns:
            Moving average array (same length as input)
        """
        if len(data) < window:
            return np.zeros_like(data)
        
        # Cumulative sum method for cleaner edge handling
        cumsum = np.cumsum(np.insert(data, 0, 0))
        ma = (cumsum[window:] - cumsum[:-window]) / window
        
        # Pad to match original length (center-aligned)
        pad_left = window // 2
        pad_right = window - window // 2 - 1
        ma_padded = np.pad(ma, (pad_left, pad_right), mode='edge')
        
        return ma_padded
    
    def _empty_results(self) -> Dict:
        """
        Return empty results dictionary for edge cases.
        
        Returns:
            Dictionary with zero/empty values
        """
        return {
            "jitter": {i: 0.0 for i in range(17)},
            "spike_ratio": {i: 0.0 for i in range(17)},
            "limb_variance": {
                "left_thigh": 0.0,
                "right_thigh": 0.0,
                "left_shank": 0.0,
                "right_shank": 0.0
            },
            "root_accel_std": 0.0,
            "jerk_std": 0.0
        }


# Test hook (standalone, no pipeline integration)
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Create sample test data
    print("Testing Pose2DTemporalAnalyzer...")
    
    # Create analyzer
    analyzer = Pose2DTemporalAnalyzer()
    
    # Create minimal test PoseTrack
    from football_app.backend.models.pose_estimation_2d import PoseTrack, Pose2D, Keypoint2D
    from football_app.backend.models.detection_tracking import BoundingBox
    
    # Generate synthetic test data (T=100 frames)
    T = 100
    test_poses = []
    
    for t in range(T):
        # Create keypoints with some controlled jitter
        keypoints = []
        for j in range(17):
            # Base position with some temporal variation
            base_x = 100.0 + j * 10.0 + np.sin(t * 0.1) * 5.0
            base_y = 200.0 + j * 15.0 + np.cos(t * 0.1) * 5.0
            
            # Add small jitter
            jitter_x = np.random.normal(0, 0.5)
            jitter_y = np.random.normal(0, 0.5)
            
            keypoints.append(Keypoint2D(
                x=base_x + jitter_x,
                y=base_y + jitter_y,
                confidence=0.9
            ))
        
        bbox = BoundingBox(x=50.0, y=150.0, w=200.0, h=300.0, confidence=0.95)
        pose = Pose2D(frame_id=t + 1, keypoints=keypoints, bbox=bbox)
        test_poses.append(pose)
    
    # Create PoseTrack
    test_track = PoseTrack(track_id=1, poses=test_poses)
    
    # Run analysis
    results = analyzer.analyze_pose_track(test_track)
    
    # Print results
    print("\n" + "=" * 80)
    print("Temporal Analysis Results")
    print("=" * 80)
    
    print("\n📊 High-Frequency Jitter (per joint):")
    for joint_id, value in sorted(results["jitter"].items()):
        print(f"  Joint {joint_id:2d}: {value:.4f}")
    
    print("\n⚡ Velocity Spike Ratio (per joint):")
    for joint_id, value in sorted(results["spike_ratio"].items()):
        print(f"  Joint {joint_id:2d}: {value:.4f}")
    
    print("\n🦵 Limb Length Variance (normalized):")
    for limb, value in results["limb_variance"].items():
        print(f"  {limb:15s}: {value:.4f}")
    
    print("\n📈 Root Acceleration Std:")
    print(f"  {results['root_accel_std']:.4f}")
    
    print("\n📉 Jerk Std:")
    print(f"  {results['jerk_std']:.4f}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
