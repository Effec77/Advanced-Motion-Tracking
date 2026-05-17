"""
Unit tests for Stage 4: Motion Metrics & Stability Engine

Tests cover:
- Data structure creation and validation
- Metric computation correctness
- Stability index calculation
- Symmetry metric calculation
- Status handling (COMPLETE, PARTIAL, FAILED)
- Determinism verification
"""

import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
from football_app.backend.models.motion_metrics import (
    MotionMetricsEngine, MetricTrack, MetricTrackSet, MetricStatus,
    JointMetrics, FrameMetrics, SummaryMetrics
)
from football_app.backend.models.pose_lifting_3d import (
    Pose3DTrack, Pose3DTrackSet, Pose3D, Joint3D, Pose3DStatus
)


class TestDataStructures(unittest.TestCase):
    """Test metric data structures"""
    
    def test_joint_metrics_creation(self):
        """Test JointMetrics creation"""
        velocities = np.array([0.0, 1.0, 2.0])
        accelerations = np.array([0.0, 1.0, 1.0])
        jerks = np.array([0.0, 1.0, 0.0])
        
        jm = JointMetrics(
            joint_id=5,
            velocities=velocities,
            accelerations=accelerations,
            jerks=jerks
        )
        
        self.assertEqual(jm.joint_id, 5)
        self.assertEqual(len(jm.velocities), 3)
        np.testing.assert_array_equal(jm.velocities, velocities)
    
    def test_frame_metrics_creation(self):
        """Test FrameMetrics creation"""
        fm = FrameMetrics(
            frame_id=10,
            V_frame=1.5,
            A_frame=0.5,
            J_frame=0.2,
            D_COM=0.8
        )
        
        self.assertEqual(fm.frame_id, 10)
        self.assertAlmostEqual(fm.V_frame, 1.5)
        self.assertAlmostEqual(fm.D_COM, 0.8)
    
    def test_summary_metrics_creation(self):
        """Test SummaryMetrics creation"""
        sm = SummaryMetrics(
            V_avg=1.2,
            V_peak=2.5,
            A_var=0.3,
            J_avg=0.15,
            Stability=0.85,
            Symmetry=0.92
        )
        
        self.assertAlmostEqual(sm.V_avg, 1.2)
        self.assertAlmostEqual(sm.Stability, 0.85)
        self.assertAlmostEqual(sm.Symmetry, 0.92)
        
        # Verify ranges
        self.assertGreater(sm.Stability, 0.0)
        self.assertLessEqual(sm.Stability, 1.0)
        self.assertGreaterEqual(sm.Symmetry, 0.0)
        self.assertLessEqual(sm.Symmetry, 1.0)
    
    def test_metric_track_creation(self):
        """Test MetricTrack creation"""
        mt = MetricTrack(
            track_id=1,
            status=MetricStatus.COMPLETE
        )
        
        self.assertEqual(mt.track_id, 1)
        self.assertEqual(mt.status, MetricStatus.COMPLETE)
        self.assertEqual(mt.num_frames(), 0)
    
    def test_metric_track_set_creation(self):
        """Test MetricTrackSet creation"""
        mts = MetricTrackSet(view_id="cam_0")
        
        self.assertEqual(mts.view_id, "cam_0")
        self.assertEqual(mts.num_tracks(), 0)
        
        # Add tracks
        mt1 = MetricTrack(track_id=1, status=MetricStatus.COMPLETE)
        mt2 = MetricTrack(track_id=2, status=MetricStatus.PARTIAL)
        mts.add_metric_track(mt1)
        mts.add_metric_track(mt2)
        
        self.assertEqual(mts.num_tracks(), 2)
        self.assertEqual(len(mts.get_complete_tracks()), 1)


class TestMotionMetricsEngine(unittest.TestCase):
    """Test MotionMetricsEngine computation"""
    
    def setUp(self):
        """Create test engine"""
        self.engine = MotionMetricsEngine()
    
    def _create_test_pose_3d_track(self, num_frames: int = 10) -> Pose3DTrack:
        """Create a test Pose3DTrack with synthetic motion"""
        track = Pose3DTrack(track_id=1, status=Pose3DStatus.VALID)
        
        for t in range(num_frames):
            # Create synthetic joint positions (simple linear motion)
            joints = []
            for j in range(17):
                # Linear motion in x, constant in y and z
                x = float(t * 0.1 + j * 0.01)
                y = float(j * 0.1)
                z = float(j * 0.05)
                joints.append(Joint3D(x=x, y=y, z=z))
            
            pose_3d = Pose3D(frame_id=t, joints=joints)
            track.add_pose_3d(pose_3d)
        
        return track
    
    def test_process_valid_track(self):
        """Test processing a VALID Pose3DTrack"""
        pose_3d_track = self._create_test_pose_3d_track(num_frames=10)
        
        metric_track = self.engine._process_single_pose_3d_track(pose_3d_track)
        
        self.assertEqual(metric_track.track_id, 1)
        self.assertEqual(metric_track.status, MetricStatus.COMPLETE)
        self.assertEqual(len(metric_track.joint_metrics), 17)
        self.assertEqual(len(metric_track.frame_metrics), 10)
        self.assertIsNotNone(metric_track.summary_metrics)
    
    def test_process_insufficient_length_track(self):
        """Test processing track with < 3 frames"""
        pose_3d_track = self._create_test_pose_3d_track(num_frames=2)
        
        metric_track = self.engine._process_single_pose_3d_track(pose_3d_track)
        
        self.assertEqual(metric_track.status, MetricStatus.PARTIAL)
        self.assertEqual(len(metric_track.frame_metrics), 2)
        self.assertIsNone(metric_track.summary_metrics)
    
    def test_process_failed_track(self):
        """Test processing a FAILED Pose3DTrack"""
        pose_3d_track = Pose3DTrack(track_id=1, status=Pose3DStatus.FAILED)
        
        metric_track = self.engine._process_single_pose_3d_track(pose_3d_track)
        
        self.assertEqual(metric_track.status, MetricStatus.FAILED)
        self.assertEqual(len(metric_track.joint_metrics), 0)
        self.assertEqual(len(metric_track.frame_metrics), 0)
    
    def test_joint_metrics_computation(self):
        """Test joint-level metric computation"""
        # Create simple trajectory: stationary → moving
        poses_3d = np.zeros((5, 17, 3))
        
        # Joint 0: stationary
        poses_3d[:, 0, :] = [0, 0, 0]
        
        # Joint 1: constant velocity
        for t in range(5):
            poses_3d[t, 1, :] = [t * 1.0, 0, 0]
        
        joint_metrics_list = self.engine._compute_joint_metrics(poses_3d)
        
        self.assertEqual(len(joint_metrics_list), 17)
        
        # Joint 0 should have zero velocity
        self.assertAlmostEqual(joint_metrics_list[0].velocities[1], 0.0)
        
        # Joint 1 should have constant velocity ≈ 1.0
        self.assertAlmostEqual(joint_metrics_list[1].velocities[1], 1.0, places=5)
        self.assertAlmostEqual(joint_metrics_list[1].velocities[2], 1.0, places=5)
    
    def test_stability_index_range(self):
        """Test stability index is in valid range"""
        pose_3d_track = self._create_test_pose_3d_track(num_frames=10)
        metric_track = self.engine._process_single_pose_3d_track(pose_3d_track)
        
        stability = metric_track.summary_metrics.Stability
        
        self.assertGreater(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
    
    def test_symmetry_metric_range(self):
        """Test symmetry metric is in valid range"""
        pose_3d_track = self._create_test_pose_3d_track(num_frames=10)
        metric_track = self.engine._process_single_pose_3d_track(pose_3d_track)
        
        symmetry = metric_track.summary_metrics.Symmetry
        
        self.assertGreaterEqual(symmetry, 0.0)
        self.assertLessEqual(symmetry, 1.0)
    
    def test_determinism(self):
        """Test that computation is deterministic"""
        pose_3d_track = self._create_test_pose_3d_track(num_frames=10)
        
        # Compute twice
        metric_track_1 = self.engine._process_single_pose_3d_track(pose_3d_track)
        metric_track_2 = self.engine._process_single_pose_3d_track(pose_3d_track)
        
        # Should be identical
        self.assertAlmostEqual(
            metric_track_1.summary_metrics.V_avg,
            metric_track_2.summary_metrics.V_avg
        )
        self.assertAlmostEqual(
            metric_track_1.summary_metrics.Stability,
            metric_track_2.summary_metrics.Stability
        )
        self.assertAlmostEqual(
            metric_track_1.summary_metrics.Symmetry,
            metric_track_2.summary_metrics.Symmetry
        )
    
    def test_process_pose_3d_track_set(self):
        """Test processing a complete Pose3DTrackSet"""
        # Create test Pose3DTrackSet
        pose_3d_track_set = Pose3DTrackSet(view_id="cam_0")
        
        track1 = self._create_test_pose_3d_track(num_frames=10)
        track2 = self._create_test_pose_3d_track(num_frames=15)
        
        pose_3d_track_set.add_pose_3d_track(track1)
        pose_3d_track_set.add_pose_3d_track(track2)
        
        # Process
        metric_track_set = self.engine.process_pose_3d_track_set(pose_3d_track_set)
        
        self.assertEqual(metric_track_set.view_id, "cam_0")
        self.assertEqual(metric_track_set.num_tracks(), 2)
        self.assertEqual(len(metric_track_set.get_complete_tracks()), 2)
    
    def test_com_trajectory_computation(self):
        """Test COM trajectory computation"""
        poses_3d = np.zeros((5, 17, 3))
        
        # Set hip positions
        poses_3d[:, 11, :] = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]]  # Left hip
        poses_3d[:, 12, :] = [[1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 1, 0], [5, 1, 0]]  # Right hip
        
        com_trajectory = self.engine._compute_com_trajectory(poses_3d)
        
        self.assertEqual(com_trajectory.shape, (5, 3))
        
        # COM should be midpoint of hips
        expected_com_0 = np.array([1.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(com_trajectory[0], expected_com_0)


class TestMetricCorrectness(unittest.TestCase):
    """Test mathematical correctness of metrics"""
    
    def setUp(self):
        """Create test engine"""
        self.engine = MotionMetricsEngine()
    
    def test_velocity_computation(self):
        """Test velocity = ||p_t - p_{t-1}||"""
        # Create trajectory with known velocities
        poses_3d = np.zeros((3, 17, 3))
        
        # Joint 0: moves 1 unit per frame in x
        poses_3d[0, 0, :] = [0, 0, 0]
        poses_3d[1, 0, :] = [1, 0, 0]
        poses_3d[2, 0, :] = [2, 0, 0]
        
        joint_metrics_list = self.engine._compute_joint_metrics(poses_3d)
        
        # Velocity should be 1.0 for frames 1 and 2
        self.assertAlmostEqual(joint_metrics_list[0].velocities[0], 0.0)
        self.assertAlmostEqual(joint_metrics_list[0].velocities[1], 1.0)
        self.assertAlmostEqual(joint_metrics_list[0].velocities[2], 1.0)
    
    def test_acceleration_computation(self):
        """Test acceleration = v_t - v_{t-1}"""
        # Create trajectory with known accelerations
        poses_3d = np.zeros((4, 17, 3))
        
        # Joint 0: accelerating motion
        poses_3d[0, 0, :] = [0, 0, 0]
        poses_3d[1, 0, :] = [1, 0, 0]  # v=1
        poses_3d[2, 0, :] = [3, 0, 0]  # v=2, a=1
        poses_3d[3, 0, :] = [6, 0, 0]  # v=3, a=1
        
        joint_metrics_list = self.engine._compute_joint_metrics(poses_3d)
        
        # Acceleration should be 1.0 for frames 2 and 3
        self.assertAlmostEqual(joint_metrics_list[0].accelerations[2], 1.0)
        self.assertAlmostEqual(joint_metrics_list[0].accelerations[3], 1.0)
    
    def test_stability_index_formula(self):
        """Test stability index formula"""
        # Create track with known characteristics
        pose_3d_track = Pose3DTrack(track_id=1, status=Pose3DStatus.VALID)
        
        # Create very stable motion (constant velocity)
        for t in range(10):
            joints = []
            for j in range(17):
                x = float(t * 1.0)  # Constant velocity
                y = 0.0
                z = 0.0
                joints.append(Joint3D(x=x, y=y, z=z))
            pose_3d_track.add_pose_3d(Pose3D(frame_id=t, joints=joints))
        
        metric_track = self.engine._process_single_pose_3d_track(pose_3d_track)
        
        # Stable motion should have high stability index
        self.assertGreater(metric_track.summary_metrics.Stability, 0.5)


if __name__ == '__main__':
    unittest.main()
