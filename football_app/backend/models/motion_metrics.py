"""
Stage 4: Motion Metrics & Stability Engine

This module implements motion metric computation for Pose3DTrackSets as specified
in the Stage 4: Motion Metrics & Stability Engine specification.

Key Principles:
- Operates on Pose3DTrackSet objects (output from Stage 3)
- Computes deterministic kinematic metrics
- Uses first-order finite differences (no smoothing)
- Produces frame-level and track-level metrics
- Calculates stability index and symmetry
- Does NOT fuse camera views
- Does NOT interpret football tactics
- Does NOT modify pose data
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np
from football_app.backend.models.pose_lifting_3d import Pose3DTrackSet, Pose3DTrack, Pose3DStatus


class MetricStatus(Enum):
    """Status of metric computation"""
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class JointMetrics:
    """
    Frame-level metrics for a single joint.
    
    Properties:
    - joint_id: Joint index (0-16 for COCO format)
    - velocities: Velocity magnitude per frame
    - accelerations: Acceleration magnitude per frame
    - jerks: Jerk magnitude per frame
    """
    joint_id: int
    velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    accelerations: np.ndarray = field(default_factory=lambda: np.array([]))
    jerks: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __repr__(self) -> str:
        return f"JointMetrics(joint={self.joint_id}, frames={len(self.velocities)})"


@dataclass
class FrameMetrics:
    """
    Aggregated metrics for a single frame across all joints.
    
    Properties:
    - frame_id: Frame number
    - V_frame: Mean velocity across all joints
    - A_frame: Mean acceleration across all joints
    - J_frame: Mean jerk across all joints
    - D_COM: COM displacement from previous frame
    """
    frame_id: int
    V_frame: float
    A_frame: float
    J_frame: float
    D_COM: float
    
    def __repr__(self) -> str:
        return f"FrameMetrics(frame={self.frame_id}, V={self.V_frame:.3f}, A={self.A_frame:.3f})"


@dataclass
class SummaryMetrics:
    """
    Track-level summary metrics computed across full sequence.
    
    Properties:
    - V_avg: Average velocity
    - V_peak: Maximum velocity
    - A_var: Acceleration variance
    - J_avg: Mean jerk magnitude
    - Stability: Stability index (0 < S ≤ 1)
    - Symmetry: Symmetry metric (0 ≤ Sym ≤ 1)
    """
    V_avg: float
    V_peak: float
    A_var: float
    J_avg: float
    Stability: float
    Symmetry: float
    
    def __repr__(self) -> str:
        return f"SummaryMetrics(V_avg={self.V_avg:.3f}, Stability={self.Stability:.3f}, Symmetry={self.Symmetry:.3f})"


@dataclass
class MetricTrack:
    """
    Complete metric set for a single Pose3DTrack.
    
    Properties:
    - track_id: Unique identifier matching Pose3DTrack
    - status: Metric computation status
    - joint_metrics: Per-joint frame-level metrics
    - frame_metrics: Per-frame aggregated metrics
    - summary_metrics: Track-level summary metrics
    - pose_3d_track: Reference to original Pose3DTrack
    """
    track_id: int
    status: MetricStatus
    joint_metrics: List[JointMetrics] = field(default_factory=list)
    frame_metrics: List[FrameMetrics] = field(default_factory=list)
    summary_metrics: Optional[SummaryMetrics] = None
    pose_3d_track: Optional[Pose3DTrack] = None
    
    def num_frames(self) -> int:
        """Return number of frames with metrics"""
        return len(self.frame_metrics)
    
    def __repr__(self) -> str:
        return f"MetricTrack(id={self.track_id}, frames={self.num_frames()}, status={self.status.value})"


@dataclass
class MetricTrackSet:
    """
    Collection of all metric tracks for a single camera view.
    
    Properties:
    - view_id: Identifier of the camera view
    - metric_tracks: List of MetricTrack objects
    - pose_3d_track_set: Reference to original Pose3DTrackSet
    - metadata: Optional metadata about metric computation
    """
    view_id: str
    metric_tracks: List[MetricTrack] = field(default_factory=list)
    pose_3d_track_set: Optional[Pose3DTrackSet] = None
    metadata: Dict = field(default_factory=dict)
    
    def add_metric_track(self, metric_track: MetricTrack) -> None:
        """Add a metric track to this set"""
        self.metric_tracks.append(metric_track)
    
    def get_metric_track_by_id(self, track_id: int) -> Optional[MetricTrack]:
        """Retrieve a specific metric track by ID"""
        for metric_track in self.metric_tracks:
            if metric_track.track_id == track_id:
                return metric_track
        return None
    
    def num_tracks(self) -> int:
        """Return number of metric tracks"""
        return len(self.metric_tracks)
    
    def get_complete_tracks(self) -> List[MetricTrack]:
        """Return only tracks with COMPLETE status"""
        return [t for t in self.metric_tracks if t.status == MetricStatus.COMPLETE]
    
    def __repr__(self) -> str:
        complete_count = len(self.get_complete_tracks())
        return f"MetricTrackSet(view={self.view_id}, tracks={self.num_tracks()}, complete={complete_count})"


class MotionMetricsEngine:
    """
    Motion metrics computation engine for Pose3DTrackSets.
    
    This class:
    - Takes a Pose3DTrackSet as input
    - Computes kinematic metrics using first-order finite differences
    - Produces frame-level and track-level metrics
    - Calculates stability index and symmetry
    - Does NOT smooth, interpolate, or modify pose data
    - Is deterministic and stateless
    """
    
    # COCO joint indices for paired joints (left/right)
    PAIRED_JOINTS = [
        (5, 6),   # Left shoulder, Right shoulder
        (7, 8),   # Left elbow, Right elbow
        (9, 10),  # Left wrist, Right wrist
        (11, 12), # Left hip, Right hip
        (13, 14), # Left knee, Right knee
        (15, 16)  # Left ankle, Right ankle
    ]
    
    # COCO joint indices for COM proxy
    COM_JOINTS = [11, 12]  # Left hip, Right hip (can add spine if available)
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize the motion metrics engine.
        
        Args:
            epsilon: Small constant to prevent division by zero
        """
        self.epsilon = epsilon
    
    def process_pose_3d_track_set(self, pose_3d_track_set: Pose3DTrackSet) -> MetricTrackSet:
        """
        Process a Pose3DTrackSet to produce a MetricTrackSet.
        
        Args:
            pose_3d_track_set: Pose3DTrackSet from Stage 3
            
        Returns:
            MetricTrackSet with metrics for all tracks
        """
        # Create metric track set
        metric_track_set = MetricTrackSet(
            view_id=pose_3d_track_set.view_id,
            pose_3d_track_set=pose_3d_track_set,
            metadata={
                "computation": "first-order-finite-differences",
                "smoothing": "none",
                "temporal_window": "none"
            }
        )
        
        # Process each pose 3D track independently
        for pose_3d_track in pose_3d_track_set.pose_3d_tracks:
            metric_track = self._process_single_pose_3d_track(pose_3d_track)
            metric_track_set.add_metric_track(metric_track)
        
        return metric_track_set
    
    def _process_single_pose_3d_track(self, pose_3d_track: Pose3DTrack) -> MetricTrack:
        """
        Process a single Pose3DTrack to produce a MetricTrack.
        
        Args:
            pose_3d_track: Pose3DTrack to process
            
        Returns:
            MetricTrack with computed metrics
        """
        # Create metric track
        metric_track = MetricTrack(
            track_id=pose_3d_track.track_id,
            status=MetricStatus.FAILED,
            pose_3d_track=pose_3d_track
        )
        
        # Handle FAILED tracks
        if pose_3d_track.status == Pose3DStatus.FAILED:
            metric_track.status = MetricStatus.FAILED
            return metric_track
        
        # Check minimum frames for computation
        num_poses = pose_3d_track.num_poses()
        if num_poses < 2:
            metric_track.status = MetricStatus.FAILED
            return metric_track
        
        try:
            # Extract 3D poses as numpy array (T, 17, 3)
            poses_3d_array = pose_3d_track.to_array()
            
            # Compute joint-level metrics
            joint_metrics_list = self._compute_joint_metrics(poses_3d_array)
            metric_track.joint_metrics = joint_metrics_list
            
            # Compute frame-level aggregated metrics
            frame_metrics_list = self._compute_frame_metrics(
                joint_metrics_list, poses_3d_array, pose_3d_track
            )
            metric_track.frame_metrics = frame_metrics_list
            
            # Compute track-level summary metrics (if sufficient frames)
            if num_poses >= 3:
                summary_metrics = self._compute_summary_metrics(
                    frame_metrics_list, joint_metrics_list
                )
                metric_track.summary_metrics = summary_metrics
                metric_track.status = MetricStatus.COMPLETE
            else:
                # INSUFFICIENT_LENGTH → PARTIAL
                metric_track.status = MetricStatus.PARTIAL
        
        except Exception as e:
            print(f"Warning: Metric computation failed for track {pose_3d_track.track_id}: {e}")
            metric_track.status = MetricStatus.FAILED
        
        return metric_track
    
    def _compute_joint_metrics(self, poses_3d: np.ndarray) -> List[JointMetrics]:
        """
        Compute per-joint frame-level metrics.
        
        Args:
            poses_3d: Array of shape (T, 17, 3)
            
        Returns:
            List of JointMetrics (one per joint)
        """
        T, num_joints, _ = poses_3d.shape
        joint_metrics_list = []
        
        for joint_id in range(num_joints):
            # Extract joint trajectory (T, 3)
            joint_traj = poses_3d[:, joint_id, :]
            
            # Compute velocity (||p_t - p_{t-1}||)
            velocities = np.zeros(T)
            for t in range(1, T):
                velocities[t] = np.linalg.norm(joint_traj[t] - joint_traj[t-1])
            
            # Compute acceleration (v_t - v_{t-1})
            accelerations = np.zeros(T)
            for t in range(1, T):
                accelerations[t] = abs(velocities[t] - velocities[t-1])
            
            # Compute jerk (a_t - a_{t-1})
            jerks = np.zeros(T)
            for t in range(1, T):
                jerks[t] = abs(accelerations[t] - accelerations[t-1])
            
            joint_metrics = JointMetrics(
                joint_id=joint_id,
                velocities=velocities,
                accelerations=accelerations,
                jerks=jerks
            )
            joint_metrics_list.append(joint_metrics)
        
        return joint_metrics_list
    
    def _compute_frame_metrics(
        self,
        joint_metrics_list: List[JointMetrics],
        poses_3d: np.ndarray,
        pose_3d_track: Pose3DTrack
    ) -> List[FrameMetrics]:
        """
        Compute per-frame aggregated metrics.
        
        Args:
            joint_metrics_list: List of JointMetrics
            poses_3d: Array of shape (T, 17, 3)
            pose_3d_track: Original Pose3DTrack for frame IDs
            
        Returns:
            List of FrameMetrics (one per frame)
        """
        T = poses_3d.shape[0]
        frame_metrics_list = []
        
        # Compute COM trajectory
        com_trajectory = self._compute_com_trajectory(poses_3d)
        
        for t in range(T):
            frame_id = pose_3d_track.poses_3d[t].frame_id
            
            # Aggregate velocities across joints
            velocities_t = [jm.velocities[t] for jm in joint_metrics_list]
            V_frame = np.mean(velocities_t)
            
            # Aggregate accelerations across joints
            accelerations_t = [jm.accelerations[t] for jm in joint_metrics_list]
            A_frame = np.mean(accelerations_t)
            
            # Aggregate jerks across joints
            jerks_t = [jm.jerks[t] for jm in joint_metrics_list]
            J_frame = np.mean(jerks_t)
            
            # COM displacement
            if t > 0:
                D_COM = np.linalg.norm(com_trajectory[t] - com_trajectory[t-1])
            else:
                D_COM = 0.0
            
            frame_metrics = FrameMetrics(
                frame_id=frame_id,
                V_frame=float(V_frame),
                A_frame=float(A_frame),
                J_frame=float(J_frame),
                D_COM=float(D_COM)
            )
            frame_metrics_list.append(frame_metrics)
        
        return frame_metrics_list
    
    def _compute_com_trajectory(self, poses_3d: np.ndarray) -> np.ndarray:
        """
        Compute center-of-mass proxy trajectory.
        
        COM = mean(left_hip, right_hip)
        
        Args:
            poses_3d: Array of shape (T, 17, 3)
            
        Returns:
            COM trajectory of shape (T, 3)
        """
        T = poses_3d.shape[0]
        com_trajectory = np.zeros((T, 3))
        
        for t in range(T):
            # Extract COM joints
            com_joints = poses_3d[t, self.COM_JOINTS, :]
            com_trajectory[t] = np.mean(com_joints, axis=0)
        
        return com_trajectory
    
    def _compute_summary_metrics(
        self,
        frame_metrics_list: List[FrameMetrics],
        joint_metrics_list: List[JointMetrics]
    ) -> SummaryMetrics:
        """
        Compute track-level summary metrics.
        
        Args:
            frame_metrics_list: List of FrameMetrics
            joint_metrics_list: List of JointMetrics
            
        Returns:
            SummaryMetrics
        """
        # Extract frame-level arrays
        V_frames = np.array([fm.V_frame for fm in frame_metrics_list])
        A_frames = np.array([fm.A_frame for fm in frame_metrics_list])
        J_frames = np.array([fm.J_frame for fm in frame_metrics_list])
        D_COMs = np.array([fm.D_COM for fm in frame_metrics_list])
        
        # Average velocity
        V_avg = float(np.mean(V_frames))
        
        # Maximum velocity
        V_peak = float(np.max(V_frames))
        
        # Acceleration variance
        A_var = float(np.var(A_frames))
        
        # Mean jerk magnitude
        J_avg = float(np.mean(J_frames))
        
        # Stability index
        S_acc = 1.0 / (1.0 + A_var)
        S_jerk = 1.0 / (1.0 + J_avg)
        S_com = 1.0 / (1.0 + np.var(D_COMs))
        Stability = float((S_acc + S_jerk + S_com) / 3.0)
        
        # Symmetry metric
        Symmetry = self._compute_symmetry(joint_metrics_list)
        
        return SummaryMetrics(
            V_avg=V_avg,
            V_peak=V_peak,
            A_var=A_var,
            J_avg=J_avg,
            Stability=Stability,
            Symmetry=Symmetry
        )
    
    def _compute_symmetry(self, joint_metrics_list: List[JointMetrics]) -> float:
        """
        Compute symmetry metric across paired joints.
        
        Sym_joint = 1 - |V_left - V_right| / (V_left + V_right + ε)
        Symmetry = mean(Sym_joint across all pairs)
        
        Args:
            joint_metrics_list: List of JointMetrics
            
        Returns:
            Symmetry value (0 ≤ Sym ≤ 1)
        """
        symmetry_scores = []
        
        for left_id, right_id in self.PAIRED_JOINTS:
            # Get velocities for paired joints
            V_left = np.mean(joint_metrics_list[left_id].velocities)
            V_right = np.mean(joint_metrics_list[right_id].velocities)
            
            # Compute symmetry for this pair
            numerator = abs(V_left - V_right)
            denominator = V_left + V_right + self.epsilon
            sym_joint = 1.0 - (numerator / denominator)
            
            # Clamp to [0, 1]
            sym_joint = max(0.0, min(1.0, sym_joint))
            symmetry_scores.append(sym_joint)
        
        # Average across all pairs
        return float(np.mean(symmetry_scores))
