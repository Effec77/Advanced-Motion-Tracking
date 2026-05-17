"""
Stage 3: 3D Pose Lifting Module

This module implements 3D pose lifting for PoseTrackSets as specified in the
Stage 3: 3D Pose Lifting Design Specification.

Key Principles:
- Operates on PoseTrackSet objects (output from Stage 2)
- Uses temporal 3D pose lifting (MotionAGFormer)
- Processes each PoseTrack independently
- Outputs relative, camera-centric 3D poses
- Does NOT fuse or compare multiple camera views
- Does NOT perform player selection
- Does NOT estimate absolute depth or world coordinates
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import torch
from football_app.backend.models.pose_estimation_2d import PoseTrackSet, PoseTrack, Pose2D


class Pose3DStatus(Enum):
    """Status of a 3D pose track"""
    VALID = "valid"
    INSUFFICIENT_LENGTH = "insufficient_length"
    FAILED = "failed"


@dataclass
class Joint3D:
    """
    Single 3D joint in camera-centric coordinates.
    
    Properties:
    - x, y, z: 3D coordinates relative to root joint
    - Root joint is defined as midpoint of left_hip and right_hip
    """
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        """Return as numpy array [x, y, z]"""
        return np.array([self.x, self.y, self.z])
    
    def __repr__(self) -> str:
        return f"Joint3D(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


@dataclass
class Pose3D:
    """
    3D pose for a single frame containing 17 joints.
    
    Joints are in COCO format, same order as 2D poses.
    Coordinates are relative to root joint (midpoint of hips).
    
    Properties:
    - frame_id: Frame number corresponding to the 2D pose
    - joints: List of 17 Joint3D objects in COCO order
    """
    frame_id: int
    joints: List[Joint3D]
    
    def __post_init__(self):
        """Validate that we have exactly 17 joints"""
        if len(self.joints) != 17:
            raise ValueError(f"Expected 17 joints, got {len(self.joints)}")
    
    def to_array(self) -> np.ndarray:
        """Return joints as numpy array of shape (17, 3) with [x, y, z]"""
        return np.array([joint.to_array() for joint in self.joints])
    
    def get_joint(self, index: int) -> Joint3D:
        """Get joint by COCO index (0-16)"""
        if not 0 <= index < 17:
            raise IndexError(f"Joint index must be 0-16, got {index}")
        return self.joints[index]
    
    def __repr__(self) -> str:
        return f"Pose3D(frame={self.frame_id}, joints=17)"


@dataclass
class Pose3DTrack:
    """
    Temporal sequence of 3D poses for a single player across frames.
    
    This parallels PoseTrack but contains 3D pose information.
    
    Properties:
    - track_id: Unique identifier matching the PoseTrack
    - poses_3d: List of Pose3D objects ordered by frame
    - status: Processing status (VALID, INSUFFICIENT_LENGTH, FAILED)
    - pose_track: Reference to the original PoseTrack
    """
    track_id: int
    poses_3d: List[Pose3D] = field(default_factory=list)
    status: Pose3DStatus = Pose3DStatus.VALID
    pose_track: Optional[PoseTrack] = None
    
    def add_pose_3d(self, pose_3d: Pose3D) -> None:
        """Add a 3D pose to this track"""
        self.poses_3d.append(pose_3d)
    
    def get_pose_3d_at_frame(self, frame_id: int) -> Optional[Pose3D]:
        """Get 3D pose at a specific frame"""
        for pose_3d in self.poses_3d:
            if pose_3d.frame_id == frame_id:
                return pose_3d
        return None
    
    def num_poses(self) -> int:
        """Return number of 3D poses in this track"""
        return len(self.poses_3d)
    
    def frame_range(self) -> Tuple[int, int]:
        """Return (first_frame, last_frame) tuple"""
        if not self.poses_3d:
            return (0, 0)
        frames = [p.frame_id for p in self.poses_3d]
        return (min(frames), max(frames))
    
    def to_array(self) -> np.ndarray:
        """
        Return all 3D poses as numpy array of shape (T, 17, 3).
        
        T = number of frames
        17 = number of joints
        3 = [x, y, z]
        """
        if not self.poses_3d:
            return np.empty((0, 17, 3))
        return np.array([pose.to_array() for pose in self.poses_3d])
    
    def __repr__(self) -> str:
        start, end = self.frame_range()
        return f"Pose3DTrack(id={self.track_id}, frames={start}-{end}, poses={self.num_poses()}, status={self.status.value})"


@dataclass
class Pose3DTrackSet:
    """
    Collection of all 3D pose tracks for a single camera view.
    
    This parallels PoseTrackSet but contains 3D pose information.
    
    Properties:
    - view_id: Identifier of the camera view
    - pose_3d_tracks: List of Pose3DTrack objects
    - pose_track_set: Reference to the original PoseTrackSet
    - metadata: Optional metadata about the 3D pose lifting process
    """
    view_id: str
    pose_3d_tracks: List[Pose3DTrack] = field(default_factory=list)
    pose_track_set: Optional[PoseTrackSet] = None
    metadata: Dict = field(default_factory=dict)
    
    def add_pose_3d_track(self, pose_3d_track: Pose3DTrack) -> None:
        """Add a 3D pose track to this set"""
        self.pose_3d_tracks.append(pose_3d_track)
    
    def get_pose_3d_track_by_id(self, track_id: int) -> Optional[Pose3DTrack]:
        """Retrieve a specific 3D pose track by ID"""
        for pose_3d_track in self.pose_3d_tracks:
            if pose_3d_track.track_id == track_id:
                return pose_3d_track
        return None
    
    def num_tracks(self) -> int:
        """Return number of 3D pose tracks"""
        return len(self.pose_3d_tracks)
    
    def get_valid_tracks(self) -> List[Pose3DTrack]:
        """Return only tracks with VALID status"""
        return [t for t in self.pose_3d_tracks if t.status == Pose3DStatus.VALID]
    
    def get_poses_3d_at_frame(self, frame_id: int) -> List[Tuple[int, Pose3D]]:
        """Get all 3D poses at a specific frame as (track_id, pose_3d) tuples"""
        results = []
        for pose_3d_track in self.pose_3d_tracks:
            pose_3d = pose_3d_track.get_pose_3d_at_frame(frame_id)
            if pose_3d:
                results.append((pose_3d_track.track_id, pose_3d))
        return results
    
    def __repr__(self) -> str:
        valid_count = len(self.get_valid_tracks())
        return f"Pose3DTrackSet(view={self.view_id}, tracks={self.num_tracks()}, valid={valid_count})"


class PoseLifter3D:
    """
    3D pose lifting orchestrator that processes PoseTrackSets.
    
    This class:
    - Takes a PoseTrackSet as input
    - Uses temporal 3D pose lifting (MotionAGFormer)
    - Processes each PoseTrack independently
    - Produces a Pose3DTrackSet with camera-centric 3D poses
    - Does NOT fuse or compare multiple camera views
    - Does NOT perform player selection
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temporal_window: int = 243,  # Must match checkpoint training (243 frames)
        confidence_threshold: float = 0.2
    ):
        """
        Initialize the 3D pose lifter.
        
        Args:
            model_path: Path to MotionAGFormer model weights (optional)
            device: Device to run inference on ("cpu" or "cuda")
            temporal_window: Temporal window size for lifting (default: 243, matches checkpoint)
            confidence_threshold: Threshold for masking low-confidence joints (default: 0.2)
        """
        self.model_path = model_path
        self.device = device
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        
        # Lazy-load model
        self._model = None
    
    def _load_model(self):
        """Lazy-load MotionAGFormer model"""
        if self._model is None:
            try:
                import sys
                from pathlib import Path
                
                # Add MotionAGFormer to path
                motionagformer_path = Path(__file__).parent.parent.parent.parent / "3dsp_utils" / "MotionAGFormer"
                sys.path.insert(0, str(motionagformer_path))
                
                from model.MotionAGFormer import MotionAGFormer
                
                # Initialize model with checkpoint-compatible parameters
                # NOTE: These parameters MUST match the checkpoint architecture
                # Checkpoint was trained with: dim_in=3, dim_feat=128, n_frames=243
                self._model = MotionAGFormer(
                    n_layers=16,
                    dim_in=3,  # Input dimension (x, y, confidence) - checkpoint expects 3
                    dim_feat=128,  # Feature dimension - checkpoint has 128
                    dim_rep=512,  # Motion representation dimension
                    dim_out=3,  # Output dimension (x, y, z)
                    mlp_ratio=4,
                    num_heads=8,
                    num_joints=17,
                    n_frames=243  # Checkpoint was trained with 243 frames
                )
                
                # Load weights if provided
                if self.model_path and Path(self.model_path).exists():
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    
                    # Extract state dict
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    
                    # Strip 'module.' prefix if present (from DataParallel training)
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('module.'):
                            new_key = key[7:]  # Remove 'module.' prefix
                        else:
                            new_key = key
                        new_state_dict[new_key] = value
                    
                    self._model.load_state_dict(new_state_dict)
                
                self._model.to(self.device)
                self._model.eval()
                
            except ImportError as e:
                raise ImportError(f"MotionAGFormer dependencies not available: {e}")
        
        return self._model
    
    def process_pose_track_set(self, pose_track_set: PoseTrackSet) -> Pose3DTrackSet:
        """
        Process a PoseTrackSet to produce a Pose3DTrackSet.
        
        Args:
            pose_track_set: PoseTrackSet from Stage 2
            
        Returns:
            Pose3DTrackSet with 3D poses for all tracks
        """
        # Initialize model
        model = self._load_model()
        
        # Create 3D pose track set
        pose_3d_track_set = Pose3DTrackSet(
            view_id=pose_track_set.view_id,
            pose_track_set=pose_track_set,
            metadata={
                "model": "MotionAGFormer",
                "temporal_window": self.temporal_window,
                "confidence_threshold": self.confidence_threshold,
                "coordinate_system": "camera-centric-relative"
            }
        )
        
        # Process each pose track independently
        for pose_track in pose_track_set.pose_tracks:
            pose_3d_track = self._process_single_pose_track(pose_track, model)
            pose_3d_track_set.add_pose_3d_track(pose_3d_track)
        
        return pose_3d_track_set
    
    def _process_single_pose_track(
        self,
        pose_track: PoseTrack,
        model
    ) -> Pose3DTrack:
        """
        Process a single PoseTrack to produce a Pose3DTrack.
        
        Args:
            pose_track: PoseTrack to process
            model: MotionAGFormer model instance
            
        Returns:
            Pose3DTrack with 3D poses
        """
        # Create 3D pose track
        pose_3d_track = Pose3DTrack(
            track_id=pose_track.track_id,
            pose_track=pose_track
        )
        
        # Check if track has sufficient length
        if pose_track.num_poses() < self.temporal_window:
            pose_3d_track.status = Pose3DStatus.INSUFFICIENT_LENGTH
            return pose_3d_track
        
        try:
            # Extract 2D poses as numpy array (T, 17, 3) where 3 = [x, y, confidence]
            poses_2d_array = pose_track.to_array()
            
            # Process in non-overlapping windows
            num_poses = poses_2d_array.shape[0]
            num_windows = num_poses // self.temporal_window
            
            if num_windows == 0:
                pose_3d_track.status = Pose3DStatus.INSUFFICIENT_LENGTH
                return pose_3d_track
            
            for window_idx in range(num_windows):
                start_idx = window_idx * self.temporal_window
                end_idx = start_idx + self.temporal_window
                
                # Extract window
                window_2d = poses_2d_array[start_idx:end_idx]  # (243, 17, 3)
                
                # Check for valid frames (mask low-confidence joints)
                confidences = window_2d[:, :, 2]  # (243, 17)
                valid_mask = confidences >= self.confidence_threshold
                
                # Skip window if too many invalid joints
                valid_ratio = valid_mask.sum() / valid_mask.size
                if valid_ratio < 0.5:  # Require at least 50% valid joints
                    continue
                
                # Prepare input for model (use x, y, confidence - 3 channels as checkpoint expects)
                input_2d = window_2d.copy()  # (243, 17, 3)
                
                # Apply confidence masking (set low-confidence joints to zero)
                for t in range(self.temporal_window):
                    for j in range(17):
                        if not valid_mask[t, j]:
                            input_2d[t, j] = 0.0
                
                # Normalize 2D poses (center around root joint)
                input_2d_normalized = self._normalize_2d_poses(input_2d)
                
                # Convert to tensor
                input_tensor = torch.from_numpy(input_2d_normalized).float().unsqueeze(0)  # (1, 243, 17, 3)
                input_tensor = input_tensor.to(self.device)
                
                # Run inference
                with torch.no_grad():
                    output_3d = model(input_tensor)  # (1, 243, 17, 3)
                
                # Convert to numpy
                output_3d_np = output_3d.cpu().numpy()[0]  # (243, 17, 3)
                
                # Create Pose3D objects
                for t in range(self.temporal_window):
                    frame_id = pose_track.poses[start_idx + t].frame_id
                    
                    # Create Joint3D objects
                    joints_3d = []
                    for j in range(17):
                        joints_3d.append(Joint3D(
                            x=float(output_3d_np[t, j, 0]),
                            y=float(output_3d_np[t, j, 1]),
                            z=float(output_3d_np[t, j, 2])
                        ))
                    
                    # Create Pose3D
                    pose_3d = Pose3D(
                        frame_id=frame_id,
                        joints=joints_3d
                    )
                    
                    pose_3d_track.add_pose_3d(pose_3d)
            
            # Mark as valid if we processed at least one window
            if pose_3d_track.num_poses() > 0:
                pose_3d_track.status = Pose3DStatus.VALID
            else:
                pose_3d_track.status = Pose3DStatus.INSUFFICIENT_LENGTH
        
        except Exception as e:
            # Mark as failed if any error occurs
            print(f"Warning: 3D pose lifting failed for track {pose_track.track_id}: {e}")
            pose_3d_track.status = Pose3DStatus.FAILED
        
        return pose_3d_track
    
    def _normalize_2d_poses(self, poses_2d: np.ndarray) -> np.ndarray:
        """
        Normalize 2D poses by centering around root joint.
        
        Root joint is defined as midpoint of left_hip (11) and right_hip (12).
        
        Args:
            poses_2d: Array of shape (T, 17, 2)
            
        Returns:
            Normalized poses of shape (T, 17, 2)
        """
        poses_2d_normalized = poses_2d.copy()
        
        for t in range(poses_2d.shape[0]):
            # Calculate root joint (midpoint of hips)
            left_hip = poses_2d[t, 11]  # COCO index 11
            right_hip = poses_2d[t, 12]  # COCO index 12
            root = (left_hip + right_hip) / 2.0
            
            # Center all joints around root
            poses_2d_normalized[t] = poses_2d[t] - root
        
        return poses_2d_normalized


def export_pose_3d_track_to_numpy(pose_3d_track: Pose3DTrack, output_path: Path) -> None:
    """
    Export Pose3DTrack to numpy format.
    
    Format: (T, 17, 3) array where T is number of frames
    
    Args:
        pose_3d_track: Pose3DTrack to export
        output_path: Path to output .npy file
    """
    pose_3d_array = pose_3d_track.to_array()
    np.save(output_path, pose_3d_array)


def export_pose_3d_track_set_to_numpy(pose_3d_track_set: Pose3DTrackSet, output_dir: Path) -> None:
    """
    Export all 3D pose tracks in a Pose3DTrackSet to numpy files.
    
    Args:
        pose_3d_track_set: Pose3DTrackSet to export
        output_dir: Directory to save .npy files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pose_3d_track in pose_3d_track_set.pose_3d_tracks:
        if pose_3d_track.status == Pose3DStatus.VALID:
            filename = f"track_3d_{pose_3d_track.track_id:04d}.npy"
            output_path = output_dir / filename
            export_pose_3d_track_to_numpy(pose_3d_track, output_path)
