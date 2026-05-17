"""
Stage 2: 2D Pose Estimation Module

This module implements 2D pose estimation for PlayerTrackSets as specified in the
Confidential Multi-View Human Motion Analysis Architecture.

Key Principles:
- Operates on PlayerTrackSet objects (output from Stage 1)
- Runs 2D pose estimation per PlayerTrack
- Produces PoseTrackSets with 17-keypoint COCO format
- Does NOT perform player selection
- Does NOT merge or compare across views
- Does NOT run 3D pose lifting
- Does NOT save images, crops, or visualizations
- Preserves frame alignment and keypoint confidence scores
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from football_app.backend.models.detection_tracking import (
    PlayerTrack,
    PlayerTrackSet,
    PlayerDetection,
    BoundingBox,
    CameraView,
    MediaType
)


@dataclass
class Keypoint2D:
    """
    Single 2D keypoint with confidence score.
    
    Properties:
    - x, y: Pixel coordinates in the image
    - confidence: Confidence score (0-1) from pose estimator
    """
    x: float
    y: float
    confidence: float
    
    def to_array(self) -> np.ndarray:
        """Return as numpy array [x, y, confidence]"""
        return np.array([self.x, self.y, self.confidence])
    
    def __repr__(self) -> str:
        return f"Keypoint2D(x={self.x:.1f}, y={self.y:.1f}, conf={self.confidence:.2f})"


@dataclass
class Pose2D:
    """
    2D pose for a single frame containing 17 keypoints in COCO format.
    
    COCO 17-keypoint format:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    Properties:
    - frame_id: Frame number corresponding to the detection
    - keypoints: List of 17 Keypoint2D objects in COCO order
    - bbox: Reference to the bounding box used for pose estimation
    """
    frame_id: int
    keypoints: List[Keypoint2D]
    bbox: BoundingBox
    
    def __post_init__(self):
        """Validate that we have exactly 17 keypoints"""
        if len(self.keypoints) != 17:
            raise ValueError(f"Expected 17 keypoints, got {len(self.keypoints)}")
    
    def to_array(self) -> np.ndarray:
        """Return keypoints as numpy array of shape (17, 3) with [x, y, confidence]"""
        return np.array([kp.to_array() for kp in self.keypoints])
    
    def get_keypoint(self, index: int) -> Keypoint2D:
        """Get keypoint by COCO index (0-16)"""
        if not 0 <= index < 17:
            raise IndexError(f"Keypoint index must be 0-16, got {index}")
        return self.keypoints[index]
    
    def average_confidence(self) -> float:
        """Calculate average confidence across all keypoints"""
        return sum(kp.confidence for kp in self.keypoints) / len(self.keypoints)
    
    def __repr__(self) -> str:
        avg_conf = self.average_confidence()
        return f"Pose2D(frame={self.frame_id}, keypoints=17, avg_conf={avg_conf:.2f})"


@dataclass
class PoseTrack:
    """
    Temporal sequence of 2D poses for a single player across frames.
    
    This parallels PlayerTrack but contains pose information instead of just bounding boxes.
    
    Properties:
    - track_id: Unique identifier matching the PlayerTrack
    - poses: List of Pose2D objects ordered by frame
    - player_track: Reference to the original PlayerTrack
    """
    track_id: int
    poses: List[Pose2D] = field(default_factory=list)
    player_track: Optional[PlayerTrack] = None
    
    def add_pose(self, pose: Pose2D) -> None:
        """Add a pose to this track"""
        self.poses.append(pose)
    
    def get_pose_at_frame(self, frame_id: int) -> Optional[Pose2D]:
        """Get pose at a specific frame"""
        for pose in self.poses:
            if pose.frame_id == frame_id:
                return pose
        return None
    
    def num_poses(self) -> int:
        """Return number of poses in this track"""
        return len(self.poses)
    
    def frame_range(self) -> Tuple[int, int]:
        """Return (first_frame, last_frame) tuple"""
        if not self.poses:
            return (0, 0)
        frames = [p.frame_id for p in self.poses]
        return (min(frames), max(frames))
    
    def average_confidence(self) -> float:
        """Calculate average confidence across all poses"""
        if not self.poses:
            return 0.0
        return sum(p.average_confidence() for p in self.poses) / len(self.poses)
    
    def to_array(self) -> np.ndarray:
        """
        Return all poses as numpy array of shape (T, 17, 3).
        
        T = number of frames
        17 = number of keypoints
        3 = [x, y, confidence]
        """
        if not self.poses:
            return np.empty((0, 17, 3))
        return np.array([pose.to_array() for pose in self.poses])
    
    def __repr__(self) -> str:
        start, end = self.frame_range()
        return f"PoseTrack(id={self.track_id}, frames={start}-{end}, poses={self.num_poses()})"


@dataclass
class PoseTrackSet:
    """
    Collection of all pose tracks for a single camera view.
    
    This parallels PlayerTrackSet but contains pose information.
    
    Properties:
    - view_id: Identifier of the camera view
    - pose_tracks: List of PoseTrack objects
    - player_track_set: Reference to the original PlayerTrackSet
    - metadata: Optional metadata about the pose estimation process
    """
    view_id: str
    pose_tracks: List[PoseTrack] = field(default_factory=list)
    player_track_set: Optional[PlayerTrackSet] = None
    metadata: Dict = field(default_factory=dict)
    
    def add_pose_track(self, pose_track: PoseTrack) -> None:
        """Add a pose track to this set"""
        self.pose_tracks.append(pose_track)
    
    def get_pose_track_by_id(self, track_id: int) -> Optional[PoseTrack]:
        """Retrieve a specific pose track by ID"""
        for pose_track in self.pose_tracks:
            if pose_track.track_id == track_id:
                return pose_track
        return None
    
    def num_tracks(self) -> int:
        """Return number of pose tracks"""
        return len(self.pose_tracks)
    
    def get_poses_at_frame(self, frame_id: int) -> List[Tuple[int, Pose2D]]:
        """Get all poses at a specific frame as (track_id, pose) tuples"""
        results = []
        for pose_track in self.pose_tracks:
            pose = pose_track.get_pose_at_frame(frame_id)
            if pose:
                results.append((pose_track.track_id, pose))
        return results
    
    def __repr__(self) -> str:
        return f"PoseTrackSet(view={self.view_id}, tracks={self.num_tracks()})"


class PoseEstimator2D:
    """
    2D pose estimation orchestrator that processes PlayerTrackSets.
    
    This class:
    - Takes a PlayerTrackSet and CameraView as input
    - Runs 2D pose estimation on each PlayerTrack
    - Produces a PoseTrackSet with 17-keypoint COCO format poses
    - Does NOT perform player selection
    - Does NOT save images or visualizations
    - Preserves frame alignment and confidence scores
    """
    
    def __init__(
        self,
        pose_model_path: Optional[str] = None,
        det_model_path: Optional[str] = None,
        device: str = "cpu",
        backend: str = "onnxruntime",
        pose_input_size: Tuple[int, int] = (288, 384),
        det_input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize the 2D pose estimator.
        
        Args:
            pose_model_path: Path to RTMPose ONNX model (optional, will download if None)
            det_model_path: Path to detection ONNX model (optional, will download if None)
            device: Device to run inference on ("cpu" or "cuda")
            backend: Backend for inference ("onnxruntime" or "openvino")
            pose_input_size: Input size for pose model (width, height)
            det_input_size: Input size for detection model (width, height)
        """
        self.pose_model_path = pose_model_path
        self.det_model_path = det_model_path
        self.device = device
        self.backend = backend
        self.pose_input_size = pose_input_size
        self.det_input_size = det_input_size
        
        # Lazy-load pose estimator
        self._pose_estimator = None
    
    def _load_pose_estimator(self):
        """Lazy-load RTMPose estimator"""
        if self._pose_estimator is None:
            try:
                import sys
                from pathlib import Path
                
                # Add rtmlib to path
                rtmlib_path = Path(__file__).parent.parent.parent.parent / "3dsp_utils" / "rtmlib"
                sys.path.insert(0, str(rtmlib_path))
                
                from rtmlib import Body
                
                # Use provided models or default URLs
                det_model = self.det_model_path or 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip'
                pose_model = self.pose_model_path or 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip'
                
                self._pose_estimator = Body(
                    det=det_model,
                    det_input_size=self.det_input_size,
                    pose=pose_model,
                    pose_input_size=self.pose_input_size,
                    backend=self.backend,
                    device=self.device
                )
                
            except ImportError as e:
                raise ImportError(f"rtmlib package is required for 2D pose estimation: {e}")
        
        return self._pose_estimator
    
    def process_player_track_set(
        self,
        player_track_set: PlayerTrackSet,
        camera_view: CameraView
    ) -> PoseTrackSet:
        """
        Process a PlayerTrackSet to produce a PoseTrackSet.
        
        Args:
            player_track_set: PlayerTrackSet from Stage 1
            camera_view: CameraView to read frames from
            
        Returns:
            PoseTrackSet with 2D poses for all tracks
        """
        # Initialize pose estimator
        pose_estimator = self._load_pose_estimator()
        
        # Create pose track set
        pose_track_set = PoseTrackSet(
            view_id=player_track_set.view_id,
            player_track_set=player_track_set,
            metadata={
                "pose_model": "RTMPose",
                "keypoint_format": "COCO-17",
                "source": str(camera_view.media_path)
            }
        )
        
        # Process based on media type
        if camera_view.media_type == MediaType.VIDEO:
            self._process_video_tracks(
                player_track_set,
                camera_view,
                pose_estimator,
                pose_track_set
            )
        elif camera_view.media_type == MediaType.IMAGE_SEQUENCE:
            self._process_image_sequence_tracks(
                player_track_set,
                camera_view,
                pose_estimator,
                pose_track_set
            )
        else:
            raise ValueError(f"Unsupported media type: {camera_view.media_type}")
        
        return pose_track_set
    
    def _process_video_tracks(
        self,
        player_track_set: PlayerTrackSet,
        camera_view: CameraView,
        pose_estimator,
        pose_track_set: PoseTrackSet
    ) -> None:
        """
        Process tracks from a video file.
        
        Args:
            player_track_set: PlayerTrackSet to process
            camera_view: CameraView with video file
            pose_estimator: RTMPose estimator instance
            pose_track_set: PoseTrackSet to populate
        """
        # Open video
        cap = cv2.VideoCapture(str(camera_view.media_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {camera_view.media_path}")
        
        # Build frame-to-detections mapping for efficient lookup
        frame_detections: Dict[int, List[Tuple[int, PlayerDetection]]] = {}
        for track in player_track_set.tracks:
            for detection in track.detections:
                frame_id = detection.frame_id
                if frame_id not in frame_detections:
                    frame_detections[frame_id] = []
                frame_detections[frame_id].append((track.track_id, detection))
        
        # Initialize pose tracks
        pose_tracks_dict: Dict[int, PoseTrack] = {}
        for track in player_track_set.tracks:
            pose_tracks_dict[track.track_id] = PoseTrack(
                track_id=track.track_id,
                player_track=track
            )
        
        # Process frames
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            
            # Skip frames with no detections
            if frame_id not in frame_detections:
                continue
            
            # Process each detection in this frame
            for track_id, detection in frame_detections[frame_id]:
                # Extract crop using bounding box
                bbox = detection.bbox
                x1, y1 = int(bbox.x), int(bbox.y)
                x2, y2 = int(bbox.x + bbox.w), int(bbox.y + bbox.h)
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Skip invalid crops
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop = frame[y1:y2, x1:x2]
                
                # Run pose estimation on crop
                try:
                    keypoints, scores = pose_estimator(crop)
                    
                    # Handle multiple detections (take first one)
                    if len(keypoints) > 0:
                        kpts = keypoints[0] if keypoints.ndim == 3 else keypoints
                        scrs = scores[0] if scores.ndim == 2 else scores
                        
                        # Convert to absolute coordinates (crop -> full frame)
                        kpts_absolute = kpts.copy()
                        kpts_absolute[:, 0] += x1  # Add crop x offset
                        kpts_absolute[:, 1] += y1  # Add crop y offset
                        
                        # Create Keypoint2D objects
                        keypoint_objects = []
                        for i in range(17):
                            keypoint_objects.append(Keypoint2D(
                                x=float(kpts_absolute[i, 0]),
                                y=float(kpts_absolute[i, 1]),
                                confidence=float(scrs[i])
                            ))
                        
                        # Create Pose2D
                        pose = Pose2D(
                            frame_id=frame_id,
                            keypoints=keypoint_objects,
                            bbox=bbox
                        )
                        
                        # Add to pose track
                        pose_tracks_dict[track_id].add_pose(pose)
                
                except Exception as e:
                    # Skip this detection if pose estimation fails
                    print(f"Warning: Pose estimation failed for track {track_id} at frame {frame_id}: {e}")
                    continue
        
        cap.release()
        
        # Add all pose tracks to pose track set
        for pose_track in pose_tracks_dict.values():
            pose_track_set.add_pose_track(pose_track)
    
    def _process_image_sequence_tracks(
        self,
        player_track_set: PlayerTrackSet,
        camera_view: CameraView,
        pose_estimator,
        pose_track_set: PoseTrackSet
    ) -> None:
        """
        Process tracks from an image sequence.
        
        Args:
            player_track_set: PlayerTrackSet to process
            camera_view: CameraView with image sequence directory
            pose_estimator: RTMPose estimator instance
            pose_track_set: PoseTrackSet to populate
        """
        # Get sorted list of image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            f for f in camera_view.media_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No images found in: {camera_view.media_path}")
        
        # Build frame-to-detections mapping
        frame_detections: Dict[int, List[Tuple[int, PlayerDetection]]] = {}
        for track in player_track_set.tracks:
            for detection in track.detections:
                frame_id = detection.frame_id
                if frame_id not in frame_detections:
                    frame_detections[frame_id] = []
                frame_detections[frame_id].append((track.track_id, detection))
        
        # Initialize pose tracks
        pose_tracks_dict: Dict[int, PoseTrack] = {}
        for track in player_track_set.tracks:
            pose_tracks_dict[track.track_id] = PoseTrack(
                track_id=track.track_id,
                player_track=track
            )
        
        # Process images
        for frame_id, image_path in enumerate(image_files, start=1):
            # Skip frames with no detections
            if frame_id not in frame_detections:
                continue
            
            # Read image
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            
            # Process each detection in this frame
            for track_id, detection in frame_detections[frame_id]:
                # Extract crop using bounding box
                bbox = detection.bbox
                x1, y1 = int(bbox.x), int(bbox.y)
                x2, y2 = int(bbox.x + bbox.w), int(bbox.y + bbox.h)
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Skip invalid crops
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop = frame[y1:y2, x1:x2]
                
                # Run pose estimation on crop
                try:
                    keypoints, scores = pose_estimator(crop)
                    
                    # Handle multiple detections (take first one)
                    if len(keypoints) > 0:
                        kpts = keypoints[0] if keypoints.ndim == 3 else keypoints
                        scrs = scores[0] if scores.ndim == 2 else scores
                        
                        # Convert to absolute coordinates
                        kpts_absolute = kpts.copy()
                        kpts_absolute[:, 0] += x1
                        kpts_absolute[:, 1] += y1
                        
                        # Create Keypoint2D objects
                        keypoint_objects = []
                        for i in range(17):
                            keypoint_objects.append(Keypoint2D(
                                x=float(kpts_absolute[i, 0]),
                                y=float(kpts_absolute[i, 1]),
                                confidence=float(scrs[i])
                            ))
                        
                        # Create Pose2D
                        pose = Pose2D(
                            frame_id=frame_id,
                            keypoints=keypoint_objects,
                            bbox=bbox
                        )
                        
                        # Add to pose track
                        pose_tracks_dict[track_id].add_pose(pose)
                
                except Exception as e:
                    print(f"Warning: Pose estimation failed for track {track_id} at frame {frame_id}: {e}")
                    continue
        
        # Add all pose tracks to pose track set
        for pose_track in pose_tracks_dict.values():
            pose_track_set.add_pose_track(pose_track)


def export_pose_track_to_numpy(pose_track: PoseTrack, output_path: Path) -> None:
    """
    Export PoseTrack to numpy format for compatibility.
    
    Format: (T, 17, 3) array where T is number of frames
    
    Args:
        pose_track: PoseTrack to export
        output_path: Path to output .npy file
    """
    pose_array = pose_track.to_array()
    np.save(output_path, pose_array)


def export_pose_track_set_to_numpy(pose_track_set: PoseTrackSet, output_dir: Path) -> None:
    """
    Export all pose tracks in a PoseTrackSet to numpy files.
    
    Args:
        pose_track_set: PoseTrackSet to export
        output_dir: Directory to save .npy files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pose_track in pose_track_set.pose_tracks:
        filename = f"track_{pose_track.track_id:04d}.npy"
        output_path = output_dir / filename
        export_pose_track_to_numpy(pose_track, output_path)
