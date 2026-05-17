"""
Multi-View Detection and Tracking Orchestration Module

This module implements detection and tracking orchestration for ActionInstance objects
as specified in the Confidential Multi-View Human Motion Analysis Architecture.

Key Principles:
- Operates on ActionInstance objects
- Runs detection & tracking per CameraView
- Produces PlayerTrackSets per view
- Does NOT perform player selection
- Does NOT run pose estimation
- Does NOT save frames or images
- Compatible with football multi-player scenarios
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from enum import Enum
import cv2
import torch
from football_app.backend.models.action_instance import ActionInstance, CameraView, MediaType


class TrackStatus(Enum):
    """Status of a player track"""
    ACTIVE = "active"
    LOST = "lost"
    TERMINATED = "terminated"


@dataclass
class BoundingBox:
    """
    Bounding box representation for a detected player.
    
    Format: (x, y, w, h) where:
    - x, y: top-left corner coordinates
    - w, h: width and height
    """
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    
    def to_tlwh(self) -> Tuple[float, float, float, float]:
        """Return as (top, left, width, height) tuple"""
        return (self.x, self.y, self.w, self.h)
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Return as (x1, y1, x2, y2) tuple"""
        return (self.x, self.y, self.x + self.w, self.y + self.h)
    
    def area(self) -> float:
        """Calculate bounding box area"""
        return self.w * self.h
    
    def center(self) -> Tuple[float, float]:
        """Calculate center point"""
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class PlayerDetection:
    """
    Single frame detection of a player.
    
    Contains:
    - frame_id: Frame number in the video
    - bbox: Bounding box
    - confidence: Detection confidence score
    """
    frame_id: int
    bbox: BoundingBox
    confidence: float
    
    def __repr__(self) -> str:
        return f"Detection(frame={self.frame_id}, bbox=({self.bbox.x:.1f},{self.bbox.y:.1f},{self.bbox.w:.1f},{self.bbox.h:.1f}), conf={self.confidence:.2f})"


@dataclass
class PlayerTrack:
    """
    Temporal sequence of detections for a single player across frames.
    
    A PlayerTrack represents one player's trajectory through the video.
    In football scenarios, each player gets their own track.
    
    Properties:
    - track_id: Unique identifier for this track within the view
    - detections: List of PlayerDetection objects ordered by frame
    - status: Current tracking status
    """
    track_id: int
    detections: List[PlayerDetection] = field(default_factory=list)
    status: TrackStatus = TrackStatus.ACTIVE
    
    def add_detection(self, detection: PlayerDetection) -> None:
        """Add a detection to this track"""
        self.detections.append(detection)
    
    def get_detection_at_frame(self, frame_id: int) -> Optional[PlayerDetection]:
        """Get detection at a specific frame"""
        for detection in self.detections:
            if detection.frame_id == frame_id:
                return detection
        return None
    
    def num_detections(self) -> int:
        """Return number of detections in this track"""
        return len(self.detections)
    
    def frame_range(self) -> Tuple[int, int]:
        """Return (first_frame, last_frame) tuple"""
        if not self.detections:
            return (0, 0)
        frames = [d.frame_id for d in self.detections]
        return (min(frames), max(frames))
    
    def average_confidence(self) -> float:
        """Calculate average detection confidence"""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)
    
    def __repr__(self) -> str:
        start, end = self.frame_range()
        return f"PlayerTrack(id={self.track_id}, frames={start}-{end}, detections={self.num_detections()}, status={self.status.value})"


@dataclass
class PlayerTrackSet:
    """
    Collection of all player tracks for a single camera view.
    
    In football scenarios, this contains tracks for ALL players visible in the view.
    No player selection is performed at this stage.
    
    Properties:
    - view_id: Identifier of the camera view
    - tracks: List of PlayerTrack objects
    - num_frames: Total number of frames processed
    - metadata: Optional metadata about the tracking process
    """
    view_id: str
    tracks: List[PlayerTrack] = field(default_factory=list)
    num_frames: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def add_track(self, track: PlayerTrack) -> None:
        """Add a player track to this set"""
        self.tracks.append(track)
    
    def get_track_by_id(self, track_id: int) -> Optional[PlayerTrack]:
        """Retrieve a specific track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def num_tracks(self) -> int:
        """Return number of tracks"""
        return len(self.tracks)
    
    def get_active_tracks(self) -> List[PlayerTrack]:
        """Return only active tracks"""
        return [t for t in self.tracks if t.status == TrackStatus.ACTIVE]
    
    def get_tracks_at_frame(self, frame_id: int) -> List[Tuple[int, PlayerDetection]]:
        """Get all detections at a specific frame as (track_id, detection) tuples"""
        results = []
        for track in self.tracks:
            detection = track.get_detection_at_frame(frame_id)
            if detection:
                results.append((track.track_id, detection))
        return results
    
    def __repr__(self) -> str:
        return f"PlayerTrackSet(view={self.view_id}, tracks={self.num_tracks()}, frames={self.num_frames})"


class DetectionTrackingOrchestrator:
    """
    Orchestrates detection and tracking across multiple camera views of an ActionInstance.
    
    This class:
    - Takes an ActionInstance as input
    - Runs detection & tracking on each CameraView independently
    - Produces a PlayerTrackSet for each view
    - Does NOT perform player selection
    - Does NOT run pose estimation
    - Does NOT save frames or images
    
    Compatible with multi-player football scenarios.
    """
    
    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: float = 10.0,
        aspect_ratio_thresh: float = 1.6
    ):
        """
        Initialize the orchestrator.
        
        Args:
            yolo_model_path: Path to YOLO model weights (optional)
            device: Device to run inference on
            track_high_thresh: High confidence threshold for tracking
            track_low_thresh: Low confidence threshold for tracking
            new_track_thresh: Threshold for creating new tracks
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Matching threshold for data association
            min_box_area: Minimum bounding box area
            aspect_ratio_thresh: Maximum aspect ratio for valid boxes
        """
        self.device = device
        self.yolo_model_path = yolo_model_path
        
        # Tracking parameters
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.aspect_ratio_thresh = aspect_ratio_thresh
        
        # Lazy-load models
        self._yolo_model = None
        self._tracker = None
    
    def _load_yolo_model(self):
        """Lazy-load YOLO model"""
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
                if self.yolo_model_path and Path(self.yolo_model_path).exists():
                    self._yolo_model = YOLO(self.yolo_model_path)
                else:
                    # Use default YOLOv8 model
                    self._yolo_model = YOLO('yolov8n.pt')
            except ImportError:
                raise ImportError("ultralytics package is required for YOLO detection")
        return self._yolo_model
    
    def _initialize_tracker(self):
        """Initialize BoT-SORT tracker"""
        if self._tracker is None:
            try:
                import sys
                from pathlib import Path
                # Add parent of bot_sort to path so bot_sort can be imported as a package
                bot_sort_parent = Path(__file__).parent.parent.parent.parent / "3dsp_utils"
                sys.path.insert(0, str(bot_sort_parent))
                
                from bot_sort.tracker.bot_sort_app import BoTSORT
                
                # Create args object for BoTSORT
                class Args:
                    pass
                
                args = Args()
                args.track_high_thresh = self.track_high_thresh
                args.track_low_thresh = self.track_low_thresh
                args.new_track_thresh = self.new_track_thresh
                args.track_buffer = self.track_buffer
                args.match_thresh = self.match_thresh
                args.aspect_ratio_thresh = self.aspect_ratio_thresh
                args.min_box_area = self.min_box_area
                args.cmc_method = "sparseOptFlow"
                args.with_reid = False
                args.proximity_thresh = 0.5
                args.appearance_thresh = 0.25
                args.name = "BoTSORT"
                args.ablation = ""
                args.device = self.device
                args.mot20 = False  # MOT20 dataset specific flag
                
                # Create tracker
                self._tracker = BoTSORT(args, frame_rate=25)
            except ImportError as e:
                raise ImportError(f"BoT-SORT tracker dependencies not available: {e}")
        return self._tracker
    
    def process_action_instance(self, action_instance: ActionInstance) -> Dict[str, PlayerTrackSet]:
        """
        Process an entire ActionInstance, running detection and tracking on all views.
        
        Args:
            action_instance: ActionInstance object to process
            
        Returns:
            Dictionary mapping view_id to PlayerTrackSet
        """
        results = {}
        
        for camera_view in action_instance.camera_views:
            print(f"Processing view: {camera_view.view_id}")
            track_set = self.process_camera_view(camera_view)
            results[camera_view.view_id] = track_set
        
        return results
    
    def process_camera_view(self, camera_view: CameraView) -> PlayerTrackSet:
        """
        Process a single camera view, running detection and tracking.
        
        Args:
            camera_view: CameraView object to process
            
        Returns:
            PlayerTrackSet containing all detected player tracks
        """
        if camera_view.media_type == MediaType.VIDEO:
            return self._process_video(camera_view)
        elif camera_view.media_type == MediaType.IMAGE_SEQUENCE:
            return self._process_image_sequence(camera_view)
        else:
            raise ValueError(f"Unsupported media type: {camera_view.media_type}")
    
    def _process_video(self, camera_view: CameraView) -> PlayerTrackSet:
        """
        Process a video file for detection and tracking.
        
        Args:
            camera_view: CameraView with video file
            
        Returns:
            PlayerTrackSet with all player tracks
        """
        # Initialize models
        yolo_model = self._load_yolo_model()
        tracker = self._initialize_tracker()
        
        # Open video
        cap = cv2.VideoCapture(str(camera_view.media_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {camera_view.media_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create track set
        track_set = PlayerTrackSet(
            view_id=camera_view.view_id,
            num_frames=total_frames,
            metadata={
                "fps": fps,
                "width": width,
                "height": height,
                "source": str(camera_view.media_path)
            }
        )
        
        # Track storage: track_id -> PlayerTrack
        tracks_dict: Dict[int, PlayerTrack] = {}
        
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            
            # Run YOLO detection
            results = yolo_model(frame, verbose=False)
            
            # Extract detections (filter for person class = 0)
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        # Convert to tlwh format
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        detections.append([x, y, w, h, conf])
            
            if len(detections) == 0:
                detections = np.empty((0, 5))
            else:
                detections = np.array(detections)
            
            # Run tracker update
            online_targets = tracker.update(detections, frame)
            
            # Process tracked objects
            for track in online_targets:
                track_id = track.track_id
                tlwh = track.tlwh
                conf = track.score
                
                # Create bounding box
                bbox = BoundingBox(
                    x=float(tlwh[0]),
                    y=float(tlwh[1]),
                    w=float(tlwh[2]),
                    h=float(tlwh[3]),
                    confidence=float(conf)
                )
                
                # Create detection
                detection = PlayerDetection(
                    frame_id=frame_id,
                    bbox=bbox,
                    confidence=float(conf)
                )
                
                # Add to track
                if track_id not in tracks_dict:
                    tracks_dict[track_id] = PlayerTrack(track_id=track_id)
                
                tracks_dict[track_id].add_detection(detection)
        
        cap.release()
        
        # Add all tracks to track set
        for track in tracks_dict.values():
            track_set.add_track(track)
        
        return track_set
    
    def _process_image_sequence(self, camera_view: CameraView) -> PlayerTrackSet:
        """
        Process an image sequence for detection and tracking.
        
        Args:
            camera_view: CameraView with image sequence directory
            
        Returns:
            PlayerTrackSet with all player tracks
        """
        # Initialize models
        yolo_model = self._load_yolo_model()
        tracker = self._initialize_tracker()
        
        # Get sorted list of image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            f for f in camera_view.media_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No images found in: {camera_view.media_path}")
        
        # Read first image to get dimensions
        first_img = cv2.imread(str(image_files[0]))
        height, width = first_img.shape[:2]
        
        # Create track set
        track_set = PlayerTrackSet(
            view_id=camera_view.view_id,
            num_frames=len(image_files),
            metadata={
                "fps": 25,  # Default assumption
                "width": width,
                "height": height,
                "source": str(camera_view.media_path)
            }
        )
        
        # Track storage
        tracks_dict: Dict[int, PlayerTrack] = {}
        
        for frame_id, image_path in enumerate(image_files, start=1):
            # Read image
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            
            # Run YOLO detection
            results = yolo_model(frame, verbose=False)
            
            # Extract detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        detections.append([x, y, w, h, conf])
            
            if len(detections) == 0:
                detections = np.empty((0, 5))
            else:
                detections = np.array(detections)
            
            # Run tracker update
            online_targets = tracker.update(detections, frame)
            
            # Process tracked objects
            for track in online_targets:
                track_id = track.track_id
                tlwh = track.tlwh
                conf = track.score
                
                bbox = BoundingBox(
                    x=float(tlwh[0]),
                    y=float(tlwh[1]),
                    w=float(tlwh[2]),
                    h=float(tlwh[3]),
                    confidence=float(conf)
                )
                
                detection = PlayerDetection(
                    frame_id=frame_id,
                    bbox=bbox,
                    confidence=float(conf)
                )
                
                if track_id not in tracks_dict:
                    tracks_dict[track_id] = PlayerTrack(track_id=track_id)
                
                tracks_dict[track_id].add_detection(detection)
        
        # Add all tracks to track set
        for track in tracks_dict.values():
            track_set.add_track(track)
        
        return track_set


def export_trackset_to_mot_format(track_set: PlayerTrackSet, output_path: Path) -> None:
    """
    Export PlayerTrackSet to MOT challenge format for compatibility.
    
    Format: frame,id,x,y,w,h,score,-1,-1,-1
    
    Args:
        track_set: PlayerTrackSet to export
        output_path: Path to output file
    """
    with open(output_path, 'w') as f:
        for track in track_set.tracks:
            for detection in track.detections:
                line = f"{detection.frame_id},{track.track_id}," \
                       f"{detection.bbox.x:.1f},{detection.bbox.y:.1f}," \
                       f"{detection.bbox.w:.1f},{detection.bbox.h:.1f}," \
                       f"{detection.confidence:.2f},-1,-1,-1\n"
                f.write(line)
