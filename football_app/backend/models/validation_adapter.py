"""
Validation-Only Adapter for Pre-Cropped Single-Player Videos

This adapter bypasses Stage 1 (Detection & Tracking) for validation datasets
where videos are already cropped to single players.

IMPORTANT: This is validation-only code, NOT for production use.

Input: Folder with cam_X.mp4 files (one player per video)
Output: PlayerTrackSet per camera (ready for Stage 2)

Constraints:
- Does NOT run YOLO or BoT-SORT
- Does NOT modify Stage 1, 2, or 3 code
- Does NOT save frames or images
- Does NOT add football logic or metrics
"""

from pathlib import Path
from typing import Dict
import cv2
from football_app.backend.models.detection_tracking import (
    PlayerTrack, PlayerTrackSet, PlayerDetection, BoundingBox, TrackStatus
)


class ValidationAdapter:
    """
    Adapter for pre-cropped single-player video validation.
    
    Treats each video as one PlayerTrack with full-frame bounding boxes.
    """
    
    def __init__(self):
        """Initialize the validation adapter."""
        pass
    
    def load_validation_folder(self, folder_path: Path) -> Dict[str, PlayerTrackSet]:
        """
        Load pre-cropped videos from a validation folder.
        
        Args:
            folder_path: Path to folder containing cam_X.mp4 or CAMX.avi files
            
        Returns:
            Dictionary mapping camera ID to PlayerTrackSet
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")
        
        # Find all video files matching cam_X or CAMX patterns
        video_files = []
        
        # Try lowercase cam_X.mp4 pattern
        video_files.extend(folder_path.glob("cam_*.mp4"))
        
        # Try uppercase CAMX.avi pattern
        video_files.extend(folder_path.glob("CAM*.avi"))
        
        # Try other common patterns
        video_files.extend(folder_path.glob("cam_*.avi"))
        video_files.extend(folder_path.glob("CAM*.mp4"))
        
        # Remove duplicates and sort
        video_files = sorted(set(video_files))
        
        if not video_files:
            raise ValueError(f"No camera video files found in: {folder_path}")
        
        results = {}
        
        for video_file in video_files:
            # Extract camera ID from filename
            # Handles: cam_0.mp4 -> cam_0, CAM0.avi -> CAM0
            camera_id = video_file.stem
            
            print(f"Processing {camera_id}...")
            
            # Process video and create PlayerTrackSet
            track_set = self._process_single_player_video(video_file, camera_id)
            results[camera_id] = track_set
        
        return results
    
    def _process_single_player_video(
        self, 
        video_path: Path, 
        camera_id: str
    ) -> PlayerTrackSet:
        """
        Process a single pre-cropped player video.
        
        Args:
            video_path: Path to video file
            camera_id: Camera identifier
            
        Returns:
            PlayerTrackSet with one track covering all frames
        """
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create PlayerTrack (track_id = 1 since there's only one player)
        player_track = PlayerTrack(track_id=1, status=TrackStatus.ACTIVE)
        
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            
            # Create full-frame bounding box
            bbox = BoundingBox(
                x=0.0,
                y=0.0,
                w=float(width),
                h=float(height),
                confidence=1.0
            )
            
            # Create detection
            detection = PlayerDetection(
                frame_id=frame_id,
                bbox=bbox,
                confidence=1.0
            )
            
            # Add to track
            player_track.add_detection(detection)
        
        cap.release()
        
        # Create PlayerTrackSet
        track_set = PlayerTrackSet(
            view_id=camera_id,
            num_frames=total_frames,
            metadata={
                "fps": fps,
                "width": width,
                "height": height,
                "source": str(video_path),
                "validation_mode": True
            }
        )
        
        # Add the single player track
        track_set.add_track(player_track)
        
        return track_set
