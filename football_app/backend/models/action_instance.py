"""
ActionInstance Data Structure and Folder Ingestion Logic

This module implements the core ActionInstance abstraction as specified in the
Confidential Multi-View Human Motion Analysis Architecture documentation.

Key Principles:
- One folder = one ActionInstance
- Each video/image sequence = one CameraView
- No video processing, no ML, no sport assumptions
- Multi-view support by design
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
import configparser


class MediaType(Enum):
    """Type of media in a camera view"""
    VIDEO = "video"
    IMAGE_SEQUENCE = "image_sequence"
    UNKNOWN = "unknown"


@dataclass
class CameraView:
    """
    Represents one camera's observation of an ActionInstance.
    
    A CameraView contains:
    - Path to the media (video file or image sequence directory)
    - Media type (video or image sequence)
    - Optional camera metadata
    
    No processing, no ML, no assumptions about content.
    """
    view_id: str
    media_path: Path
    media_type: MediaType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that the media path exists"""
        if not self.media_path.exists():
            raise FileNotFoundError(f"Media path does not exist: {self.media_path}")
    
    def __repr__(self) -> str:
        return f"CameraView(id={self.view_id}, type={self.media_type.value}, path={self.media_path})"


@dataclass
class ActionInstance:
    """
    Canonical unit of analysis representing one real-world physical movement event.
    
    An ActionInstance may be observed by multiple cameras (CameraViews).
    
    Key Properties:
    - instance_id: Unique identifier for this action
    - camera_views: List of camera observations
    - metadata: Optional metadata from info files or other sources
    
    Physical reality does not change with camera angle.
    Multiple videos ≠ multiple actions.
    Multiple videos = multiple observations of the same action.
    """
    instance_id: str
    camera_views: List[CameraView] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_folder: Optional[Path] = None
    
    def add_camera_view(self, view: CameraView) -> None:
        """Add a camera view to this ActionInstance"""
        self.camera_views.append(view)
    
    def get_view_by_id(self, view_id: str) -> Optional[CameraView]:
        """Retrieve a specific camera view by its ID"""
        for view in self.camera_views:
            if view.view_id == view_id:
                return view
        return None
    
    def num_views(self) -> int:
        """Return the number of camera views"""
        return len(self.camera_views)
    
    def __repr__(self) -> str:
        return f"ActionInstance(id={self.instance_id}, views={self.num_views()}, source={self.source_folder})"


class ActionInstanceLoader:
    """
    Loads ActionInstance objects from folder structures.
    
    Folder Structure Assumptions:
    - One folder = one ActionInstance
    - Video files (.mp4, .avi, .mov) = individual camera views
    - Image sequence folders (img/, images/, frames/) = individual camera views
    - Optional info.ini or metadata.json for metadata
    
    No video processing occurs during loading.
    """
    
    SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    IMAGE_SEQUENCE_FOLDER_NAMES = {'img', 'images', 'frames', 'image_sequence'}
    METADATA_FILE_NAMES = {'info.ini', 'metadata.json', 'info.json'}
    
    @staticmethod
    def _detect_media_type(path: Path) -> MediaType:
        """Detect whether a path is a video file or image sequence"""
        if path.is_file():
            if path.suffix.lower() in ActionInstanceLoader.SUPPORTED_VIDEO_EXTENSIONS:
                return MediaType.VIDEO
        elif path.is_dir():
            # Check if directory contains image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            has_images = any(
                f.suffix.lower() in image_extensions 
                for f in path.iterdir() 
                if f.is_file()
            )
            if has_images:
                return MediaType.IMAGE_SEQUENCE
        return MediaType.UNKNOWN
    
    @staticmethod
    def _load_metadata_from_ini(ini_path: Path) -> Dict[str, Any]:
        """Load metadata from an INI file"""
        config = configparser.ConfigParser()
        config.read(ini_path)
        
        metadata = {}
        for section in config.sections():
            metadata[section] = dict(config[section])
        
        return metadata
    
    @staticmethod
    def _load_metadata(folder: Path) -> Dict[str, Any]:
        """Load metadata from any supported metadata file in the folder"""
        for metadata_file in ActionInstanceLoader.METADATA_FILE_NAMES:
            metadata_path = folder / metadata_file
            if metadata_path.exists():
                if metadata_file.endswith('.ini'):
                    return ActionInstanceLoader._load_metadata_from_ini(metadata_path)
                elif metadata_file.endswith('.json'):
                    import json
                    with open(metadata_path, 'r') as f:
                        return json.load(f)
        return {}
    
    @staticmethod
    def load_from_folder(folder_path: Path) -> ActionInstance:
        """
        Load an ActionInstance from a folder.
        
        Args:
            folder_path: Path to the folder containing the action data
            
        Returns:
            ActionInstance object with all detected camera views
            
        Raises:
            FileNotFoundError: If folder does not exist
            ValueError: If no valid camera views are found
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        # Use folder name as instance ID
        instance_id = folder_path.name
        
        # Load metadata if available
        metadata = ActionInstanceLoader._load_metadata(folder_path)
        
        # Create ActionInstance
        action_instance = ActionInstance(
            instance_id=instance_id,
            metadata=metadata,
            source_folder=folder_path
        )
        
        # Scan for video files
        for item in folder_path.iterdir():
            if item.is_file() and item.suffix.lower() in ActionInstanceLoader.SUPPORTED_VIDEO_EXTENSIONS:
                media_type = MediaType.VIDEO
                view_id = item.stem  # Use filename without extension as view ID
                
                camera_view = CameraView(
                    view_id=view_id,
                    media_path=item,
                    media_type=media_type
                )
                action_instance.add_camera_view(camera_view)
        
        # Scan for image sequence folders
        for item in folder_path.iterdir():
            if item.is_dir() and item.name.lower() in ActionInstanceLoader.IMAGE_SEQUENCE_FOLDER_NAMES:
                media_type = ActionInstanceLoader._detect_media_type(item)
                if media_type == MediaType.IMAGE_SEQUENCE:
                    view_id = item.name
                    
                    camera_view = CameraView(
                        view_id=view_id,
                        media_path=item,
                        media_type=media_type
                    )
                    action_instance.add_camera_view(camera_view)
        
        # Validate that at least one camera view was found
        if action_instance.num_views() == 0:
            raise ValueError(f"No valid camera views found in folder: {folder_path}")
        
        return action_instance
    
    @staticmethod
    def load_from_dataset(dataset_path: Path, recursive: bool = False) -> List[ActionInstance]:
        """
        Load multiple ActionInstances from a dataset directory.
        
        Args:
            dataset_path: Path to the dataset root directory
            recursive: If True, recursively search for action folders
            
        Returns:
            List of ActionInstance objects
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        action_instances = []
        
        if recursive:
            # Recursively find all folders that could be ActionInstances
            for folder in dataset_path.rglob('*'):
                if folder.is_dir():
                    try:
                        action_instance = ActionInstanceLoader.load_from_folder(folder)
                        action_instances.append(action_instance)
                    except (ValueError, FileNotFoundError):
                        # Skip folders that don't contain valid ActionInstances
                        continue
        else:
            # Only check immediate subdirectories
            for folder in dataset_path.iterdir():
                if folder.is_dir():
                    try:
                        action_instance = ActionInstanceLoader.load_from_folder(folder)
                        action_instances.append(action_instance)
                    except (ValueError, FileNotFoundError):
                        # Skip folders that don't contain valid ActionInstances
                        continue
        
        return action_instances
