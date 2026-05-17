import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

class MultiSportDataset(Dataset):
    """
    Unified dataset loader for multi-sport pose analysis
    Supports: Soccer, Tennis, High Jump, Baseball, Volleyball
    """
    
    def __init__(self, 
                 dataset_path: str = "D:/Sportspose",
                 sports: List[str] = None,
                 environment: str = "both",  # "indoors", "outdoors", "both"
                 mode: str = "train",
                 transform=None):
        
        self.dataset_path = Path(dataset_path)
        self.sports = sports or ['soccer', 'tennis', 'hi_jump', 'throw_baseball', 'volley']
        self.environment = environment
        self.mode = mode
        self.transform = transform
        
        # Sport-specific configurations
        self.sport_config = {
            'soccer': {'fps': 25, 'sequence_length': 20, 'target_size': (100, 100)},
            'tennis': {'fps': 30, 'sequence_length': 15, 'target_size': (96, 96)},
            'hi_jump': {'fps': 60, 'sequence_length': 30, 'target_size': (112, 112)},
            'throw_baseball': {'fps': 30, 'sequence_length': 25, 'target_size': (100, 100)},
            'volley': {'fps': 25, 'sequence_length': 18, 'target_size': (96, 96)}
        }
        
        self.data_samples = self._load_dataset_index()
        
    def _load_dataset_index(self) -> List[Dict]:
        """
        Create an index of all available data samples
        """
        samples = []
        videos_path = self.dataset_path / "videos"
        
        environments = []
        if self.environment == "both":
            environments = ["indoors", "outdoors"]
        else:
            environments = [self.environment]
            
        for env in environments:
            env_path = videos_path / env
            if not env_path.exists():
                continue
                
            # Get all scenarios (S00, S01, etc.)
            scenarios = sorted([d for d in env_path.iterdir() if d.is_dir() and d.name.startswith('S')])
            
            for scenario in scenarios:
                # Get all video folders
                video_folders = sorted([d for d in scenario.iterdir() if d.is_dir() and d.name.startswith('Video_')])
                
                for video_folder in video_folders:
                    sample = {
                        'environment': env,
                        'scenario': scenario.name,
                        'video_folder': video_folder.name,
                        'video_path': video_folder,
                        'sport': self._detect_sport_from_path(video_folder),  # Will implement
                        'data_path': self._get_corresponding_data_path(env, scenario.name, video_folder.name)
                    }
                    samples.append(sample)
        
        print(f"Loaded {len(samples)} samples from {len(environments)} environments")
        return samples
    
    def _detect_sport_from_path(self, video_path: Path) -> str:
        """
        Detect sport type from video path or metadata
        For now, return 'soccer' as default - will enhance based on actual data structure
        """
        # This would be enhanced based on actual dataset structure
        # Could use folder names, metadata files, or video analysis
        return 'soccer'  # Default for now
    
    def _get_corresponding_data_path(self, env: str, scenario: str, video_folder: str) -> Optional[Path]:
        """
        Get the corresponding annotation/data path for a video
        """
        data_path = self.dataset_path / "data" / env / scenario
        if data_path.exists():
            return data_path
        return None
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single data sample
        """
        sample = self.data_samples[idx]
        
        # Load video frames
        video_frames = self._load_video_sequence(sample['video_path'], sample['sport'])
        
        # Load annotations if available
        annotations = self._load_annotations(sample['data_path'], sample['sport'])
        
        # Apply transforms
        if self.transform:
            video_frames = self.transform(video_frames)
        
        return {
            'frames': video_frames,
            'annotations': annotations,
            'sport': sample['sport'],
            'environment': sample['environment'],
            'scenario': sample['scenario'],
            'metadata': sample
        }
    
    def _load_video_sequence(self, video_path: Path, sport: str) -> torch.Tensor:
        """
        Load video sequence based on sport-specific requirements
        """
        config = self.sport_config[sport]
        sequence_length = config['sequence_length']
        target_size = config['target_size']
        
        # Find video files in the folder
        video_files = list(video_path.glob("*.mp4")) + list(video_path.glob("*.avi"))
        
        if not video_files:
            # Return dummy data if no video found
            return torch.zeros(sequence_length, 3, target_size[1], target_size[0])
        
        # Load first video file
        video_file = video_files[0]
        cap = cv2.VideoCapture(str(video_file))
        
        frames = []
        frame_count = 0
        
        while frame_count < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        # Pad sequence if needed
        while len(frames) < sequence_length:
            frames.append(frames[-1] if frames else np.zeros((target_size[1], target_size[0], 3)))
        
        # Convert to tensor (T, H, W, C) -> (T, C, H, W)
        frames = np.array(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        return frames
    
    def _load_annotations(self, data_path: Optional[Path], sport: str) -> Dict:
        """
        Load annotations/poses for the sample
        """
        if data_path is None:
            return {}
        
        # This would load NPY/PKL files based on actual structure
        # For now, return empty dict
        return {}
    
    def get_sport_distribution(self) -> Dict[str, int]:
        """
        Get distribution of sports in the dataset
        """
        sport_counts = {}
        for sample in self.data_samples:
            sport = sample['sport']
            sport_counts[sport] = sport_counts.get(sport, 0) + 1
        return sport_counts

def create_multi_sport_dataloaders(dataset_path: str = "D:/Sportspose",
                                 batch_size: int = 4,
                                 num_workers: int = 2,
                                 train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for multi-sport dataset
    """
    
    # Create full dataset
    full_dataset = MultiSportDataset(dataset_path, environment="both")
    
    # Split into train/val
    total_samples = len(full_dataset)
    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataloader
    print("🏃‍♂️ Testing Multi-Sport DataLoader...")
    
    try:
        dataset = MultiSportDataset()
        print(f"Dataset size: {len(dataset)}")
        print(f"Sport distribution: {dataset.get_sport_distribution()}")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Frames shape: {sample['frames'].shape}")
            print(f"Sport: {sample['sport']}")
            print(f"Environment: {sample['environment']}")
        
        print("✅ DataLoader test successful!")
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        print("This is expected if dataset path is not accessible from workspace")