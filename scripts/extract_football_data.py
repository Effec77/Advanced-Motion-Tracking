"""
Football Data Extraction Script
Extracts football/soccer data from your 298GB SportsNet dataset
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List
import argparse

class FootballDataExtractor:
    """
    Extract and organize football data from the SportsNet dataset
    """
    
    def __init__(self, source_path: str = "D:/Sportspose", output_path: str = "./football_dataset"):
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.stats = {
            'videos_found': 0,
            'videos_copied': 0,
            'annotations_found': 0,
            'annotations_copied': 0,
            'scenarios_processed': 0
        }
    
    def create_output_structure(self):
        """Create organized output directory structure"""
        print("📁 Creating output directory structure...")
        
        # Main directories
        (self.output_path / "videos" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_path / "videos" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_path / "videos" / "test").mkdir(parents=True, exist_ok=True)
        
        (self.output_path / "annotations" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_path / "annotations" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_path / "annotations" / "test").mkdir(parents=True, exist_ok=True)
        
        (self.output_path / "metadata").mkdir(exist_ok=True)
        
        print(f"✅ Output structure created at: {self.output_path}")
    
    def identify_football_data(self) -> Dict[str, List[Path]]:
        """
        Identify football/soccer related data in the dataset
        """
        print("🔍 Scanning dataset for football data...")
        
        football_data = {
            'indoor_videos': [],
            'outdoor_videos': [],
            'indoor_annotations': [],
            'outdoor_annotations': []
        }
        
        # Scan videos
        videos_path = self.source_path / "videos"
        if videos_path.exists():
            # Indoor videos
            indoor_path = videos_path / "indoors"
            if indoor_path.exists():
                for scenario in indoor_path.glob("S*"):
                    if scenario.is_dir():
                        video_folders = list(scenario.glob("Video_*"))
                        football_data['indoor_videos'].extend(video_folders)
                        self.stats['scenarios_processed'] += 1
            
            # Outdoor videos  
            outdoor_path = videos_path / "outdoors"
            if outdoor_path.exists():
                for scenario in outdoor_path.glob("S*"):
                    if scenario.is_dir():
                        video_folders = list(scenario.glob("Video_*"))
                        football_data['outdoor_videos'].extend(video_folders)
                        self.stats['scenarios_processed'] += 1
        
        # Scan annotations (if data folder is extracted)
        data_path = self.source_path / "data"
        if data_path.exists():
            # Indoor annotations
            indoor_data = data_path / "indoors"
            if indoor_data.exists():
                for scenario in indoor_data.glob("S*"):
                    soccer_path = scenario / "soccer"
                    if soccer_path.exists():
                        football_data['indoor_annotations'].append(soccer_path)
            
            # Outdoor annotations
            outdoor_data = data_path / "outdoors" 
            if outdoor_data.exists():
                for scenario in outdoor_data.glob("S*"):
                    soccer_path = scenario / "soccer"
                    if soccer_path.exists():
                        football_data['outdoor_annotations'].append(soccer_path)
        
        # Update stats
        self.stats['videos_found'] = len(football_data['indoor_videos']) + len(football_data['outdoor_videos'])
        self.stats['annotations_found'] = len(football_data['indoor_annotations']) + len(football_data['outdoor_annotations'])
        
        print(f"📊 Found {self.stats['videos_found']} video folders")
        print(f"📊 Found {self.stats['annotations_found']} annotation folders")
        print(f"📊 Processed {self.stats['scenarios_processed']} scenarios")
        
        return football_data
    
    def copy_sample_data(self, football_data: Dict, sample_size: int = 50):
        """
        Copy a sample of football data for development
        """
        print(f"📋 Copying sample data (max {sample_size} videos per split)...")
        
        # Combine all video folders
        all_videos = football_data['indoor_videos'] + football_data['outdoor_videos']
        
        if len(all_videos) == 0:
            print("❌ No video data found!")
            return
        
        # Create train/val/test splits (70/20/10)
        total_videos = min(len(all_videos), sample_size * 3)  # Limit total for development
        train_size = int(total_videos * 0.7)
        val_size = int(total_videos * 0.2)
        test_size = total_videos - train_size - val_size
        
        splits = {
            'train': all_videos[:train_size],
            'val': all_videos[train_size:train_size + val_size],
            'test': all_videos[train_size + val_size:train_size + val_size + test_size]
        }
        
        # Copy videos for each split
        for split_name, video_folders in splits.items():
            print(f"  📁 Processing {split_name} split ({len(video_folders)} videos)...")
            
            for i, video_folder in enumerate(video_folders):
                try:
                    # Create destination folder
                    dest_folder = self.output_path / "videos" / split_name / f"video_{i:04d}"
                    dest_folder.mkdir(exist_ok=True)
                    
                    # Copy video files
                    video_files = list(video_folder.glob("*.mp4")) + list(video_folder.glob("*.avi"))
                    for video_file in video_files[:1]:  # Copy first video file only
                        dest_file = dest_folder / video_file.name
                        if not dest_file.exists():
                            shutil.copy2(video_file, dest_file)
                            self.stats['videos_copied'] += 1
                
                except Exception as e:
                    print(f"    ⚠️ Error copying {video_folder}: {e}")
        
        print(f"✅ Copied {self.stats['videos_copied']} video files")
    
    def copy_annotations(self, football_data: Dict):
        """
        Copy football annotation data
        """
        if not football_data['indoor_annotations'] and not football_data['outdoor_annotations']:
            print("ℹ️ No annotation data found (data folder may be compressed)")
            return
        
        print("📋 Copying annotation data...")
        
        all_annotations = football_data['indoor_annotations'] + football_data['outdoor_annotations']
        
        for i, annotation_folder in enumerate(all_annotations):
            try:
                dest_folder = self.output_path / "annotations" / "train" / f"scenario_{i:04d}"
                if annotation_folder.exists():
                    shutil.copytree(annotation_folder, dest_folder, dirs_exist_ok=True)
                    self.stats['annotations_copied'] += 1
            except Exception as e:
                print(f"    ⚠️ Error copying annotations {annotation_folder}: {e}")
        
        print(f"✅ Copied {self.stats['annotations_copied']} annotation folders")
    
    def create_metadata(self, football_data: Dict):
        """
        Create metadata files for the extracted dataset
        """
        print("📝 Creating metadata files...")
        
        metadata = {
            'dataset_info': {
                'name': 'Football Analysis Dataset',
                'source': 'SportsNet 298GB Dataset',
                'extraction_date': str(Path().cwd()),
                'total_scenarios': self.stats['scenarios_processed'],
                'total_videos': self.stats['videos_found'],
                'copied_videos': self.stats['videos_copied']
            },
            'splits': {
                'train': {'videos': len(list((self.output_path / "videos" / "train").glob("*")))},
                'val': {'videos': len(list((self.output_path / "videos" / "val").glob("*")))},
                'test': {'videos': len(list((self.output_path / "videos" / "test").glob("*")))}
            },
            'data_structure': {
                'videos': 'MP4/AVI files organized by train/val/test splits',
                'annotations': 'NPY/PKL files with pose and metadata',
                'format': 'Each video folder contains one representative video file'
            }
        }
        
        # Save metadata
        metadata_file = self.output_path / "metadata" / "dataset_info.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create README
        readme_content = f"""# Football Analysis Dataset

## Overview
Extracted from SportsNet 298GB multi-sport dataset for football-specific analysis.

## Statistics
- **Total Videos Found**: {self.stats['videos_found']}
- **Videos Copied**: {self.stats['videos_copied']}
- **Scenarios Processed**: {self.stats['scenarios_processed']}
- **Annotations**: {self.stats['annotations_copied']} folders

## Structure
```
football_dataset/
├── videos/
│   ├── train/          # Training videos
│   ├── val/            # Validation videos
│   └── test/           # Test videos
├── annotations/        # Pose annotations (if available)
├── metadata/           # Dataset information
└── README.md          # This file
```

## Usage
Use this dataset to train and improve football-specific models:
- YOLOv8 player detection
- Tracklet selection
- 3D pose estimation
- Technique analysis

## Next Steps
1. Verify data quality
2. Create training scripts
3. Enhance models with this data
4. Build football analysis pipeline
"""
        
        readme_file = self.output_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Metadata saved to: {metadata_file}")
        print(f"✅ README created: {readme_file}")
    
    def extract_football_dataset(self, sample_size: int = 50):
        """
        Main extraction process
        """
        print("⚽ Starting Football Dataset Extraction...")
        print(f"Source: {self.source_path}")
        print(f"Output: {self.output_path}")
        
        # Create output structure
        self.create_output_structure()
        
        # Identify football data
        football_data = self.identify_football_data()
        
        if self.stats['videos_found'] == 0:
            print("❌ No football data found! Check your dataset path.")
            return
        
        # Copy sample data
        self.copy_sample_data(football_data, sample_size)
        
        # Copy annotations (if available)
        self.copy_annotations(football_data)
        
        # Create metadata
        self.create_metadata(football_data)
        
        # Final summary
        print("\n🎉 Football Dataset Extraction Complete!")
        print(f"📊 Final Statistics:")
        print(f"   Videos Found: {self.stats['videos_found']}")
        print(f"   Videos Copied: {self.stats['videos_copied']}")
        print(f"   Annotations: {self.stats['annotations_copied']}")
        print(f"   Output Location: {self.output_path.absolute()}")
        
        return self.output_path

def main():
    parser = argparse.ArgumentParser(description="Extract football data from SportsNet dataset")
    parser.add_argument("--source", type=str, default="D:/Sportspose", 
                       help="Path to SportsNet dataset")
    parser.add_argument("--output", type=str, default="./football_dataset",
                       help="Output directory for football data")
    parser.add_argument("--sample_size", type=int, default=50,
                       help="Number of videos per split (train/val/test)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only scan and report, don't copy files")
    
    args = parser.parse_args()
    
    extractor = FootballDataExtractor(args.source, args.output)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - Scanning only...")
        football_data = extractor.identify_football_data()
        print(f"Would extract {extractor.stats['videos_found']} videos")
    else:
        extractor.extract_football_dataset(args.sample_size)

if __name__ == "__main__":
    main()