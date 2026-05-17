"""
Batch Processing System for Large Dataset
Handles processing multiple videos from the 350GB+ SportsNet dataset
"""

import os
import sys
from pathlib import Path
import json
import time
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Add 3dsp_utils to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "3dsp_utils"))

# Import our working pipeline functions
from working_server import (
    run_yolo_tracking, run_tracklet_selection, run_2d_pose_estimation,
    run_3d_pose_estimation, run_football_analysis
)

@dataclass
class VideoJob:
    """Represents a video processing job"""
    video_path: str
    output_dir: str
    video_name: str
    status: str = "pending"  # pending, processing, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Optional[Dict] = None

class BatchProcessor:
    """Batch processor for multiple videos"""
    
    def __init__(self, dataset_root: str = "D:/Sportspose"):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path("C:/FootballData/batch_analysis")
        self.jobs: List[VideoJob] = []
        self.current_job: Optional[VideoJob] = None
        self.setup_logging()
        self.setup_directories()
    
    def setup_logging(self):
        """Setup logging for batch processing"""
        log_dir = Path("C:/FootballData/logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"batch_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Setup output directories"""
        directories = [
            self.output_root,
            "C:/FootballData/logs",
            "C:/FootballData/manifests"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def scan_dataset(self, sport_filter: str = "football") -> List[str]:
        """
        Scan the dataset and find all video files
        """
        self.logger.info(f"Scanning dataset at: {self.dataset_root}")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        video_files = []
        
        if not self.dataset_root.exists():
            self.logger.error(f"Dataset root not found: {self.dataset_root}")
            return []
        
        # Scan for video files
        for root, dirs, files in os.walk(self.dataset_root):
            for file in files:
                if Path(file).suffix.lower() in video_extensions:
                    full_path = Path(root) / file
                    
                    # Filter by sport if specified
                    if sport_filter and sport_filter.lower() not in str(full_path).lower():
                        continue
                    
                    video_files.append(str(full_path))
        
        self.logger.info(f"Found {len(video_files)} video files")
        return video_files
    
    def create_dataset_manifest(self, video_files: List[str]) -> str:
        """Create a manifest file of all videos with metadata"""
        manifest = {
            "created_at": datetime.now().isoformat(),
            "dataset_root": str(self.dataset_root),
            "total_videos": len(video_files),
            "videos": []
        }
        
        for video_path in video_files:
            video_info = {
                "path": video_path,
                "name": Path(video_path).name,
                "size_mb": round(Path(video_path).stat().st_size / (1024*1024), 2),
                "relative_path": str(Path(video_path).relative_to(self.dataset_root))
            }
            manifest["videos"].append(video_info)
        
        # Save manifest
        manifest_path = Path("C:/FootballData/manifests") / f"dataset_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Dataset manifest saved to: {manifest_path}")
        return str(manifest_path)
    
    def add_videos_to_queue(self, video_files: List[str], limit: Optional[int] = None):
        """Add videos to processing queue"""
        if limit:
            video_files = video_files[:limit]
        
        for video_path in video_files:
            video_name = Path(video_path).name
            output_dir = self.output_root / Path(video_path).stem
            
            job = VideoJob(
                video_path=video_path,
                output_dir=str(output_dir),
                video_name=video_name
            )
            self.jobs.append(job)
        
        self.logger.info(f"Added {len(video_files)} videos to processing queue")
    
    def process_single_video(self, job: VideoJob) -> bool:
        """Process a single video through the complete pipeline"""
        self.current_job = job
        job.status = "processing"
        job.start_time = datetime.now()
        
        self.logger.info(f"Processing: {job.video_name}")
        
        try:
            # Create output directory
            os.makedirs(job.output_dir, exist_ok=True)
            
            pipeline_results = {}
            
            # Step 1: YOLO Tracking
            self.logger.info(f"Step 1: YOLO tracking for {job.video_name}")
            success, message = run_yolo_tracking(job.video_path, job.output_dir, job.video_name)
            pipeline_results["step_1_yolo"] = {"success": success, "message": message}
            
            if not success:
                raise Exception(f"YOLO tracking failed: {message}")
            
            # Step 2: Tracklet Selection
            self.logger.info(f"Step 2: Tracklet selection for {job.video_name}")
            success, message, shooter_img = run_tracklet_selection(job.video_path, job.output_dir, job.video_name)
            pipeline_results["step_2_tracklets"] = {"success": success, "message": message}
            
            if not success:
                raise Exception(f"Tracklet selection failed: {message}")
            
            # Step 3: 2D Pose Estimation
            self.logger.info(f"Step 3: 2D pose estimation for {job.video_name}")
            success, message = run_2d_pose_estimation(shooter_img, job.output_dir)
            pipeline_results["step_3_2d_pose"] = {"success": success, "message": message}
            
            if not success:
                raise Exception(f"2D pose estimation failed: {message}")
            
            # Step 4: 3D Pose Estimation
            self.logger.info(f"Step 4: 3D pose estimation for {job.video_name}")
            success, message = run_3d_pose_estimation(job.output_dir, job.video_name)
            pipeline_results["step_4_3d_pose"] = {"success": success, "message": message}
            
            if not success:
                raise Exception(f"3D pose estimation failed: {message}")
            
            # Step 5: Football Analysis
            self.logger.info(f"Step 5: Football analysis for {job.video_name}")
            success, message, analysis = run_football_analysis(job.output_dir)
            pipeline_results["step_5_analysis"] = {"success": success, "message": message}
            
            if not success:
                raise Exception(f"Football analysis failed: {message}")
            
            # Save complete results
            job.results = {
                "analysis": analysis,
                "pipeline_results": pipeline_results,
                "processing_time": (datetime.now() - job.start_time).total_seconds()
            }
            
            # Save job results to file
            results_path = Path(job.output_dir) / "batch_processing_results.json"
            with open(results_path, 'w') as f:
                json.dump({
                    "video_name": job.video_name,
                    "video_path": job.video_path,
                    "processed_at": datetime.now().isoformat(),
                    "results": job.results
                }, f, indent=2, default=str)
            
            job.status = "completed"
            job.end_time = datetime.now()
            
            self.logger.info(f"✅ Completed: {job.video_name} in {job.results['processing_time']:.2f}s")
            return True
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.now()
            
            self.logger.error(f"❌ Failed: {job.video_name} - {str(e)}")
            return False
    
    def process_batch(self, max_videos: Optional[int] = None):
        """Process all videos in the queue"""
        if max_videos:
            jobs_to_process = self.jobs[:max_videos]
        else:
            jobs_to_process = self.jobs
        
        self.logger.info(f"Starting batch processing of {len(jobs_to_process)} videos")
        
        start_time = datetime.now()
        completed = 0
        failed = 0
        
        for i, job in enumerate(jobs_to_process, 1):
            self.logger.info(f"Processing video {i}/{len(jobs_to_process)}: {job.video_name}")
            
            if self.process_single_video(job):
                completed += 1
            else:
                failed += 1
            
            # Progress update
            progress = (i / len(jobs_to_process)) * 100
            self.logger.info(f"Progress: {progress:.1f}% ({completed} completed, {failed} failed)")
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Generate batch summary
        summary = {
            "batch_completed_at": datetime.now().isoformat(),
            "total_videos": len(jobs_to_process),
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / len(jobs_to_process)) * 100,
            "total_processing_time": total_time,
            "average_time_per_video": total_time / len(jobs_to_process) if jobs_to_process else 0
        }
        
        # Save batch summary
        summary_path = Path("C:/FootballData/logs") / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Batch processing completed: {completed}/{len(jobs_to_process)} successful")
        self.logger.info(f"Summary saved to: {summary_path}")
        
        return summary
    
    def get_status(self) -> Dict:
        """Get current processing status"""
        total = len(self.jobs)
        completed = len([j for j in self.jobs if j.status == "completed"])
        failed = len([j for j in self.jobs if j.status == "failed"])
        pending = len([j for j in self.jobs if j.status == "pending"])
        processing = len([j for j in self.jobs if j.status == "processing"])
        
        return {
            "total_jobs": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "processing": processing,
            "current_job": self.current_job.video_name if self.current_job else None
        }

def main():
    """Command line interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process videos from SportsNet dataset")
    parser.add_argument("--scan", action="store_true", help="Scan dataset and create manifest")
    parser.add_argument("--process", type=int, help="Process N videos from dataset")
    parser.add_argument("--sport", type=str, default="football", help="Filter by sport type")
    parser.add_argument("--dataset", type=str, default="D:/Sportspose", help="Dataset root directory")
    
    args = parser.parse_args()
    
    processor = BatchProcessor(args.dataset)
    
    if args.scan:
        print("🔍 Scanning dataset...")
        video_files = processor.scan_dataset(args.sport)
        manifest_path = processor.create_dataset_manifest(video_files)
        print(f"📊 Found {len(video_files)} videos")
        print(f"📄 Manifest saved to: {manifest_path}")
    
    if args.process:
        print(f"🚀 Processing {args.process} videos...")
        video_files = processor.scan_dataset(args.sport)
        processor.add_videos_to_queue(video_files, limit=args.process)
        summary = processor.process_batch()
        print(f"✅ Batch completed: {summary['completed']}/{summary['total_videos']} successful")

if __name__ == "__main__":
    main()