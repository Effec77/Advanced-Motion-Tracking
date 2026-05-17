"""
Working Football Analysis API Server
Uses the step-by-step approach that we know works
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
import tempfile
from pathlib import Path
import json
import time
import shutil

# Add 3dsp_utils to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "3dsp_utils"))

# Initialize FastAPI app
app = FastAPI(
    title="Football Analysis API (Working)",
    description="Step-by-step football analysis that actually works",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_directories():
    directories = [
        "C:/FootballData/uploads",
        "C:/FootballData/analysis", 
        "C:/FootballData/temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

setup_directories()

def run_yolo_tracking(video_path: str, output_dir: str, video_filename: str) -> tuple[bool, str]:
    """Run YOLO detection and tracking"""
    try:
        original_cwd = os.getcwd()
        os.chdir(project_root / "3dsp_utils")
        
        from bot_sort.tools.shot_post import sn_demo
        
        class Args:
            def __init__(self):
                self.target_clip = video_path
                self.root = output_dir
                self.yolov8_param = str(project_root / "3dsp_utils/bot_sort/yolov8_player/best.pt")
                self.save_image = True
                self.device = "cpu"
                self.gpu = "0"
                self.num_frame = 20
        
        class TrackArgs:
            def __init__(self):
                self.ablation = False
                self.fuse_score = False
                self.track_high_thresh = 0.6
                self.track_low_thresh = 0.1
                self.new_track_thresh = 0.7
                self.track_buffer = 30
                self.match_thresh = 0.8
                self.aspect_ratio_thresh = 1.6
                self.min_box_area = 10
                self.cmc_method = "sparseOptFlow"
                self.with_reid = False
                self.proximity_thresh = 0.5
                self.appearance_thresh = 0.25
                self.device = "gpu"
                self.fp16 = False
                self.fuse = False
                self.fps = 25
                self.name = "demo"
                self.save_result = True
        
        args = Args()
        args_track = TrackArgs()
        args_track.mot20 = not args_track.fuse_score
        args.save_result = args.save_image
        
        os.makedirs(args.root + "/tracking_result", exist_ok=True)
        
        sn_demo(args.target_clip, args.yolov8_param, args.root + "/tracking_result", video_filename, args_track, args)
        
        # Check if tracking file was created
        tracking_file = os.path.join(args.root, "tracking_result", f"{Path(video_filename).stem}.txt")
        
        os.chdir(original_cwd)
        
        if os.path.exists(tracking_file):
            return True, "YOLO tracking completed successfully"
        else:
            return False, "No tracking results generated"
            
    except Exception as e:
        os.chdir(original_cwd)
        return False, f"YOLO tracking failed: {str(e)}"

def run_tracklet_selection(video_path: str, output_dir: str, video_filename: str) -> tuple[bool, str, list]:
    """Run tracklet selection"""
    try:
        original_cwd = os.getcwd()
        os.chdir(project_root / "3dsp_utils")
        
        from demo import select_tracklets
        
        class Args:
            def __init__(self):
                self.target_clip = video_path
                self.root = output_dir
                self.save_image = True
                self.device = "cpu"
                self.video_filename = video_filename
                self.num_frame = 20
        
        args = Args()
        
        shooter_img = select_tracklets(args)
        
        os.chdir(original_cwd)
        
        return True, f"Tracklet selection completed - {len(shooter_img)} images", shooter_img
        
    except Exception as e:
        os.chdir(original_cwd)
        return False, f"Tracklet selection failed: {str(e)}", []

def run_2d_pose_estimation(shooter_img: list, output_dir: str) -> tuple[bool, str]:
    """Run 2D pose estimation"""
    try:
        original_cwd = os.getcwd()
        os.chdir(project_root / "3dsp_utils")
        
        from demo import gen_2d_pose
        
        class Args:
            def __init__(self):
                self.root = output_dir
                self.num_frame = 20
        
        args = Args()
        
        gen_2d_pose(shooter_img, args)
        
        # Check if 2D keypoints were generated
        keypoints_2d_file = os.path.join(args.root, "input_2D", "keypoints.npz")
        
        os.chdir(original_cwd)
        
        if os.path.exists(keypoints_2d_file):
            return True, "2D pose estimation completed"
        else:
            return False, "2D pose estimation failed - no keypoints generated"
            
    except Exception as e:
        os.chdir(original_cwd)
        return False, f"2D pose estimation failed: {str(e)}"

def run_3d_pose_estimation(output_dir: str, video_filename: str) -> tuple[bool, str]:
    """Run 3D pose estimation"""
    try:
        original_cwd = os.getcwd()
        os.chdir(project_root / "3dsp_utils")
        
        from MotionAGFormer.demo.vis_sn import get_pose3D_demo, img2video_demo
        
        class Args:
            def __init__(self):
                self.root = output_dir
                self.video_length = 20
                self.gpu = "0"
        
        args = Args()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        
        get_pose3D_demo(args.root + "/shooter_image", args.root, args.video_length)
        img2video_demo(args.root + "/shooter_image", args.root, Path(video_filename).stem)
        
        # Check if 3D keypoints were generated
        keypoints_3d_file = os.path.join(args.root, "output_3D", "keypoints.npz")
        
        os.chdir(original_cwd)
        
        if os.path.exists(keypoints_3d_file):
            return True, "3D pose estimation completed"
        else:
            return False, "3D pose estimation failed - no keypoints generated"
            
    except Exception as e:
        os.chdir(original_cwd)
        return False, f"3D pose estimation failed: {str(e)}"

def run_football_analysis(output_dir: str) -> tuple[bool, str, dict]:
    """Run football technique analysis"""
    try:
        from football_badminton_setup import FootballBadmintonAnalyzer
        
        analyzer = FootballBadmintonAnalyzer()
        
        keypoints_path = os.path.join(output_dir, "output_3D", "keypoints.npz")
        
        if not os.path.exists(keypoints_path):
            return False, "No 3D keypoints found for analysis", {}
        
        import numpy as np
        keypoints_data = np.load(keypoints_path)
        keypoints_3d = keypoints_data['reconstruction']
        
        analysis = analyzer.analyze_football_technique(keypoints_3d)
        
        # Save analysis results
        analysis_path = os.path.join(output_dir, "football_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return True, "Football analysis completed", analysis
        
    except Exception as e:
        return False, f"Football analysis failed: {str(e)}", {}

@app.get("/")
async def root():
    return {
        "message": "Football Analysis API (Working Version)",
        "version": "1.0.0",
        "status": "ready",
        "pipeline": "step-by-step verified working"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline": "working",
        "storage": "C:/FootballData ready"
    }

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """
    Analyze uploaded football video using step-by-step pipeline
    """
    try:
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        print(f"🎥 Starting analysis for: {video.filename}")
        
        # Save uploaded video
        video_name = video.filename
        upload_path = f"C:/FootballData/uploads/{video_name}"
        
        with open(upload_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        # Create analysis directory
        analysis_name = Path(video_name).stem
        analysis_dir = f"C:/FootballData/analysis/{analysis_name}"
        os.makedirs(analysis_dir, exist_ok=True)
        
        pipeline_results = {}
        
        # Step 1: YOLO Tracking
        print("📹 Step 1: Running YOLO detection and tracking...")
        success, message = run_yolo_tracking(upload_path, analysis_dir, video_name)
        pipeline_results["step_1_yolo"] = {"success": success, "message": message}
        
        if not success:
            raise HTTPException(status_code=500, detail=f"YOLO tracking failed: {message}")
        
        # Step 2: Tracklet Selection
        print("🎯 Step 2: Selecting best tracklet...")
        success, message, shooter_img = run_tracklet_selection(upload_path, analysis_dir, video_name)
        pipeline_results["step_2_tracklets"] = {"success": success, "message": message}
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Tracklet selection failed: {message}")
        
        # Step 3: 2D Pose Estimation
        print("🤸‍♂️ Step 3: Estimating 2D poses...")
        success, message = run_2d_pose_estimation(shooter_img, analysis_dir)
        pipeline_results["step_3_2d_pose"] = {"success": success, "message": message}
        
        if not success:
            raise HTTPException(status_code=500, detail=f"2D pose estimation failed: {message}")
        
        # Step 4: 3D Pose Estimation
        print("🏃‍♂️ Step 4: Estimating 3D poses...")
        success, message = run_3d_pose_estimation(analysis_dir, video_name)
        pipeline_results["step_4_3d_pose"] = {"success": success, "message": message}
        
        if not success:
            raise HTTPException(status_code=500, detail=f"3D pose estimation failed: {message}")
        
        # Step 5: Football Analysis
        print("⚽ Step 5: Analyzing football technique...")
        success, message, analysis = run_football_analysis(analysis_dir)
        pipeline_results["step_5_analysis"] = {"success": success, "message": message}
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Football analysis failed: {message}")
        
        print("✅ Analysis completed successfully!")
        
        return JSONResponse(content={
            "status": "success",
            "video_name": video_name,
            "analysis": analysis,
            "output_path": analysis_dir,
            "pipeline_results": pipeline_results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/history")
async def get_analysis_history():
    """Get analysis history"""
    try:
        analysis_dir = Path("C:/FootballData/analysis")
        if not analysis_dir.exists():
            return {"status": "success", "count": 0, "analyses": []}
        
        analyses = []
        for folder in analysis_dir.iterdir():
            if folder.is_dir():
                analysis_file = folder / "football_analysis.json"
                if analysis_file.exists():
                    try:
                        with open(analysis_file, 'r') as f:
                            analysis = json.load(f)
                        analyses.append({
                            "name": folder.name,
                            "path": str(folder),
                            "analysis": analysis,
                            "timestamp": folder.stat().st_mtime
                        })
                    except:
                        pass
        
        analyses.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            "status": "success",
            "count": len(analyses),
            "analyses": analyses[:10]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

# Import batch processor for dataset operations
from batch_processor import BatchProcessor

# Global batch processor instance
batch_processor = BatchProcessor()

@app.post("/batch/scan-dataset")
async def scan_dataset(sport_filter: str = "football"):
    """Scan the 350GB dataset and create manifest"""
    try:
        video_files = batch_processor.scan_dataset(sport_filter)
        manifest_path = batch_processor.create_dataset_manifest(video_files)
        
        return {
            "status": "success",
            "total_videos": len(video_files),
            "manifest_path": manifest_path,
            "sport_filter": sport_filter
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset scan failed: {str(e)}")

@app.post("/batch/start-processing")
async def start_batch_processing(max_videos: int = 10, sport_filter: str = "football"):
    """Start batch processing of dataset videos"""
    try:
        # Scan and queue videos
        video_files = batch_processor.scan_dataset(sport_filter)
        batch_processor.add_videos_to_queue(video_files, limit=max_videos)
        
        # Start processing in background
        import threading
        
        def process_in_background():
            batch_processor.process_batch(max_videos)
        
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return {
            "status": "started",
            "queued_videos": max_videos,
            "total_available": len(video_files),
            "message": f"Processing {max_videos} videos in background"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/batch/status")
async def get_batch_status():
    """Get current batch processing status"""
    try:
        status = batch_processor.get_status()
        return {
            "status": "success",
            "batch_status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/dataset/info")
async def get_dataset_info():
    """Get information about the dataset"""
    try:
        dataset_root = Path("D:/Sportspose")
        
        if not dataset_root.exists():
            return {
                "status": "error",
                "message": "Dataset not found at D:/Sportspose"
            }
        
        # Quick scan for basic info
        total_size = 0
        video_count = 0
        
        for root, dirs, files in os.walk(dataset_root):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
                    video_count += 1
                    total_size += file_path.stat().st_size
        
        return {
            "status": "success",
            "dataset_root": str(dataset_root),
            "total_videos": video_count,
            "total_size_gb": round(total_size / (1024**3), 2),
            "available": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Dataset info failed: {str(e)}"
        }

if __name__ == "__main__":
    print("🚀 Starting Working Football Analysis API...")
    print("📊 API Documentation: http://localhost:8003/docs")
    print("🎯 Health Check: http://localhost:8003/health")
    print("⚽ Ready for football video analysis!")
    
    uvicorn.run(app, host="0.0.0.0", port=8003)