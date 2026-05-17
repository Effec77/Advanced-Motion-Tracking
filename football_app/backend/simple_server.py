"""
Simple Football Analysis API Server
Integrates directly with your 3dsp_utils pipeline
"""

import os
import sys
from pathlib import Path
import json
import shutil
import argparse

# Add 3dsp_utils to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "3dsp_utils"))

print(f"🔍 Looking for 3dsp_utils at: {project_root / '3dsp_utils'}")

try:
    from football_badminton_setup import FootballBadmintonAnalyzer
    print("✅ Successfully imported FootballBadmintonAnalyzer")
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import FootballBadmintonAnalyzer: {e}")
    ANALYZER_AVAILABLE = False

class FootballAnalysisServer:
    """
    Simple server for football video analysis
    """
    
    def __init__(self):
        self.analyzer = FootballBadmintonAnalyzer() if ANALYZER_AVAILABLE else None
        self.setup_directories()
    
    def setup_directories(self):
        """Setup C drive directories"""
        directories = [
            "C:/FootballData/uploads",
            "C:/FootballData/analysis", 
            "C:/FootballData/temp"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Directory ready: {directory}")
    
    def analyze_video(self, video_path: str, output_name: str = None) -> dict:
        """
        Analyze a football video using your 3dsp_utils pipeline
        """
        if not self.analyzer:
            return {"error": "AI analyzer not available", "status": "failed"}
        
        try:
            print(f"🎥 Analyzing video: {video_path}")
            
            # Create output name if not provided
            if not output_name:
                output_name = Path(video_path).stem
                # Clean the output name to avoid path issues
                output_name = output_name.replace(" ", "_").replace("(", "").replace(")", "")
            
            # Set up arguments for your pipeline
            class Args:
                def __init__(self):
                    self.target_clip = str(Path(video_path).resolve())  # Full path for processing
                    self.root = f"C:/FootballData/analysis/{output_name}"
                    self.yolov8_param = str(project_root / "3dsp_utils/bot_sort/yolov8_player/best.pt")
                    self.save_image = True
                    self.num_frame = 20
                    self.device = "cuda" if os.path.exists("C:/Program Files/NVIDIA GPU Computing Toolkit") else "cpu"
                    self.gpu = "0"
                    # Add the filename for tracking file reference
                    self.video_filename = Path(video_path).name
            
            args = Args()
            
            # Import and run your pipeline
            try:
                # Change to 3dsp_utils directory for proper paths
                original_cwd = os.getcwd()
                os.chdir(project_root / "3dsp_utils")
                
                # Import the functions we need
                from bot_sort.tools.shot_post import sn_demo, make_parser
                from demo import select_tracklets, gen_2d_pose, gen_3d_pose
                
                print("📹 Running object detection and tracking...")
                
                # Create tracking args (copied from bbox_tracklet function)
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
                
                args_track = TrackArgs()
                args_track.mot20 = not args_track.fuse_score
                args.save_result = args.save_image
                os.makedirs(args.root + "/tracking_result", exist_ok=True)
                
                # Use a clean filename for tracking
                clean_filename = f"{output_name}.mp4"
                
                print(f"🎯 Running tracking with file: {clean_filename}")
                sn_demo(args.target_clip, args.yolov8_param, args.root + "/tracking_result", clean_filename, args_track, args)
                
                # Check if tracking results exist using the clean filename
                expected_tracking_file = os.path.join(args.root, "tracking_result", f"{output_name}.txt")
                
                print(f"🔍 Looking for tracking file: {expected_tracking_file}")
                
                # List what files actually exist
                tracking_dir = os.path.join(args.root, "tracking_result")
                if os.path.exists(tracking_dir):
                    existing_files = os.listdir(tracking_dir)
                    print(f"📁 Files in tracking_result: {existing_files}")
                    
                    if not existing_files:
                        os.chdir(original_cwd)
                        return {"error": "No people detected in video. Please ensure the video contains football players.", "status": "failed"}
                else:
                    os.chdir(original_cwd)
                    return {"error": "No people detected in video. Please ensure the video contains football players.", "status": "failed"}
                
                print("🎯 Selecting best athlete...")
                # Fix the path issue by temporarily modifying args.target_clip for select_tracklets
                original_target_clip = args.target_clip
                args.target_clip = clean_filename  # Use the clean filename for path construction
                shooter_img = select_tracklets(args)
                args.target_clip = original_target_clip  # Restore original path
                
                print("🤸‍♂️ Estimating 2D poses...")
                gen_2d_pose(shooter_img, args)
                
                print("🏃‍♂️ Estimating 3D poses...")
                gen_3d_pose(args)
                
                # Run football-specific analysis
                print("⚽ Running football analysis...")
                keypoints_path = os.path.join(args.root, "output_3D", "keypoints.npz")
                
                if os.path.exists(keypoints_path):
                    import numpy as np
                    keypoints_data = np.load(keypoints_path)
                    keypoints_3d = keypoints_data['reconstruction']
                    
                    # Use your football analyzer
                    analysis = self.analyzer.analyze_football_technique(keypoints_3d)
                    
                    # Save analysis results
                    analysis_path = os.path.join(args.root, "football_analysis.json")
                    with open(analysis_path, 'w') as f:
                        json.dump(analysis, f, indent=2, default=str)
                    
                    print("✅ Analysis complete!")
                    
                    # Restore original working directory
                    os.chdir(original_cwd)
                    
                    return {
                        "status": "success",
                        "analysis": analysis,
                        "output_path": args.root,
                        "keypoints_path": keypoints_path,
                        "analysis_file": analysis_path
                    }
                else:
                    os.chdir(original_cwd)
                    return {"error": "3D pose estimation failed", "status": "failed"}
                    
            except ImportError as e:
                os.chdir(original_cwd)
                return {"error": f"Could not import pipeline modules: {e}", "status": "failed"}
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def upload_and_analyze(self, source_video: str) -> dict:
        """
        Upload video to C drive and analyze it
        """
        try:
            # Copy video to uploads folder
            video_name = Path(source_video).name
            upload_path = f"C:/FootballData/uploads/{video_name}"
            
            print(f"📤 Uploading video to: {upload_path}")
            shutil.copy2(source_video, upload_path)
            
            # Analyze the uploaded video
            result = self.analyze_video(upload_path, Path(source_video).stem)
            
            return result
            
        except Exception as e:
            return {"error": f"Upload failed: {e}", "status": "failed"}
    
    def get_analysis_history(self) -> list:
        """
        Get list of previous analyses
        """
        analysis_dir = Path("C:/FootballData/analysis")
        if not analysis_dir.exists():
            return []
        
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
        
        # Sort by timestamp (newest first)
        analyses.sort(key=lambda x: x['timestamp'], reverse=True)
        return analyses

def main():
    """
    Command line interface for the server
    """
    parser = argparse.ArgumentParser(description="Football Analysis Server")
    parser.add_argument("--video", "-v", type=str, help="Video file to analyze")
    parser.add_argument("--history", action="store_true", help="Show analysis history")
    parser.add_argument("--test", action="store_true", help="Test with example video")
    
    args = parser.parse_args()
    
    # Initialize server
    print("⚽ Football Analysis Server Starting...")
    server = FootballAnalysisServer()
    
    if args.history:
        print("📊 Analysis History:")
        history = server.get_analysis_history()
        for i, analysis in enumerate(history[:5], 1):
            print(f"  {i}. {analysis['name']} - Score: {analysis['analysis'].get('technique_score', 'N/A'):.2f}")
    
    elif args.test:
        # Test with example video
        example_video = project_root / "3dsp_utils/example/test_00001.mp4"
        if example_video.exists():
            print(f"🧪 Testing with: {example_video}")
            result = server.upload_and_analyze(str(example_video))
            print(f"📊 Result: {json.dumps(result, indent=2, default=str)}")
        else:
            print("❌ Example video not found")
    
    elif args.video:
        if Path(args.video).exists():
            print(f"🎥 Analyzing: {args.video}")
            result = server.upload_and_analyze(args.video)
            print(f"📊 Result: {json.dumps(result, indent=2, default=str)}")
        else:
            print(f"❌ Video file not found: {args.video}")
    
    else:
        print("🚀 Server ready! Use --help for options")
        print("Examples:")
        print("  python simple_server.py --test")
        print("  python simple_server.py --video path/to/video.mp4")
        print("  python simple_server.py --history")

if __name__ == "__main__":
    main()