"""
Debug Pipeline - Test Each Step Individually
"""

import os
import sys
from pathlib import Path
import json
import time

# Add 3dsp_utils to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "3dsp_utils"))

def test_step_1_yolo_tracking():
    """Test YOLO detection and tracking"""
    print("🔍 Testing Step 1: YOLO Detection & Tracking...")
    
    try:
        # Change to 3dsp_utils directory
        original_cwd = os.getcwd()
        os.chdir(project_root / "3dsp_utils")
        
        from bot_sort.tools.shot_post import sn_demo
        
        # Set up test arguments
        class Args:
            def __init__(self):
                self.target_clip = str(project_root / "3dsp_utils/example/test_00001.mp4")
                self.root = "C:/FootballData/debug_test"
                self.yolov8_param = str(project_root / "3dsp_utils/bot_sort/yolov8_player/best.pt")
                self.save_image = True
                self.device = "cpu"  # Force CPU to avoid GPU issues
                self.gpu = "0"
                self.num_frame = 20  # Add missing attribute
        
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
        
        # Create output directory
        os.makedirs(args.root + "/tracking_result", exist_ok=True)
        
        print(f"📹 Input video: {args.target_clip}")
        print(f"📁 Output dir: {args.root}")
        
        # Run with timeout
        start_time = time.time()
        
        # This is where it might get stuck - let's add a timeout
        try:
            sn_demo(args.target_clip, args.yolov8_param, args.root + "/tracking_result", "test_00001.mp4", args_track, args)
            elapsed = time.time() - start_time
            print(f"✅ Step 1 completed in {elapsed:.2f} seconds")
            
            # Check results
            tracking_files = os.listdir(args.root + "/tracking_result")
            print(f"📁 Generated files: {tracking_files}")
            
            os.chdir(original_cwd)
            return True, f"Success - generated {len(tracking_files)} files"
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Step 1 failed after {elapsed:.2f} seconds: {e}")
            os.chdir(original_cwd)
            return False, str(e)
            
    except Exception as e:
        print(f"❌ Step 1 setup failed: {e}")
        return False, str(e)

def test_step_2_tracklet_selection():
    """Test tracklet selection (this is where it usually gets stuck)"""
    print("🎯 Testing Step 2: Tracklet Selection...")
    
    try:
        # First ensure we have tracking results
        tracking_file = "C:/FootballData/debug_test/tracking_result/test_00001.txt"
        if not os.path.exists(tracking_file):
            return False, "No tracking results from Step 1"
        
        original_cwd = os.getcwd()
        os.chdir(project_root / "3dsp_utils")
        
        from demo import select_tracklets
        
        class Args:
            def __init__(self):
                self.target_clip = str(project_root / "3dsp_utils/example/test_00001.mp4")
                self.root = "C:/FootballData/debug_test"
                self.save_image = True
                self.device = "cpu"  # Force CPU
                self.video_filename = "test_00001.mp4"
                self.num_frame = 20  # Add missing attribute
        
        args = Args()
        
        start_time = time.time()
        
        # This is likely where it gets stuck
        shooter_img = select_tracklets(args)
        
        elapsed = time.time() - start_time
        print(f"✅ Step 2 completed in {elapsed:.2f} seconds")
        print(f"📊 Generated {len(shooter_img)} shooter images")
        
        os.chdir(original_cwd)
        return True, f"Success - {len(shooter_img)} images"
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Step 2 failed after {elapsed:.2f} seconds: {e}")
        os.chdir(original_cwd)
        return False, str(e)

def run_debug_pipeline():
    """Run the debug pipeline step by step"""
    print("🚀 Starting Debug Pipeline...")
    
    results = {}
    
    # Step 1: YOLO Tracking
    success, message = test_step_1_yolo_tracking()
    results["step_1_yolo"] = {"success": success, "message": message}
    
    if not success:
        print("❌ Stopping at Step 1 - YOLO failed")
        return results
    
    # Step 2: Tracklet Selection
    success, message = test_step_2_tracklet_selection()
    results["step_2_tracklets"] = {"success": success, "message": message}
    
    if not success:
        print("❌ Stopping at Step 2 - Tracklet selection failed")
        return results
    
    print("✅ All steps completed successfully!")
    return results

if __name__ == "__main__":
    results = run_debug_pipeline()
    print("\n📊 Debug Results:")
    for step, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {step}: {result['message']}")