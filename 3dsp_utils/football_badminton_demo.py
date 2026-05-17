"""
Football + Badminton Specialized Demo
"""

import os
import sys
import numpy as np
sys.path.append('.')
from demo import bbox_tracklet, select_tracklets, gen_2d_pose, gen_3d_pose
from football_badminton_setup import FootballBadmintonAnalyzer
import json
import argparse
import torch
from pathlib import Path

def football_badminton_pipeline(args):
    """
    Specialized pipeline for football and badminton analysis
    """
    print("⚽🏸 Football + Badminton Analysis Pipeline")
    print(f"Video: {args.target_clip}")
    
    # Initialize analyzer
    analyzer = FootballBadmintonAnalyzer()
    
    # Detect sport (football or badminton)
    detected_sport = analyzer.detect_sport_from_video(args.target_clip)
    print(f"🎯 Detected Sport: {detected_sport.upper()}")
    
    # Get sport-specific config
    config = analyzer.get_sport_specific_config(detected_sport)
    
    # Adjust parameters based on sport
    args.num_frame = config['sequence_length']
    
    print("📹 Running detection and tracking...")
    bbox_tracklet(args)
    
    print("🎯 Selecting best athlete...")
    shooter_img = select_tracklets(args)
    
    print("🤸‍♂️ Estimating 2D poses...")
    gen_2d_pose(shooter_img, args)
    
    print("🏃‍♂️ Estimating 3D poses...")
    gen_3d_pose(args)
    
    # Sport-specific analysis
    print(f"⚽🏸 Running {detected_sport} analysis...")
    
    # Load 3D keypoints
    keypoints_path = os.path.join(args.root, "output_3D", "keypoints.npz")
    analysis = {}
    
    if os.path.exists(keypoints_path):
        keypoints_data = np.load(keypoints_path)
        keypoints_3d = keypoints_data['reconstruction']
        
        # Analyze based on detected sport
        if detected_sport == 'football':
            analysis = analyzer.analyze_football_technique(keypoints_3d)
        else:  # badminton
            analysis = analyzer.analyze_badminton_technique(keypoints_3d)
        
        # Save results
        analysis_path = os.path.join(args.root, f"{detected_sport}_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Display results
        print("\n📊 Analysis Results:")
        print(f"   Sport: {analysis['sport'].upper()}")
        print(f"   Technique Score: {analysis['technique_score']:.2f}")
        
        if detected_sport == 'football':
            print(f"   Shot Power: {analysis['shot_power']:.2f}")
            print(f"   Accuracy: {analysis['accuracy']:.2f}")
            print(f"   Ball Control: {analysis['ball_control']:.2f}")
        else:  # badminton
            print(f"   Racket Speed: {analysis['racket_speed']:.1f} km/h")
            print(f"   Timing: {analysis['timing']:.2f}")
            print(f"   Footwork: {analysis['footwork']:.2f}")
            print(f"   Power: {analysis['power']:.2f}")
        
        print("   Top Recommendations:")
        for i, rec in enumerate(analysis['recommendations'][:3], 1):
            print(f"     {i}. {rec}")
    
    print("✅ Analysis Complete!")
    return detected_sport, analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football + Badminton Analysis")
    parser.add_argument('--root', '-p', type=str, default=os.getcwd()+"/output", 
                       help="Root path to save results")
    parser.add_argument('--yolov8_param', type=str, 
                       default=os.getcwd()+"/bot_sort/yolov8_player/best.pt",
                       help="Path to YOLOv8 model")
    parser.add_argument('--save_image', action='store_true', default=False,
                       help="Save intermediate images")
    parser.add_argument('--num_frame', type=int, default=20,
                       help="Number of frames (will be adjusted per sport)")
    parser.add_argument("--device", type=str, 
                       default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                       help="Device to use")
    parser.add_argument('--gpu', type=str, default='0', help="GPU ID")
    parser.add_argument("-t", "--target_clip", type=str, 
                       default=os.getcwd()+"/example/test_00001.mp4",
                       help="Target video file")
    parser.add_argument("--sport", type=str, default="auto",
                       choices=["auto", "football", "badminton"],
                       help="Force specific sport (auto-detect if not specified)")
    
    args = parser.parse_args()
    args.root = os.path.join(args.root, Path(args.target_clip).stem)
    
    # Run analysis
    sport, analysis = football_badminton_pipeline(args)
    
    print(f"\n🎉 {sport.upper()} analysis complete!")
    print(f"Results saved to: {args.root}")
    
    if analysis:
        print(f"\n🏆 Final Score: {analysis['technique_score']:.1%}")
        if sport == 'football':
            print("⚽ Focus areas: Shot power, Accuracy, Ball control")
        else:
            print("🏸 Focus areas: Racket technique, Timing, Footwork")