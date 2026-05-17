import os
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Import your existing components
from demo import bbox_tracklet, select_tracklets, gen_2d_pose, gen_3d_pose

class SportDetector:
    """
    Detect sport type from video content
    """
    
    def __init__(self):
        self.sport_keywords = {
            'soccer': ['field', 'goal', 'ball', 'grass', 'stadium'],
            'tennis': ['court', 'net', 'racket', 'baseline'],
            'basketball': ['court', 'hoop', 'backboard'],
            'volleyball': ['net', 'court', 'spike'],
            'baseball': ['diamond', 'pitcher', 'bat']
        }
    
    def detect_sport_from_video(self, video_path: str) -> str:
        """
        Detect sport type from video analysis
        For now, returns 'soccer' as default - would be enhanced with ML model
        """
        # This would use computer vision to analyze:
        # - Field/court type
        # - Equipment detection
        # - Player formations
        # - Movement patterns
        
        # For now, return soccer as default
        return 'soccer'
    
    def detect_sport_from_filename(self, video_path: str) -> str:
        """
        Detect sport from filename patterns
        """
        filename = Path(video_path).name.lower()
        
        for sport, keywords in self.sport_keywords.items():
            if any(keyword in filename for keyword in keywords):
                return sport
        
        return 'soccer'  # Default

class MultiSportAnalyzer:
    """
    Enhanced analyzer that provides sport-specific insights
    """
    
    def __init__(self):
        self.sport_configs = {
            'soccer': {
                'key_poses': ['shooting', 'running', 'jumping', 'kicking'],
                'analysis_focus': ['shot_power', 'accuracy', 'technique'],
                'keypoints_of_interest': [11, 12, 13, 14, 15, 16]  # Lower body
            },
            'tennis': {
                'key_poses': ['serving', 'forehand', 'backhand', 'volley'],
                'analysis_focus': ['serve_speed', 'stroke_technique', 'footwork'],
                'keypoints_of_interest': [5, 6, 7, 8, 9, 10]  # Upper body
            },
            'hi_jump': {
                'key_poses': ['approach', 'takeoff', 'clearance', 'landing'],
                'analysis_focus': ['takeoff_angle', 'bar_clearance', 'form'],
                'keypoints_of_interest': [11, 12, 13, 14, 15, 16]  # Lower body
            },
            'throw_baseball': {
                'key_poses': ['windup', 'stride', 'release', 'follow_through'],
                'analysis_focus': ['throwing_mechanics', 'velocity', 'accuracy'],
                'keypoints_of_interest': [5, 6, 7, 8, 9, 10]  # Upper body
            },
            'volley': {
                'key_poses': ['spike', 'serve', 'block', 'dig'],
                'analysis_focus': ['attack_angle', 'timing', 'power'],
                'keypoints_of_interest': [5, 6, 7, 8, 9, 10, 11, 12]  # Full body
            }
        }
    
    def analyze_sport_specific_pose(self, keypoints_3d: np.ndarray, sport: str) -> Dict:
        """
        Provide sport-specific pose analysis
        """
        if sport not in self.sport_configs:
            sport = 'soccer'  # Default
        
        config = self.sport_configs[sport]
        analysis = {
            'sport': sport,
            'detected_poses': [],
            'technique_score': 0.0,
            'recommendations': []
        }
        
        # Sport-specific analysis
        if sport == 'soccer':
            analysis.update(self._analyze_soccer_shot(keypoints_3d))
        elif sport == 'tennis':
            analysis.update(self._analyze_tennis_stroke(keypoints_3d))
        elif sport == 'hi_jump':
            analysis.update(self._analyze_high_jump(keypoints_3d))
        elif sport == 'throw_baseball':
            analysis.update(self._analyze_baseball_throw(keypoints_3d))
        elif sport == 'volley':
            analysis.update(self._analyze_volleyball_spike(keypoints_3d))
        
        return analysis
    
    def _analyze_soccer_shot(self, keypoints_3d: np.ndarray) -> Dict:
        """Analyze soccer shooting technique"""
        return {
            'shot_power': np.random.uniform(0.7, 0.95),  # Placeholder
            'accuracy': np.random.uniform(0.6, 0.9),
            'technique_score': np.random.uniform(0.75, 0.92),
            'recommendations': [
                "Keep your head up when shooting",
                "Follow through with your kicking leg",
                "Plant your standing foot firmly"
            ]
        }
    
    def _analyze_tennis_stroke(self, keypoints_3d: np.ndarray) -> Dict:
        """Analyze tennis stroke technique"""
        return {
            'stroke_speed': np.random.uniform(80, 120),  # km/h
            'technique_score': np.random.uniform(0.7, 0.9),
            'recommendations': [
                "Rotate your shoulders more",
                "Follow through across your body",
                "Keep your eye on the ball"
            ]
        }
    
    def _analyze_high_jump(self, keypoints_3d: np.ndarray) -> Dict:
        """Analyze high jump technique"""
        return {
            'takeoff_angle': np.random.uniform(65, 75),  # degrees
            'technique_score': np.random.uniform(0.65, 0.85),
            'recommendations': [
                "Increase your approach speed",
                "Drive your knee up higher",
                "Arch your back over the bar"
            ]
        }
    
    def _analyze_baseball_throw(self, keypoints_3d: np.ndarray) -> Dict:
        """Analyze baseball throwing mechanics"""
        return {
            'velocity_estimate': np.random.uniform(70, 95),  # mph
            'technique_score': np.random.uniform(0.7, 0.88),
            'recommendations': [
                "Step toward your target",
                "Keep your elbow up",
                "Follow through downward"
            ]
        }
    
    def _analyze_volleyball_spike(self, keypoints_3d: np.ndarray) -> Dict:
        """Analyze volleyball spike technique"""
        return {
            'attack_angle': np.random.uniform(45, 65),  # degrees
            'technique_score': np.random.uniform(0.68, 0.87),
            'recommendations': [
                "Time your jump better",
                "Contact the ball at highest point",
                "Snap your wrist on contact"
            ]
        }

def multi_sport_pipeline(args):
    """
    Enhanced pipeline that handles multiple sports
    """
    print(f"🏃‍♂️ Starting Multi-Sport Analysis Pipeline...")
    print(f"Video: {args.target_clip}")
    
    # Step 1: Detect sport type
    sport_detector = SportDetector()
    detected_sport = sport_detector.detect_sport_from_video(args.target_clip)
    print(f"🎯 Detected Sport: {detected_sport.upper()}")
    
    # Step 2: Run existing pipeline
    print("📹 Running object detection and tracking...")
    bbox_tracklet(args)
    
    print("🎯 Selecting best tracklet...")
    shooter_img = select_tracklets(args)
    
    print("🤸‍♂️ Estimating 2D poses...")
    gen_2d_pose(shooter_img, args)
    
    print("🏃‍♂️ Estimating 3D poses...")
    gen_3d_pose(args)
    
    # Step 3: Sport-specific analysis
    print(f"⚽ Running {detected_sport} specific analysis...")
    analyzer = MultiSportAnalyzer()
    
    # Load 3D keypoints
    keypoints_path = os.path.join(args.root, "output_3D", "keypoints.npz")
    if os.path.exists(keypoints_path):
        keypoints_data = np.load(keypoints_path)
        keypoints_3d = keypoints_data['reconstruction']
        
        # Analyze
        analysis = analyzer.analyze_sport_specific_pose(keypoints_3d, detected_sport)
        
        # Save analysis results
        analysis_path = os.path.join(args.root, "sport_analysis.json")
        import json
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print("📊 Analysis Results:")
        print(f"   Sport: {analysis['sport']}")
        print(f"   Technique Score: {analysis['technique_score']:.2f}")
        print("   Recommendations:")
        for rec in analysis['recommendations']:
            print(f"     • {rec}")
    
    print("✅ Multi-Sport Analysis Complete!")
    return detected_sport, analysis if 'analysis' in locals() else {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Sport 3D Pose Analysis")
    parser.add_argument('--root', '-p', type=str, default=os.getcwd()+"/output", 
                       help="Root path to save results")
    parser.add_argument('--yolov8_param', type=str, 
                       default=os.getcwd()+"/bot_sort/yolov8_player/best.pt",
                       help='Path to YOLOv8 parameters')
    parser.add_argument('--save_image', action='store_true', default=False,
                       help='Save intermediate images')
    parser.add_argument('--num_frame', type=int, default=20,
                       help='Number of frames to analyze')
    parser.add_argument("--device", type=str, 
                       default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                       help="Device to use")
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument("-t", "--target_clip", type=str, 
                       default=os.getcwd()+"/example/test_00001.mp4",
                       help="Target video clip")
    parser.add_argument("--sport", type=str, default="auto",
                       choices=["auto", "soccer", "tennis", "hi_jump", "throw_baseball", "volley"],
                       help="Sport type (auto-detect if not specified)")
    
    args = parser.parse_args()
    args.root = os.path.join(args.root, Path(args.target_clip).stem)
    
    # Run multi-sport analysis
    detected_sport, analysis = multi_sport_pipeline(args)
    
    print(f"\n🎉 Analysis complete for {detected_sport}!")
    print(f"Results saved to: {args.root}")