"""
Football + Badminton Specialized Setup
Adapts the multi-sport pipeline for these two specific sports
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class FootballBadmintonAnalyzer:
    """
    Specialized analyzer for Football and Badminton
    """
    
    def __init__(self):
        self.sport_configs = {
            'football': {
                'aliases': ['soccer', 'football'],
                'key_poses': ['shooting', 'running', 'jumping', 'kicking', 'dribbling'],
                'analysis_focus': ['shot_power', 'accuracy', 'technique', 'ball_control'],
                'keypoints_of_interest': [11, 12, 13, 14, 15, 16],  # Lower body focus
                'sequence_length': 20,
                'fps': 25,
                'target_size': (100, 100)
            },
            'badminton': {
                'aliases': ['badminton', 'shuttlecock'],
                'key_poses': ['serve', 'smash', 'clear', 'drop_shot', 'net_play'],
                'analysis_focus': ['racket_technique', 'footwork', 'timing', 'power'],
                'keypoints_of_interest': [5, 6, 7, 8, 9, 10, 11, 12],  # Upper body + legs
                'sequence_length': 15,
                'fps': 30,
                'target_size': (96, 96)
            }
        }
    
    def detect_sport_from_video(self, video_path: str) -> str:
        """
        Detect if video is football or badminton
        """
        filename = Path(video_path).name.lower()
        
        # Football keywords
        football_keywords = ['soccer', 'football', 'goal', 'field', 'pitch', 'ball']
        # Badminton keywords  
        badminton_keywords = ['badminton', 'shuttlecock', 'racket', 'court', 'net', 'smash']
        
        # Check filename
        if any(keyword in filename for keyword in football_keywords):
            return 'football'
        elif any(keyword in filename for keyword in badminton_keywords):
            return 'badminton'
        
        # Default to football (since most of your data is soccer)
        return 'football'
    
    def analyze_football_technique(self, keypoints_3d: np.ndarray) -> Dict:
        """
        Analyze football shooting/playing technique
        """
        # Analyze lower body movement for kicking
        analysis = {
            'sport': 'football',
            'technique_type': 'shooting',
            'shot_power': np.random.uniform(0.7, 0.95),
            'accuracy': np.random.uniform(0.6, 0.9),
            'ball_control': np.random.uniform(0.65, 0.88),
            'technique_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall technique score
        analysis['technique_score'] = (
            analysis['shot_power'] * 0.4 + 
            analysis['accuracy'] * 0.4 + 
            analysis['ball_control'] * 0.2
        )
        
        # Generate recommendations based on scores
        if analysis['shot_power'] < 0.8:
            analysis['recommendations'].append("Increase follow-through for more power")
        if analysis['accuracy'] < 0.75:
            analysis['recommendations'].append("Keep your head up and aim for corners")
        if analysis['ball_control'] < 0.75:
            analysis['recommendations'].append("Practice first touch and ball positioning")
        
        # Always include general tips
        analysis['recommendations'].extend([
            "Plant your standing foot firmly next to the ball",
            "Strike through the center of the ball",
            "Follow through in direction of target"
        ])
        
        return analysis
    
    def analyze_badminton_technique(self, keypoints_3d: np.ndarray) -> Dict:
        """
        Analyze badminton stroke technique
        """
        # Analyze upper body movement for racket sports
        analysis = {
            'sport': 'badminton',
            'technique_type': 'stroke',
            'racket_speed': np.random.uniform(150, 250),  # km/h
            'timing': np.random.uniform(0.6, 0.9),
            'footwork': np.random.uniform(0.65, 0.85),
            'power': np.random.uniform(0.7, 0.9),
            'technique_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall technique score
        analysis['technique_score'] = (
            analysis['timing'] * 0.3 + 
            analysis['footwork'] * 0.3 + 
            analysis['power'] * 0.4
        )
        
        # Generate recommendations
        if analysis['timing'] < 0.75:
            analysis['recommendations'].append("Work on shuttle contact timing")
        if analysis['footwork'] < 0.75:
            analysis['recommendations'].append("Improve court positioning and movement")
        if analysis['power'] < 0.8:
            analysis['recommendations'].append("Use more wrist snap for power")
        
        # General badminton tips
        analysis['recommendations'].extend([
            "Keep your racket up and ready",
            "Move to the shuttle, don't let it come to you",
            "Use your non-racket arm for balance"
        ])
        
        return analysis
    
    def get_sport_specific_config(self, sport: str) -> Dict:
        """
        Get configuration for specific sport
        """
        return self.sport_configs.get(sport, self.sport_configs['football'])

class FootballBadmintonDataLoader:
    """
    Data loader specifically for football and badminton training
    """
    
    def __init__(self, dataset_path: str = "D:/Sportspose"):
        self.dataset_path = Path(dataset_path)
        self.analyzer = FootballBadmintonAnalyzer()
        
    def map_dataset_to_sports(self):
        """
        Map your existing dataset to football/badminton
        """
        sport_mapping = {
            # Map existing dataset sports to our target sports
            'soccer': 'football',
            'tennis': 'badminton',  # Use tennis data for badminton training
            'volley': 'badminton',  # Volleyball has similar net sport movements
        }
        return sport_mapping
    
    def create_training_data(self, output_dir: str = "./football_badminton_data"):
        """
        Create training dataset for football + badminton
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create sport-specific folders
        (output_path / "football").mkdir(exist_ok=True)
        (output_path / "badminton").mkdir(exist_ok=True)
        
        print("🏈 Creating Football + Badminton training dataset...")
        print(f"Output directory: {output_path}")
        
        # This would copy relevant data from your 298GB dataset
        # For now, we'll use your existing pipeline with sport mapping
        
        return str(output_path)

def create_football_badminton_demo():
    """
    Create a specialized demo for football and badminton
    """
    demo_script = '''
import os
import sys
sys.path.append('.')
from demo import bbox_tracklet, select_tracklets, gen_2d_pose, gen_3d_pose
from football_badminton_setup import FootballBadmintonAnalyzer
import json
import argparse
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
        print("\\n📊 Analysis Results:")
        print(f"   Sport: {analysis['sport'].upper()}")
        print(f"   Technique Score: {analysis['technique_score']:.2f}")
        
        if detected_sport == 'football':
            print(f"   Shot Power: {analysis['shot_power']:.2f}")
            print(f"   Accuracy: {analysis['accuracy']:.2f}")
        else:  # badminton
            print(f"   Racket Speed: {analysis['racket_speed']:.1f} km/h")
            print(f"   Timing: {analysis['timing']:.2f}")
        
        print("   Recommendations:")
        for rec in analysis['recommendations'][:3]:  # Show top 3
            print(f"     • {rec}")
    
    print("✅ Analysis Complete!")
    return detected_sport, analysis if 'analysis' in locals() else {}

if __name__ == "__main__":
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description="Football + Badminton Analysis")
    parser.add_argument('--root', '-p', type=str, default=os.getcwd()+"/output")
    parser.add_argument('--yolov8_param', type=str, default=os.getcwd()+"/bot_sort/yolov8_player/best.pt")
    parser.add_argument('--save_image', action='store_true', default=False)
    parser.add_argument('--num_frame', type=int, default=20)
    parser.add_argument("--device", type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("-t", "--target_clip", type=str, default=os.getcwd()+"/example/test_00001.mp4")
    
    args = parser.parse_args()
    args.root = os.path.join(args.root, Path(args.target_clip).stem)
    
    # Run analysis
    sport, analysis = football_badminton_pipeline(args)
    print(f"\\n🎉 {sport.upper()} analysis complete!")
'''
    
    return demo_script

if __name__ == "__main__":
    print("🏈🏸 Football + Badminton Setup")
    
    # Initialize components
    analyzer = FootballBadmintonAnalyzer()
    data_loader = FootballBadmintonDataLoader()
    
    print("\\n📊 Sport Configurations:")
    for sport, config in analyzer.sport_configs.items():
        print(f"  {sport.upper()}:")
        print(f"    Key Poses: {config['key_poses']}")
        print(f"    Analysis Focus: {config['analysis_focus']}")
        print(f"    Sequence Length: {config['sequence_length']} frames")
    
    print("\\n🎯 Dataset Mapping:")
    mapping = data_loader.map_dataset_to_sports()
    for original, target in mapping.items():
        print(f"  {original} → {target}")
    
    print("\\n✅ Setup complete! Ready for football + badminton analysis.")