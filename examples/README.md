# 🎯 Barca Motion AI - Examples & Demos

This directory contains sample code, test videos, and example outputs to help you get started with Barca Motion AI.

## 📁 Directory Structure

```
examples/
├── videos/                    # Sample video files
│   ├── football_shot_demo.mp4       # Football shooting technique
│   ├── football_dribble_demo.mp4    # Ball control demonstration
│   └── multi_player_demo.mp4        # Multiple players scenario
├── api_examples/              # API usage examples
│   ├── single_video_analysis.py     # Basic video analysis
│   ├── batch_processing.py          # Multiple video processing
│   └── real_time_monitoring.py      # Live processing status
├── outputs/                   # Sample analysis results
│   ├── sample_analysis.json         # Complete analysis output
│   ├── technique_scores.json        # Technique scoring examples
│   └── coaching_insights.json       # Coaching recommendations
└── notebooks/                 # Jupyter notebooks
    ├── getting_started.ipynb        # Quick start tutorial
    ├── advanced_analysis.ipynb      # Deep dive into results
    └── visualization.ipynb          # Data visualization examples
```

## 🎮 Quick Start Examples

### 1. Basic Video Analysis
```python
import requests

# Upload and analyze a video
with open('examples/videos/football_shot_demo.mp4', 'rb') as video_file:
    response = requests.post(
        'http://localhost:8003/analyze',
        files={'video': video_file}
    )

result = response.json()
print(f"Technique Score: {result['analysis']['technique_score']:.1%}")
```

### 2. Batch Processing
```python
from pathlib import Path
import requests

# Process multiple videos
video_dir = Path('examples/videos')
for video_file in video_dir.glob('*.mp4'):
    with open(video_file, 'rb') as f:
        response = requests.post(
            'http://localhost:8003/analyze',
            files={'video': f}
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"{video_file.name}: {result['analysis']['technique_score']:.1%}")
```

### 3. Real-time Monitoring
```python
import requests
import time

# Start batch processing
response = requests.post(
    'http://localhost:8003/batch/start-processing',
    params={'max_videos': 5, 'sport_filter': 'football'}
)

# Monitor progress
while True:
    status = requests.get('http://localhost:8003/batch/status').json()
    print(f"Progress: {status['batch_status']['completed']}/{status['batch_status']['total_jobs']}")
    
    if status['batch_status']['processing'] == 0:
        break
    
    time.sleep(10)
```

## 📊 Sample Analysis Results

### Complete Analysis Output
```json
{
  "status": "success",
  "video_name": "football_shot_demo.mp4",
  "analysis": {
    "sport": "football",
    "technique_type": "shooting",
    "technique_score": 0.849,
    "shot_power": 0.765,
    "accuracy": 0.688,
    "ball_control": 0.841,
    "recommendations": [
      "Increase follow-through for more power",
      "Keep your head up and aim for corners",
      "Plant your standing foot firmly next to the ball",
      "Strike through the center of the ball",
      "Follow through in direction of target"
    ]
  },
  "output_path": "C:/FootballData/analysis/football_shot_demo",
  "pipeline_results": {
    "step_1_yolo": {"success": true, "message": "YOLO tracking completed successfully"},
    "step_2_tracklets": {"success": true, "message": "Tracklet selection completed - 20 images"},
    "step_3_2d_pose": {"success": true, "message": "2D pose estimation completed"},
    "step_4_3d_pose": {"success": true, "message": "3D pose estimation completed"},
    "step_5_analysis": {"success": true, "message": "Football analysis completed"}
  }
}
```

### Technique Scoring Breakdown
```json
{
  "technique_metrics": {
    "overall_score": 84.9,
    "shot_power": {
      "score": 76.5,
      "components": {
        "leg_swing_velocity": 0.82,
        "hip_rotation": 0.74,
        "follow_through": 0.71
      }
    },
    "accuracy": {
      "score": 68.8,
      "components": {
        "body_alignment": 0.72,
        "foot_placement": 0.65,
        "head_position": 0.69
      }
    },
    "ball_control": {
      "score": 84.1,
      "components": {
        "approach_angle": 0.87,
        "first_touch": 0.81,
        "balance": 0.84
      }
    }
  }
}
```

## 🎯 Use Case Examples

### 1. Youth Training Analysis
```python
# Analyze youth player technique
def analyze_youth_player(video_path):
    with open(video_path, 'rb') as video:
        response = requests.post(
            'http://localhost:8003/analyze',
            files={'video': video}
        )
    
    result = response.json()
    
    # Focus on fundamental techniques for youth
    fundamentals = {
        'ball_control': result['analysis']['ball_control'],
        'basic_technique': result['analysis']['technique_score'],
        'improvement_areas': result['analysis']['recommendations'][:3]
    }
    
    return fundamentals
```

### 2. Professional Scouting
```python
# Professional player evaluation
def scout_player_analysis(video_path):
    with open(video_path, 'rb') as video:
        response = requests.post(
            'http://localhost:8003/analyze',
            files={'video': video}
        )
    
    result = response.json()
    
    # Professional-level metrics
    scouting_report = {
        'overall_rating': result['analysis']['technique_score'] * 100,
        'shot_power_rating': result['analysis']['shot_power'] * 100,
        'accuracy_rating': result['analysis']['accuracy'] * 100,
        'technical_skills': result['analysis']['ball_control'] * 100,
        'development_potential': calculate_potential(result['analysis'])
    }
    
    return scouting_report

def calculate_potential(analysis):
    # Custom potential calculation based on technique scores
    base_score = analysis['technique_score']
    improvement_factors = len(analysis['recommendations'])
    return min(100, base_score * 100 + (improvement_factors * 5))
```

### 3. Team Performance Analysis
```python
# Analyze multiple players from a team
def team_analysis(video_directory):
    team_results = []
    
    for video_file in Path(video_directory).glob('*.mp4'):
        with open(video_file, 'rb') as video:
            response = requests.post(
                'http://localhost:8003/analyze',
                files={'video': video}
            )
        
        if response.status_code == 200:
            result = response.json()
            team_results.append({
                'player': video_file.stem,
                'technique_score': result['analysis']['technique_score'],
                'shot_power': result['analysis']['shot_power'],
                'accuracy': result['analysis']['accuracy'],
                'ball_control': result['analysis']['ball_control']
            })
    
    # Calculate team averages
    team_average = {
        'avg_technique': sum(r['technique_score'] for r in team_results) / len(team_results),
        'avg_shot_power': sum(r['shot_power'] for r in team_results) / len(team_results),
        'avg_accuracy': sum(r['accuracy'] for r in team_results) / len(team_results),
        'avg_ball_control': sum(r['ball_control'] for r in team_results) / len(team_results)
    }
    
    return team_results, team_average
```

## 📱 Integration Examples

### Web Application Integration
```javascript
// Frontend JavaScript example
async function analyzeVideo(videoFile) {
    const formData = new FormData();
    formData.append('video', videoFile);
    
    try {
        const response = await fetch('http://localhost:8003/analyze', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // Display results
        document.getElementById('technique-score').textContent = 
            `${(result.analysis.technique_score * 100).toFixed(1)}%`;
        document.getElementById('recommendations').innerHTML = 
            result.analysis.recommendations.map(r => `<li>${r}</li>`).join('');
            
    } catch (error) {
        console.error('Analysis failed:', error);
    }
}
```

### Mobile App Integration
```dart
// Flutter/Dart example
Future<Map<String, dynamic>> analyzeVideo(File videoFile) async {
  var request = http.MultipartRequest(
    'POST', 
    Uri.parse('http://localhost:8003/analyze')
  );
  
  request.files.add(
    await http.MultipartFile.fromPath('video', videoFile.path)
  );
  
  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  
  return json.decode(responseData);
}
```

## 🔧 Development Examples

### Custom Analysis Pipeline
```python
# Custom analysis with additional metrics
class CustomFootballAnalyzer:
    def __init__(self, api_base_url="http://localhost:8003"):
        self.api_base_url = api_base_url
    
    def analyze_with_custom_metrics(self, video_path):
        # Get standard analysis
        standard_result = self.get_standard_analysis(video_path)
        
        # Add custom metrics
        custom_metrics = self.calculate_custom_metrics(standard_result)
        
        # Combine results
        enhanced_result = {
            **standard_result,
            'custom_metrics': custom_metrics
        }
        
        return enhanced_result
    
    def calculate_custom_metrics(self, standard_result):
        # Custom biomechanical calculations
        technique_score = standard_result['analysis']['technique_score']
        
        return {
            'explosiveness': self.calculate_explosiveness(standard_result),
            'consistency': self.calculate_consistency(standard_result),
            'improvement_potential': 1.0 - technique_score
        }
```

## 📊 Visualization Examples

### Performance Dashboard
```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_performance_dashboard(analysis_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Technique scores radar chart
    categories = ['Shot Power', 'Accuracy', 'Ball Control', 'Overall']
    scores = [
        analysis_results['shot_power'],
        analysis_results['accuracy'], 
        analysis_results['ball_control'],
        analysis_results['technique_score']
    ]
    
    # Create visualizations
    # ... (visualization code)
    
    plt.tight_layout()
    plt.show()
```

## 🎯 Testing Examples

### Unit Tests
```python
import pytest
from football_app.backend.working_server import analyze_video

class TestVideoAnalysis:
    def test_successful_football_analysis(self):
        """Test successful football video analysis"""
        video_path = "examples/videos/football_shot_demo.mp4"
        result = analyze_video(video_path)
        
        assert result["status"] == "success"
        assert "technique_score" in result["analysis"]
        assert 0 <= result["analysis"]["technique_score"] <= 1
        assert len(result["analysis"]["recommendations"]) > 0
    
    def test_invalid_video_format(self):
        """Test handling of invalid video formats"""
        with pytest.raises(ValueError):
            analyze_video("invalid_file.txt")
```

## 📚 Documentation Examples

### API Documentation
```python
from fastapi import FastAPI, UploadFile, File
from typing import Dict, Any

app = FastAPI()

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_video(video: UploadFile = File(...)):
    """
    Analyze football technique from uploaded video
    
    Args:
        video: MP4, AVI, MOV, or MKV video file
        
    Returns:
        Dict containing:
        - status: "success" or "error"
        - analysis: Technique scores and recommendations
        - pipeline_results: Processing step details
        
    Example:
        ```python
        with open("football_shot.mp4", "rb") as f:
            response = requests.post("/analyze", files={"video": f})
        ```
    """
    # Implementation...
```

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API Server**:
   ```bash
   cd football_app/backend
   python working_server.py
   ```

3. **Run Examples**:
   ```bash
   python examples/api_examples/single_video_analysis.py
   ```

4. **Explore Notebooks**:
   ```bash
   jupyter notebook examples/notebooks/getting_started.ipynb
   ```

## 📞 Support

For questions about these examples:
- Check the main [README.md](../README.md)
- Review [API Documentation](http://localhost:8003/docs)
- Open an issue on GitHub
- Contact the development team

---

**Happy analyzing! ⚽🤖**