"""
Minimal Football Analysis API - Bypass Complex Pipeline
Simple working server that returns mock analysis while we debug the main pipeline
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from pathlib import Path
import json
import time
import random

# Initialize FastAPI app
app = FastAPI(
    title="Football Analysis API (Minimal)",
    description="Simplified football analysis API for testing",
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

# Setup directories
def setup_directories():
    directories = [
        "C:/FootballData/uploads",
        "C:/FootballData/analysis", 
        "C:/FootballData/temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

setup_directories()

@app.get("/")
async def root():
    return {
        "message": "Football Analysis API (Minimal Version)",
        "version": "1.0.0",
        "status": "ready",
        "note": "This is a simplified version for testing"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": "minimal",
        "storage": "C:/FootballData ready"
    }

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """
    Simplified video analysis - returns mock data for now
    """
    try:
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        print(f"📤 Processing video: {video.filename}")
        
        # Save uploaded video
        video_name = video.filename
        upload_path = f"C:/FootballData/uploads/{video_name}"
        
        with open(upload_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        print(f"✅ Video saved to: {upload_path}")
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Generate mock analysis results
        mock_analysis = {
            "technique_score": round(random.uniform(0.7, 0.95), 2),
            "shot_power": round(random.uniform(0.6, 0.9), 2),
            "accuracy": round(random.uniform(0.65, 0.85), 2),
            "ball_control": round(random.uniform(0.7, 0.9), 2),
            "recommendations": [
                "Focus on follow-through for better accuracy",
                "Improve body positioning during shot preparation",
                "Work on consistent ball contact point"
            ],
            "processing_details": {
                "video_file": video_name,
                "frames_analyzed": 20,
                "players_detected": random.randint(1, 3),
                "processing_time": "2.1 seconds"
            }
        }
        
        # Save analysis results
        analysis_dir = f"C:/FootballData/analysis/{Path(video_name).stem}"
        os.makedirs(analysis_dir, exist_ok=True)
        
        analysis_file = os.path.join(analysis_dir, "football_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(mock_analysis, f, indent=2)
        
        print(f"✅ Analysis complete for {video_name}")
        
        return JSONResponse(content={
            "status": "success",
            "video_name": video_name,
            "analysis": mock_analysis,
            "output_path": analysis_dir,
            "note": "This is mock data - real AI pipeline integration in progress"
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
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

@app.get("/test-pipeline")
async def test_pipeline():
    """Test if we can run the actual pipeline components"""
    try:
        # Test basic imports
        import sys
        project_root = Path(__file__).parent.parent.parent
        sys.path.append(str(project_root / "3dsp_utils"))
        
        results = {
            "imports": {},
            "files": {},
            "directories": {}
        }
        
        # Test imports
        try:
            from football_badminton_setup import FootballBadmintonAnalyzer
            results["imports"]["FootballBadmintonAnalyzer"] = "✅ Success"
        except Exception as e:
            results["imports"]["FootballBadmintonAnalyzer"] = f"❌ Failed: {e}"
        
        try:
            from bot_sort.tools.shot_post import sn_demo
            results["imports"]["sn_demo"] = "✅ Success"
        except Exception as e:
            results["imports"]["sn_demo"] = f"❌ Failed: {e}"
        
        # Test file paths
        yolo_path = project_root / "3dsp_utils/bot_sort/yolov8_player/best.pt"
        results["files"]["yolo_model"] = "✅ Exists" if yolo_path.exists() else "❌ Missing"
        
        example_video = project_root / "3dsp_utils/example/test_00001.mp4"
        results["files"]["example_video"] = "✅ Exists" if example_video.exists() else "❌ Missing"
        
        # Test directories
        results["directories"]["C:/FootballData"] = "✅ Ready" if Path("C:/FootballData").exists() else "❌ Missing"
        
        return {
            "status": "diagnostic_complete",
            "results": results
        }
        
    except Exception as e:
        return {
            "status": "diagnostic_failed",
            "error": str(e)
        }

if __name__ == "__main__":
    import asyncio
    
    print("🚀 Starting Minimal Football Analysis API...")
    print("📊 API Documentation: http://localhost:8002/docs")
    print("🧪 Pipeline Test: http://localhost:8002/test-pipeline")
    print("⚽ Ready for testing!")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)