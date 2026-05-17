"""
FastAPI wrapper for Football Analysis Server
Provides REST API endpoints for the working backend
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from pathlib import Path
import json

# Import our working server
from simple_server import FootballAnalysisServer

# Initialize FastAPI app
app = FastAPI(
    title="Football Analysis API",
    description="AI-powered football technique analysis using 3D pose estimation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our analysis server
analysis_server = FootballAnalysisServer()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Football Analysis API",
        "version": "1.0.0",
        "status": "ready",
        "features": [
            "Video upload and analysis",
            "3D pose estimation", 
            "Football technique scoring",
            "Coaching recommendations",
            "Analysis history"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_analyzer": "available",
        "storage": "C:/FootballData ready"
    }

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """
    Analyze uploaded football video
    
    Returns:
    - technique_score: Overall technique rating (0-1)
    - shot_power: Shot power analysis (0-1)
    - accuracy: Shot accuracy analysis (0-1)
    - ball_control: Ball control analysis (0-1)
    - recommendations: List of coaching tips
    """
    try:
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid video format. Supported: mp4, avi, mov, mkv")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Analyze video using our working server
            result = analysis_server.analyze_video(temp_path, Path(video.filename).stem)
            
            if result["status"] == "success":
                return JSONResponse(content={
                    "status": "success",
                    "video_name": video.filename,
                    "analysis": result["analysis"],
                    "output_path": result["output_path"],
                    "processing_time": "~60 seconds"
                })
            else:
                raise HTTPException(status_code=500, detail=result["error"])
                
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/history")
async def get_analysis_history():
    """
    Get analysis history for user
    """
    try:
        history = analysis_server.get_analysis_history()
        return {
            "status": "success",
            "count": len(history),
            "analyses": history[:10]  # Return last 10 analyses
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.get("/analysis/{analysis_id}")
async def get_analysis_details(analysis_id: str):
    """
    Get detailed results for a specific analysis
    """
    try:
        analysis_path = Path(f"C:/FootballData/analysis/{analysis_id}")
        analysis_file = analysis_path / "football_analysis.json"
        
        if not analysis_file.exists():
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        # Check for additional files
        files = {
            "analysis_json": str(analysis_file),
            "keypoints": str(analysis_path / "output_3D" / "keypoints.npz") if (analysis_path / "output_3D" / "keypoints.npz").exists() else None,
            "result_video": str(analysis_path / f"{analysis_id}.mp4") if (analysis_path / f"{analysis_id}.mp4").exists() else None,
            "shooter_images": str(analysis_path / "shooter_image") if (analysis_path / "shooter_image").exists() else None
        }
        
        return {
            "status": "success",
            "analysis_id": analysis_id,
            "analysis": analysis,
            "files": files,
            "timestamp": analysis_path.stat().st_mtime
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis: {str(e)}")

@app.get("/stats")
async def get_server_stats():
    """
    Get server statistics
    """
    try:
        # Get storage usage
        data_path = Path("C:/FootballData")
        total_size = sum(f.stat().st_size for f in data_path.rglob('*') if f.is_file())
        
        # Get analysis count
        analysis_count = len(list((data_path / "analysis").glob("*"))) if (data_path / "analysis").exists() else 0
        
        # Get upload count  
        upload_count = len(list((data_path / "uploads").glob("*"))) if (data_path / "uploads").exists() else 0
        
        return {
            "status": "success",
            "storage": {
                "total_size_gb": round(total_size / (1024**3), 2),
                "data_path": str(data_path)
            },
            "analyses": {
                "total_count": analysis_count,
                "recent": analysis_server.get_analysis_history()[:3]
            },
            "uploads": {
                "total_count": upload_count
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Football Analysis API Server...")
    print("📊 API Documentation: http://localhost:8001/docs")
    print("🎯 Health Check: http://localhost:8001/health")
    print("⚽ Ready for football video analysis!")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)