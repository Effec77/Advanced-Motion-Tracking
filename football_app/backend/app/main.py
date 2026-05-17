
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn
from pathlib import Path
import sys
import os

# Add the parent directory to path to import from 3dsp_utils
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "3dsp_utils"))

try:
    from ai_engine.football_analyzer import FootballAnalyzer
    print("✅ Successfully imported FootballAnalyzer")
except ImportError as e:
    print(f"⚠️ Could not import FootballAnalyzer: {e}")
    FootballAnalyzer = None

app = FastAPI(
    title="Football Analysis API",
    description="AI-powered football technique analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize AI analyzer
analyzer = FootballAnalyzer() if FootballAnalyzer else None

@app.get("/")
async def root():
    return {"message": "Football Analysis API", "version": "1.0.0"}

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """
    Analyze uploaded football video
    """
    if not analyzer:
        raise HTTPException(status_code=500, detail="AI analyzer not available")
    
    try:
        # Save uploaded video to C drive
        video_path = f"C:/FootballData/temp/temp_{video.filename}"
        os.makedirs("C:/FootballData/temp", exist_ok=True)
        
        with open(video_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        # Run analysis (integrate with your 3dsp_utils pipeline)
        results = analyzer.analyze_football_video(video_path)
        
        # Clean up
        Path(video_path).unlink(missing_ok=True)
        
        return {"status": "success", "results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "analyzer": "ready"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
