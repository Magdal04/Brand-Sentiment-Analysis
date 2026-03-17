"""
FastAPI wrapper to expose your project functions to n8n
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add src to path
sys.path.insert(0, '/app/src')

app = FastAPI(title="Brand Sentiment Analysis API")


class YouTubeCollectRequest(BaseModel):
    channel_ids: list[str]
    max_results: int = 50


class AnalysisRequest(BaseModel):
    data_path: str


@app.get("/")
async def root():
    return {
        "message": "Brand Sentiment Analysis API",
        "status": "running"
    }


@app.post("/collect/youtube")
async def collect_youtube_data(request: YouTubeCollectRequest):
    """Trigger YouTube data collection"""
    try:
        # Import your youtube collector
        from ingestion.youtube_collector import collect_data
        
        result = collect_data(
            channel_ids=request.channel_ids,
            max_results=request.max_results
        )
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/sentiment")
async def analyze_sentiment(request: AnalysisRequest):
    """Trigger sentiment analysis"""
    try:
        # Import your analysis modules
        from preprocessing.text_processor import process_text
        from models.sentiment_analyzer import analyze
        
        # Your analysis logic here
        result = analyze(request.data_path)
        
        return {
            "status": "success",
            "results": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint for n8n"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
