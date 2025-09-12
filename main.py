from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
import redis
import json
import uuid
import os
import aiofiles
from datetime import datetime, timedelta
import asyncio
from typing import Optional

app = FastAPI(title="Whisper Speech-to-Text API", version="1.0.0")

# Redis connection
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 524288000))  # 500MB
CLEANUP_HOURS = int(os.getenv("CLEANUP_HOURS", 24))
RESULTS_TTL = int(os.getenv("RESULTS_TTL", 86400))  # 24 hours in seconds

# Allowed audio formats
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.mp4', '.avi', '.mov'}

def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    # Accept language from multipart form (or omit to auto-detect)
    language: Optional[str] = Form(None)
):
    """Upload audio file for transcription"""
    
    # Validate file extension
    if get_file_extension(file.filename) not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_FILE_SIZE//1024//1024}MB")
    
    # Generate job ID and save file
    job_id = str(uuid.uuid4())
    file_extension = get_file_extension(file.filename)
    audio_path = f"/app/audio_temp/{job_id}{file_extension}"
    
    # Save uploaded file
    async with aiofiles.open(audio_path, 'wb') as f:
        await f.write(file_content)
    
    # Create job in Redis
    job_data = {
        "id": job_id,
        "filename": file.filename,
        "audio_path": audio_path,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "file_size": len(file_content),
        "language": language  # None = auto-detect
    }
    
    # Store job data with TTL
    redis_client.setex(f"job:{job_id}", RESULTS_TTL, json.dumps(job_data))
    
    # Add to processing queue
    redis_client.lpush("transcription_queue", job_id)
    
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "message": "File uploaded successfully. Transcription will start shortly."
    })

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get transcription job status"""
    
    job_data = redis_client.get(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    
    job_info = json.loads(job_data)
    
    response = {
        "job_id": job_id,
        "status": job_info["status"],
        "filename": job_info["filename"],
        "created_at": job_info["created_at"],
        "file_size": job_info["file_size"]
    }
    
    # Add progress info if available
    if "progress" in job_info:
        response["progress"] = job_info["progress"]
    
    # Add completion info if done
    if job_info["status"] == "completed":
        response["completed_at"] = job_info.get("completed_at")
        response["audio_duration"] = job_info.get("audio_duration")
        response["processing_time"] = job_info.get("processing_time")
    
    # Add error info if failed
    if job_info["status"] == "failed":
        response["error"] = job_info.get("error", "Unknown error")
    
    return JSONResponse(response)

@app.get("/result/{job_id}")
async def get_transcription_result(job_id: str, format: Optional[str] = "json"):
    """Get transcription result in JSON format"""
    
    job_data = redis_client.get(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    
    job_info = json.loads(job_data)
    
    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job_info['status']}")
    
    # Get result from Redis
    result_data = redis_client.get(f"result:{job_id}")
    if not result_data:
        raise HTTPException(status_code=404, detail="Transcription result not found or expired")
    
    result = json.loads(result_data)
    
    if format == "json":
        return JSONResponse(result)
    else:
        raise HTTPException(status_code=400, detail="Only 'json' format is supported")

@app.get("/result/{job_id}/vtt")
async def download_vtt(job_id: str):
    """Download VTT subtitle file"""
    
    vtt_path = f"/app/results/{job_id}.vtt"
    if not os.path.exists(vtt_path):
        raise HTTPException(status_code=404, detail="VTT file not found or expired")
    
    return FileResponse(
        vtt_path,
        media_type="text/vtt",
        filename=f"{job_id}.vtt"
    )

@app.get("/result/{job_id}/txt")
async def download_txt(job_id: str):
    """Download plain text file"""
    
    txt_path = f"/app/results/{job_id}.txt"
    if not os.path.exists(txt_path):
        raise HTTPException(status_code=404, detail="TXT file not found or expired")
    
    return FileResponse(
        txt_path,
        media_type="text/plain",
        filename=f"{job_id}.txt"
    )

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete job and cleanup associated files"""
    
    job_data = redis_client.get(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = json.loads(job_data)
    
    # Delete files
    files_to_delete = [
        job_info.get("audio_path"),
        f"/app/results/{job_id}.vtt",
        f"/app/results/{job_id}.txt"
    ]
    
    for file_path in files_to_delete:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    
    # Delete Redis data
    redis_client.delete(f"job:{job_id}")
    redis_client.delete(f"result:{job_id}")
    
    return JSONResponse({"message": "Job deleted successfully"})

@app.get("/calibration")
async def get_calibration_status():
    """Get speed calibration status and history"""
    try:
        calibration_key = "speed_calibration_history"
        history_data = redis_client.get(calibration_key)
        
        if history_data:
            history = json.loads(history_data)
            if history:
                speeds = [h["speed"] for h in history]
                avg_speed = sum(speeds) / len(speeds)
                recent_speeds = [h["speed"] for h in history[-3:]]
                recent_avg = sum(recent_speeds) / len(recent_speeds)
                
                return JSONResponse({
                    "status": "calibrated",
                    "samples": len(history),
                    "average_speed": round(avg_speed, 2),
                    "recent_average_speed": round(recent_avg, 2),
                    "baseline_speed": float(os.getenv("SPEED_MULTIPLIER", "8.0")),
                    "recent_measurements": [
                        {
                            "speed": round(h["speed"], 2),
                            "audio_duration": round(h["audio_duration"], 1),
                            "processing_time": round(h["processing_time"], 1),
                            "text_length": h.get("text_length", 0),
                            "segments_count": h.get("segments_count", 0),
                            "timestamp": h["timestamp"]
                        } for h in history[-5:]  # Last 5 measurements
                    ]
                })
        
        return JSONResponse({
            "status": "baseline",
            "samples": 0,
            "baseline_speed": float(os.getenv("SPEED_MULTIPLIER", "8.0")),
            "message": "No calibration data yet, using baseline"
        })
        
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Failed to get calibration status: {e}"
        }, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    # Check Redis connection
    try:
        redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    # Check queue size
    queue_size = redis_client.llen("transcription_queue")
    
    return JSONResponse({
        "status": "healthy",
        "redis": redis_status,
        "queue_size": queue_size,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.get("/")
async def root():
    """API information"""
    return JSONResponse({
        "name": "Whisper Speech-to-Text API",
        "version": "1.0.0",
        "endpoints": {
            "POST /transcribe": "Upload audio file for transcription (optional: ?language=pl)",
            "GET /status/{job_id}": "Check transcription status",
            "GET /result/{job_id}": "Get transcription result (JSON)",
            "GET /result/{job_id}/vtt": "Download VTT subtitle file",
            "GET /result/{job_id}/txt": "Download plain text file",
            "DELETE /job/{job_id}": "Delete job and cleanup files",
            "GET /health": "Health check",
            "GET /calibration": "Speed calibration status"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size": f"{MAX_FILE_SIZE//1024//1024}MB"
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
