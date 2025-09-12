import redis
import json
import os
import time
import math
from datetime import datetime, timedelta
import logging
import whisper
import signal
import sys
import threading
from pathlib import Path
import subprocess

# Configure logging to stderr (CRITICAL for Docker)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

logger = logging.getLogger(__name__)

class WhisperWorker:
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        self.model_name = os.getenv("WHISPER_MODEL", "large-v3")
        self.cleanup_hours = int(os.getenv("CLEANUP_HOURS", 24))
        self.results_ttl = int(os.getenv("RESULTS_TTL", 86400))
        self.running = True
        # Expected processing speed multiplier vs real-time (e.g., 7.0 means ~7x faster than audio length)
        # This is used only for progress estimation; transcription quality is unaffected.
        try:
            self.speed_multiplier = float(os.getenv("SPEED_MULTIPLIER", "7.0"))
        except Exception:
            self.speed_multiplier = 7.0
        
        # Auto-calibration settings
        self.calibration_samples = 10  # Use last N jobs for calibration
        self.calibration_key = "speed_calibration_history"
        
        # Initialize Whisper model
        logger.info(f"Loading Whisper model: {self.model_name}")
        try:
            self.model = whisper.load_model(self.model_name)
            logger.info(f"Whisper model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            sys.exit(1)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def update_progress_async(self, job_id, start_time, estimated_duration_minutes=None):
        """Update job progress asynchronously using estimated processing duration (minutes)."""
        progress_thread = threading.Thread(
            target=self._progress_updater, 
            args=(job_id, start_time, estimated_duration_minutes),
            daemon=True
        )
        progress_thread.start()
        return progress_thread
    
    def _progress_updater(self, job_id, start_time, estimated_duration_minutes):
        """Background thread to update job progress"""
        if estimated_duration_minutes is None:
            # Conservative estimate: 1 minute processing per 6-7 minutes of audio
            # Based on our RTF of ~0.15x
            estimated_duration_minutes = 3.0  # Default fallback
        
        while self.running:
            try:
                elapsed_minutes = (time.time() - start_time) / 60
                # Linear progress estimate based on expected processing time
                if not estimated_duration_minutes or estimated_duration_minutes <= 0:
                    estimated_duration_minutes = 3.0
                progress_pct = 100.0 * (elapsed_minutes / estimated_duration_minutes)
                # Cap at 95% until actually completed
                progress_pct = min(progress_pct, 95.0)
                
                # Get current job data
                job_data_raw = self.redis_client.get(f"job:{job_id}")
                if not job_data_raw:
                    break
                    
                job_data = json.loads(job_data_raw)
                
                # Only update if still processing
                if job_data.get("status") != "processing":
                    break
                    
                # Update progress
                job_data["progress"] = f"{progress_pct:.1f}%"
                job_data["elapsed_minutes"] = f"{elapsed_minutes:.1f}"
                
                # Store updated data
                self.redis_client.setex(f"job:{job_id}", self.results_ttl, json.dumps(job_data))
                
                # Update every 3 seconds
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Progress update error for job {job_id}: {e}")
                break
    
    def format_timestamp(self, seconds):
        """Convert seconds to VTT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def generate_vtt(self, segments, job_id):
        """Generate VTT file from segments"""
        vtt_path = f"/app/results/{job_id}.vtt"
        
        try:
            with open(vtt_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                for i, segment in enumerate(segments, 1):
                    start_time = self.format_timestamp(segment['start'])
                    end_time = self.format_timestamp(segment['end'])
                    text = segment['text'].strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            
            logger.info(f"VTT file generated: {vtt_path}")
            return vtt_path
            
        except Exception as e:
            logger.error(f"Failed to generate VTT file: {e}")
            return None
    
    def generate_txt(self, full_text, job_id):
        """Generate plain text file"""
        txt_path = f"/app/results/{job_id}.txt"
        
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                # Split text into paragraphs (every ~500 characters or at sentence breaks)
                words = full_text.split()
                paragraphs = []
                current_paragraph = []
                current_length = 0
                
                for word in words:
                    current_paragraph.append(word)
                    current_length += len(word) + 1
                    
                    # Create paragraph break at sentence end or ~500 chars
                    if (current_length > 500 and word.endswith(('.', '!', '?'))) or current_length > 800:
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                        current_length = 0
                
                # Add remaining words as last paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                
                # Write paragraphs
                for paragraph in paragraphs:
                    f.write(f"{paragraph.strip()}\n\n")
            
            logger.info(f"TXT file generated: {txt_path}")
            return txt_path
            
        except Exception as e:
            logger.error(f"Failed to generate TXT file: {e}")
            return None
    
    def _ffprobe_duration_seconds(self, audio_path):
        """Return audio duration in seconds using ffprobe, or None on failure."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
            return float(out)
        except Exception as e:
            logger.warning(f"ffprobe failed to get duration for {audio_path}: {e}")
            return None

    def _estimate_processing_minutes(self, audio_path, file_size):
        """Estimate processing time (minutes) and duration seconds.

        Prefers ffprobe duration; falls back to file-size heuristic.
        """
        duration_seconds = self._ffprobe_duration_seconds(audio_path)
        logger.info(f"FFprobe returned duration: {duration_seconds} seconds for {audio_path}")
        
        if duration_seconds and duration_seconds > 0:
            audio_minutes = duration_seconds / 60.0
            logger.info(f"Using ffprobe duration: {duration_seconds}s ({audio_minutes:.1f} min)")
        else:
            try:
                size_mb = (file_size or 0) / (1024 * 1024)
                # Conservative: ~1.2MB per minute average bitrate
                audio_minutes = max(size_mb / 1.2, 1.0)
                duration_seconds = audio_minutes * 60.0
                logger.info(f"Using file size fallback: {size_mb:.1f}MB -> {duration_seconds}s ({audio_minutes:.1f} min)")
            except Exception:
                # Final fallback
                audio_minutes = 3.0
                duration_seconds = audio_minutes * 60.0
                logger.info(f"Using final fallback: {duration_seconds}s ({audio_minutes:.1f} min)")

        # Estimated processing minutes based on calibrated speed multiplier
        try:
            current_speed = self.get_calibrated_speed_multiplier()
            proc_minutes = max(audio_minutes / max(current_speed, 0.1), 1.0)
        except Exception:
            proc_minutes = max(audio_minutes / 7.0, 1.0)

        logger.info(f"Final: audio_duration={duration_seconds}s, estimated_processing={proc_minutes:.1f}min")
        return proc_minutes, duration_seconds
    
    def update_speed_calibration(self, audio_duration, processing_time, transcription_result=None):
        """Update speed multiplier calibration with new measurement, filtering invalid samples"""
        logger.info(f"update_speed_calibration CALLED: duration={audio_duration}, time={processing_time}, has_result={transcription_result is not None}")
        if audio_duration <= 0 or processing_time <= 0:
            logger.info(f"Early return: invalid parameters")
            return
            
        try:
            # Calculate actual speed multiplier (audio_duration / processing_time)
            actual_speed = audio_duration / processing_time
            
            # Simple range-based filtering for realistic speech-to-text speeds
            # Skip samples outside reasonable range for normal speech processing
            if actual_speed < 3.0 or actual_speed > 15.0:
                logger.info(f"Skipping calibration sample: speed {actual_speed:.2f}x outside realistic range (3.0-15.0x) for duration {audio_duration:.1f}s")
                return
            
            # Valid sample - add to calibration history
            history_data = self.redis_client.get(self.calibration_key)
            if history_data:
                history = json.loads(history_data)
            else:
                history = []
            
            # Add new valid measurement
            history.append({
                "speed": actual_speed,
                "audio_duration": audio_duration,
                "processing_time": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Keep only last N valid samples
            history = history[-self.calibration_samples:]
            
            # Calculate new average speed multiplier
            speeds = [h["speed"] for h in history]
            new_speed_multiplier = sum(speeds) / len(speeds)
            
            # Update calibration data
            self.redis_client.setex(
                self.calibration_key, 
                self.results_ttl, 
                json.dumps(history)
            )
            
            # Update current speed multiplier (smoothed update to avoid wild swings)
            if len(history) >= 3:  # Need at least 3 valid samples for calibration
                # Weighted average: 70% old value, 30% new calibration
                old_multiplier = self.speed_multiplier
                self.speed_multiplier = 0.7 * self.speed_multiplier + 0.3 * new_speed_multiplier
                logger.info(f"Speed multiplier auto-calibrated: {old_multiplier:.2f}x -> {self.speed_multiplier:.2f}x (from {len(history)} valid samples, avg: {new_speed_multiplier:.2f}x)")
            else:
                logger.info(f"Speed calibration: collected valid sample {len(history)}/3 (speed: {actual_speed:.2f}x), baseline: {self.speed_multiplier:.2f}x")
            
        except Exception as e:
            logger.warning(f"Speed calibration update failed: {e}")
    
    def get_calibrated_speed_multiplier(self):
        """Get current calibrated speed multiplier, with fallback to baseline"""
        try:
            history_data = self.redis_client.get(self.calibration_key)
            if history_data:
                history = json.loads(history_data)
                if len(history) >= 3:
                    # Use calibrated value if we have enough samples
                    speeds = [h["speed"] for h in history[-5:]]  # Use last 5 samples
                    calibrated_speed = sum(speeds) / len(speeds)
                    logger.info(f"Using calibrated speed multiplier: {calibrated_speed:.2f}x (from {len(speeds)} recent samples)")
                    return calibrated_speed
            
            # Fallback to configured baseline (7.0 by default)
            baseline = float(os.getenv("SPEED_MULTIPLIER", "7.0"))
            logger.info(f"Using baseline speed multiplier: {baseline:.2f}x (no calibration data)")
            return baseline
            
        except Exception as e:
            logger.warning(f"Speed calibration lookup failed: {e}, using baseline 7.0x")
            return 7.0

    def process_audio(self, job_id, audio_path):
        """Process audio file with Whisper"""
        try:
            logger.info(f"Starting transcription for job {job_id}")
            start_time = time.time()
            
            # Update job status
            job_data = json.loads(self.redis_client.get(f"job:{job_id}"))
            job_data["status"] = "processing"
            job_data["started_at"] = datetime.utcnow().isoformat()
            
            # Estimate processing time and real audio duration (for progress tracking)
            file_size = job_data.get("file_size", 0)
            estimated_duration_minutes, duration_seconds = self._estimate_processing_minutes(audio_path, file_size)

            # Initialize progress
            job_data["progress"] = "0.0%"
            job_data["elapsed_minutes"] = "0.0"
            job_data["estimated_duration_minutes"] = f"{estimated_duration_minutes:.1f}"
            job_data["audio_duration"] = duration_seconds
            
            self.redis_client.setex(f"job:{job_id}", self.results_ttl, json.dumps(job_data))
            
            # Start async progress updater
            progress_thread = self.update_progress_async(job_id, start_time, estimated_duration_minutes)
            
            # Get language from job data
            language = job_data.get("language")  # None = auto-detect
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=language,  # None = auto-detect, "pl", "en", etc.
                word_timestamps=True,
                verbose=True
            )
            
            # Extract segments
            segments = result["segments"]
            full_text = result["text"]
            
            # Generate output files
            vtt_path = self.generate_vtt(segments, job_id)
            txt_path = self.generate_txt(full_text, job_id)
            
            processing_time = time.time() - start_time
            
            # Prepare result data
            result_data = {
                "job_id": job_id,
                "text": full_text,
                "segments": [
                    {
                        "start": float(segment["start"]),
                        "end": float(segment["end"]),
                        "text": segment["text"].strip()
                    } for segment in segments
                ],
                "language": result["language"],
                "processing_time": processing_time,
                "audio_duration": duration_seconds,
                "vtt_available": vtt_path is not None,
                "txt_available": txt_path is not None
            }
            
            # Store result with TTL
            self.redis_client.setex(f"result:{job_id}", self.results_ttl, json.dumps(result_data))
            
            # Update job as completed with final progress
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.utcnow().isoformat()
            job_data["processing_time"] = processing_time
            job_data["audio_duration"] = duration_seconds  # Preserve audio duration
            job_data["progress"] = "100.0%"
            job_data["elapsed_minutes"] = f"{processing_time/60:.1f}"
            self.redis_client.setex(f"job:{job_id}", self.results_ttl, json.dumps(job_data))
            
            # Update speed calibration with this job's performance
            # Pass transcription result for quality filtering
            try:
                self.update_speed_calibration(duration_seconds, processing_time, result)
            except Exception as calib_error:
                logger.error(f"Speed calibration failed: {calib_error}")
                logger.exception("Speed calibration exception details:")
            
            logger.info(f"Job {job_id} completed successfully in {processing_time:.2f}s")
            
            # Cleanup audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Removed audio file: {audio_path}")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            
            # Mark job as failed and stop progress updates
            job_data = json.loads(self.redis_client.get(f"job:{job_id}") or "{}")
            job_data["status"] = "failed"
            job_data["error"] = str(e)
            job_data["failed_at"] = datetime.utcnow().isoformat()
            # Keep last progress info for debugging
            self.redis_client.setex(f"job:{job_id}", self.results_ttl, json.dumps(job_data))
            
            # Cleanup audio file even on failure
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    def cleanup_old_files(self):
        """Clean up old result files"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.cleanup_hours)
            
            for directory in ["/app/results", "/app/audio_temp"]:
                if not os.path.exists(directory):
                    continue
                    
                for file_path in Path(directory).iterdir():
                    if file_path.is_file():
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            file_path.unlink()
                            logger.info(f"Cleaned up old file: {file_path}")
                            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def run(self):
        """Main worker loop"""
        logger.info("Whisper worker started")
        last_cleanup = time.time()
        
        while self.running:
            try:
                # Get job from queue (blocking with timeout)
                job_data = self.redis_client.brpop("transcription_queue", timeout=5)
                
                if job_data:
                    job_id = job_data[1].decode('utf-8')
                    logger.info(f"Processing job: {job_id}")
                    
                    # Get job details
                    job_info_data = self.redis_client.get(f"job:{job_id}")
                    if not job_info_data:
                        logger.warning(f"Job {job_id} not found in Redis, skipping")
                        continue
                    
                    job_info = json.loads(job_info_data)
                    audio_path = job_info["audio_path"]
                    
                    if not os.path.exists(audio_path):
                        logger.warning(f"Audio file not found: {audio_path}")
                        continue
                    
                    # Process the job
                    self.process_audio(job_id, audio_path)
                
                # Periodic cleanup (every hour)
                if time.time() - last_cleanup > 3600:
                    self.cleanup_old_files()
                    last_cleanup = time.time()
                
            except redis.ConnectionError:
                logger.error("Redis connection lost, retrying in 5 seconds...")
                time.sleep(5)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in worker: {e}")
                time.sleep(1)
        
        logger.info("Whisper worker stopped")

if __name__ == "__main__":
    worker = WhisperWorker()
    worker.run()
