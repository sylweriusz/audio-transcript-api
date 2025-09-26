# Audio Transcript API

High-performance Speech-to-Text solution using OpenAI Whisper (PyTorch/CUDA), FastAPI, Redis, and NVIDIA GPU acceleration.

## Features

- 🎯 **Maximum Performance**: OpenAI Whisper on GPU (CUDA)
- 🔥 **Large Files**: Support for files up to 500MB
- 📦 **Job Queue**: Asynchronous processing with Redis
- 📊 **Multiple Formats**: TXT (paragraphs) + VTT (subtitles with timestamps)
- 🧹 **Auto-cleanup**: Automatic removal of old files (24h)
- 🚀 **Production-ready**: Docker Compose with GPU support
- ⚡ **Auto-calibrating**: Learns and adapts to hardware performance

## Requirements

- Docker + Docker Compose
- NVIDIA GPU with at least 6GB VRAM (recommended 8GB+)
- nvidia-docker2

## Quick Start

```bash
# Start all services
docker-compose up -d

# Check service status
curl http://localhost:8000/health
```

## API Endpoints

### Upload audio file
```bash
# Auto-detect language
curl -X POST -F "file=@audio.mp3" http://localhost:8000/transcribe

# Specify language (pl, en, de, fr, es, it, etc.)
curl -X POST -F "file=@audio.mp3" -F "language=pl" http://localhost:8000/transcribe

# Response: {"job_id": "uuid", "status": "queued"}
```

### Check status
```bash
curl http://localhost:8000/status/UUID
# Response: {"job_id": "uuid", "status": "processing|completed|failed", "progress": "45.2%"}
```

### Get result (JSON)
```bash
curl http://localhost:8000/result/UUID
# Response: {"text": "...", "segments": [...], "language": "pl", "audio_duration": 120.5, "processing_time": 18.3}
```

### Download VTT (subtitles)
```bash
curl http://localhost:8000/result/UUID/vtt -o subtitles.vtt
```

### Download TXT (paragraphs)
```bash
curl http://localhost:8000/result/UUID/txt -o transcript.txt
```

### Auto-calibration status
```bash
curl http://localhost:8000/calibration
# Response: {"status": "calibrated", "samples": 5, "average_speed": 6.8, "baseline_speed": 8.0}
```

## Supported Formats

**Audio**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.opus`  
**Video**: `.mp4`, `.avi`, `.mov` (extracts audio)

## Configuration

### Environment Variables (.env)

Environment variables are defined in a `.env` file at the project root. Docker Compose automatically loads these variables and applies default values if they are not set.

Example `.env`:
```env
WHISPER_MODEL=turbo         # tiny, base, small, medium, large-v3, turbo
MAX_FILE_SIZE=524288000     # 500MB limit
CLEANUP_HOURS=24            # Auto-delete after 24h
RESULTS_TTL=86400           # Redis TTL: 24h
SPEED_MULTIPLIER=24.0       # Performance baseline (Turbo optimized)
WHISPER_API_PORT=8000       # External port for whisper-api service
```

In `docker-compose.yml`, variables are used like this:
```yaml
environment:
    - WHISPER_MODEL=${WHISPER_MODEL:-turbo}
    - MAX_FILE_SIZE=${MAX_FILE_SIZE:-524288000}
    - CLEANUP_HOURS=${CLEANUP_HOURS:-24}
    - RESULTS_TTL=${RESULTS_TTL:-86400}
    - SPEED_MULTIPLIER=${SPEED_MULTIPLIER:-24.0}
ports:
    - "${WHISPER_API_PORT:-8000}:8000"
```

**Notes:**
- If the `.env` file does not exist, Docker Compose will use the default values.
- To change the external port for the whisper-api service, simply set `WHISPER_API_PORT` in your `.env` file.

### Whisper Models

| Model | VRAM | Quality | Speed |
|-------|------|---------|-------|
| `tiny` | ~1GB | ⭐⭐ | ⚡⚡⚡⚡⚡ |
| `base` | ~1GB | ⭐⭐⭐ | ⚡⚡⚡⚡ |
| `small` | ~2GB | ⭐⭐⭐⭐ | ⚡⚡⚡ |
| `medium` | ~5GB | ⭐⭐⭐⭐ | ⚡⚡ |
| `large-v3` | ~10GB | ⭐⭐⭐⭐⭐ | ⚡ |
| `turbo` | ~6GB | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡⚡ |

## Monitoring

```bash
# Service health
curl http://localhost:8000/health

# API logs
docker-compose logs -f whisper-api

# Worker logs  
docker-compose logs -f whisper-worker

# Redis stats
docker exec whisper-redis redis-cli info

# Performance metrics
curl http://localhost:8000/calibration
```

## Auto-Calibration System

The system automatically learns and adapts to your hardware performance:

- **Baseline**: Starts with `SPEED_MULTIPLIER=24.0` (24x faster than real-time)
- **Learning**: Collects performance data from each completed transcription
- **Adaptation**: After 3+ jobs, automatically adjusts speed estimates
- **Persistence**: Calibration data survives container restarts (Redis volume)

### Manual Performance Check

```bash
# Get job performance metrics
JOB_ID="your-job-id-here"
curl -s http://localhost:8000/result/$JOB_ID | python3 -c "
import json,sys
d = json.load(sys.stdin)
print(f'Audio duration: {d[\"audio_duration\"]:.1f}s')
print(f'Processing time: {d[\"processing_time\"]:.1f}s') 
print(f'Speed: {d[\"audio_duration\"]/d[\"processing_time\"]:.1f}x faster than real-time')
"
```

## Integration Examples

### PHP Integration
```php
<?php
// Upload file with language
$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, 'http://localhost:8000/transcribe');
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, [
    'file' => new CURLFile($audioPath, 'audio/mpeg', 'audio.mp3'),
    'language' => 'en'  // Optional: en, pl, de, fr, etc. (null = auto-detect)
]);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
$data = json_decode($response, true);
$jobId = $data['job_id'];

// Check status with progress
do {
    sleep(5);
    $status = json_decode(
        file_get_contents("http://localhost:8000/status/$jobId"), 
        true
    );
    echo "Progress: " . $status['progress'] . "\n";
} while($status['status'] !== 'completed');

// Get result
$result = json_decode(
    file_get_contents("http://localhost:8000/result/$jobId"), 
    true
);

echo $result['text']; // Full transcript
// Download VTT: http://localhost:8000/result/$jobId/vtt
?>
```

### Python Integration
```python
import requests
import time

# Upload file
with open('audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/transcribe',
        files={'file': f},
        data={'language': 'en'}
    )
job_id = response.json()['job_id']

# Monitor progress
while True:
    status = requests.get(f'http://localhost:8000/status/{job_id}').json()
    print(f"Status: {status['status']} - Progress: {status.get('progress', 'N/A')}")
    
    if status['status'] == 'completed':
        break
    elif status['status'] == 'failed':
        print(f"Error: {status['error']}")
        break
        
    time.sleep(5)

# Get results
result = requests.get(f'http://localhost:8000/result/{job_id}').json()
print(f"Transcript: {result['text']}")
print(f"Language: {result['language']}")
print(f"Duration: {result['audio_duration']}s")
print(f"Processing time: {result['processing_time']}s")
```

## Performance

With **8GB+ VRAM** GPU and **turbo** model:
- ⚡ **~24x real-time** (1h audio = 2.5min processing)
- 🎯 **Highest quality** transcription with better Polish language support
- 🚀 **4x faster** than large-v3 with similar accuracy
- 🔥 **Parallel processing** of multiple files
- 📊 **Accurate timestamps** to 0.01s precision
- 🧠 **Self-improving** speed estimates

## Troubleshooting

```bash
# Check GPU
nvidia-smi

# Check containers
docker-compose ps

# Restart services
docker-compose restart

# Full reset (clears Redis data)
docker-compose down -v
docker-compose up -d

# View calibration status
curl http://localhost:8000/calibration
```

## Development Setup

For local development with LSP support:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn redis python-multipart aiofiles

# Pyright configuration is included (pyrightconfig.json)
```

## License

MIT License - feel free to use in commercial projects.