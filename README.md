# Enhanced Interview Monitor with Machine Learning

A comprehensive interview monitoring system that combines computer vision, speech analysis, and sentiment analysis to assess candidate confidence and performance in real-time.

## 🚀 Features

### Core Functionality
- **Real-time Facial Expression Tracking**: Detects smile, frown, raised eyebrows, eye movement, and head position
- **Live Speech Transcription**: Streaming ASR with Whisper for 5-second audio chunks
- **Advanced Sentiment Analysis**: Emotion, toxicity, politeness, and hedge detection using Transformers
- **Confidence Classification**: Multi-modal fusion of facial, speech, and sentiment metrics
- **Eye Contact Tracking**: Advanced gaze analysis with MediaPipe and FER

### Enhanced Analysis
- **Audio Prosody Analysis**: Pitch, rate, pauses, and spectral features using pyAudioAnalysis
- **Facial Emotion Recognition**: FER integration for advanced emotion detection
- **Fusion Layer**: EMA smoothing and per-session normalization for robust scoring
- **Real-time Metrics**: Confidence, Relevance, and Sentiment scores

### User Interface
- **PyQt HUD Interface**: Live meters, alerts, and scrollable timeline heatmap
- **Clean Video Display**: No distracting landmarks or text overlays
- **Information Dashboard**: FPS, frame count, transcription status, and confidence levels
- **Alert System**: Real-time notifications for low confidence, toxicity, etc.

### Data & Deployment
- **BigQuery Integration**: Session tracking with `session_id`, `t_start`, `t_end`
- **Docker Deployment**: Complete containerized setup with monitoring
- **Performance Logging**: Comprehensive metrics and export functionality
- **Modular Architecture**: Clear interfaces and service separation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │    │   Audio Input   │    │   Text Input    │
│   (OpenCV)      │    │   (PyAudio)     │    │   (Whisper)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Facial Analysis │    │ Audio Prosody   │    │ Sentiment      │
│ (MediaPipe+FER) │    │ (pyAudioAnalysis)│   │ (Transformers) │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │   Fusion Layer      │
                    │ (EMA + Normalization)│
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Confidence        │
                    │   Classifier        │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   BigQuery Logger   │
                    │   + HUD Display     │
                    └─────────────────────┘
```

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Camera**: USB webcam or built-in camera
- **Microphone**: Built-in or external microphone

### Python Requirements
- **Python**: 3.8 - 3.11
- **CUDA**: Optional (for GPU acceleration)

## 🛠️ Installation

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/interview-monitor.git
   cd interview-monitor
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your GCP credentials and settings
   ```

3. **Start the system**
   ```bash
   docker-compose up -d
   ```

4. **Access the interface**
   - Main monitor: http://localhost:8000
   - HUD interface: http://localhost:3000
   - Grafana dashboard: http://localhost:3001 (admin/admin)

### Option 2: Local Installation

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies (Linux)**
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
       libopencv-dev \
       portaudio19-dev \
       libasound2-dev \
       python3-pyqt5 \
       python3-pyqt5.qtchart
   ```

4. **Set up GCP credentials**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account.json"
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# GCP Configuration
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./credentials/service-account.json
USE_BIGQUERY=true

# Database Configuration
POSTGRES_PASSWORD=secure_password
REDIS_URL=redis://localhost:6379

# Application Settings
LOG_LEVEL=INFO
CAMERA_INDEX=0
AUDIO_DEVICE_INDEX=0
SAMPLE_RATE=16000
CHUNK_DURATION=5.0

# Model Configuration
WHISPER_MODEL_SIZE=base
USE_FER=true
EMOTION_THRESHOLD=0.3
EYE_CONTACT_THRESHOLD=0.7

# UI Configuration
HUD_ENABLED=true
WEB_INTERFACE_ENABLED=true
```

### GCP Setup

1. **Create a Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable APIs**
   - Cloud Speech-to-Text API
   - BigQuery API
   - Cloud Storage API

3. **Create Service Account**
   - Go to IAM & Admin > Service Accounts
   - Create new service account
   - Download JSON key file
   - Place in `./credentials/service-account.json`

4. **Set BigQuery permissions**
   - BigQuery Data Editor
   - BigQuery Job User

## 🚀 Usage

### Basic Usage

1. **Start the monitor**
   ```bash
   python src/interview_monitor.py
   ```

2. **Start the HUD interface**
   ```bash
   python src/ui/interview_hud.py
   ```

3. **Use controls**
   - `q`: Quit monitoring
   - `s`: Save session data
   - `h`: Show help

### Advanced Usage

1. **Custom configuration**
   ```python
   from src.interview_monitor import EnhancedInterviewMonitor
   
   monitor = EnhancedInterviewMonitor(
       use_whisper=True,
       use_fer=True,
       use_bigquery=True,
       gcp_project_id="your-project-id"
   )
   monitor.run()
   ```

2. **Custom metrics**
   ```python
   from src.confidence_analysis.confidence_classifier import ConfidenceClassifier
   
   classifier = ConfidenceClassifier(
       ema_alpha=0.1,
       confidence_threshold=0.7
   )
   ```

## 📊 Data Structure

### BigQuery Tables

#### Sessions Table
```sql
CREATE TABLE `sessions` (
    session_id STRING REQUIRED,
    start_time TIMESTAMP REQUIRED,
    end_time TIMESTAMP,
    user_id STRING,
    interview_type STRING,
    duration_minutes FLOAT64,
    status STRING REQUIRED,
    created_at TIMESTAMP REQUIRED,
    updated_at TIMESTAMP REQUIRED
);
```

#### Features Table
```sql
CREATE TABLE `features` (
    session_id STRING REQUIRED,
    t_start TIMESTAMP REQUIRED,
    t_end TIMESTAMP REQUIRED,
    feature_type STRING REQUIRED,
    feature_data JSON REQUIRED,
    confidence FLOAT64 REQUIRED,
    quality_score FLOAT64 REQUIRED,
    created_at TIMESTAMP REQUIRED
);
```

### Data Export

The system exports data in multiple formats:
- **Performance Logs**: JSONL format with detailed metrics
- **Transcripts**: JSON format with timestamps and confidence
- **Session Summaries**: Aggregated statistics and insights

## 🔍 Monitoring & Analytics

### Prometheus Metrics
- `interview_confidence_score`
- `interview_relevance_score`
- `interview_sentiment_score`
- `facial_detection_fps`
- `speech_processing_latency`

### Grafana Dashboards
- Real-time confidence trends
- Feature quality metrics
- Session performance analysis
- System resource usage

## 🧪 Testing

### Run Tests
```bash
# Test facial detection
python test_facial_detection.py

# Test confidence analysis
python test_confidence_analysis.py

# Test transcription
python test_transcription.py

# Test all components
python -m pytest tests/
```

### Performance Testing
```bash
# Benchmark facial analysis
python -m src.facial_detection.enhanced_face_analyzer --benchmark

# Test audio processing
python -m src.confidence_analysis.audio_prosody_analyzer --test-audio
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera permissions
   - Verify camera index in configuration
   - Test with `cv2.VideoCapture(0)`

2. **Audio not working**
   - Check microphone permissions
   - Verify audio device index
   - Test with PyAudio examples

3. **GCP authentication errors**
   - Verify service account JSON path
   - Check API enablement
   - Verify project ID

4. **Performance issues**
   - Reduce Whisper model size
   - Disable FER if not needed
   - Use GPU acceleration if available

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python src/interview_monitor.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/
black src/

# Run tests with coverage
pytest --cov=src tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **MediaPipe** for facial landmark detection
- **OpenAI Whisper** for speech recognition
- **Hugging Face** for transformer models
- **Google Cloud** for BigQuery and Speech-to-Text

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/interview-monitor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/interview-monitor/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/interview-monitor/wiki)

## 🔄 Changelog

### v2.0.0 (Current)
- ✨ Whisper ASR integration
- ✨ Audio prosody analysis
- ✨ Enhanced facial analysis with FER
- ✨ PyQt HUD interface
- ✨ BigQuery logging
- ✨ Docker deployment
- ✨ Performance monitoring

### v1.0.0
- ✨ Basic facial expression detection
- ✨ Google Speech-to-Text integration
- ✨ Confidence classification
- ✨ Basic sentiment analysis

---

**Made with ❤️ for better interview experiences**