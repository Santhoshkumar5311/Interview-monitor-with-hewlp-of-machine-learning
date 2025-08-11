# Interview Monitor with Machine Learning

An intelligent interview monitoring system that uses computer vision and machine learning to assess candidate confidence and provide real-time transcription.

## Features

- **Live Facial Expression Tracking**: Detects smile, frown, raised eyebrows, eye movements, and head position
- **Confidence Level Classification**: AI-powered assessment of candidate confidence (High/Medium/Low)
- **Real-time Transcription**: Live English speech-to-text using Google Speech API
- **Web-based Interface**: Accessible through any web browser

## Project Structure

```
├── src/
│   ├── facial_detection/     # Facial expression detection models
│   ├── confidence_analysis/  # Confidence level classification
│   ├── transcription/        # Speech-to-text functionality
│   ├── web_interface/       # Flask web application
│   └── utils/               # Helper functions
├── models/                   # Trained ML models
├── data/                    # Training datasets
├── config/                  # Configuration files
└── tests/                   # Unit tests
```

## Setup Instructions

1. Install Python 3.8+ and pip
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Google Cloud credentials for Speech-to-Text
4. Run the application: `python src/web_interface/app.py`

## Development Status

🚧 **In Development** - Step by step implementation in progress