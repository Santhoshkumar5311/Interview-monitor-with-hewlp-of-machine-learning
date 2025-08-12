# Google Cloud Speech-to-Text Setup Guide

This guide will help you set up Google Cloud credentials for the Speech-to-Text API, which is required for live transcription functionality.

## Prerequisites

1. **Google Account**: You need a Google account
2. **Google Cloud Project**: A Google Cloud project (free tier available)
3. **Python Environment**: Python 3.7+ with pip

## Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" at the top
3. Click "New Project"
4. Enter a project name (e.g., "interview-monitor")
5. Click "Create"

## Step 2: Enable Speech-to-Text API

1. In your project, go to "APIs & Services" > "Library"
2. Search for "Speech-to-Text API"
3. Click on "Cloud Speech-to-Text API"
4. Click "Enable"

## Step 3: Create Service Account

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service Account"
3. Fill in the details:
   - **Name**: `interview-monitor-transcription`
   - **Description**: `Service account for interview monitor transcription`
4. Click "Create and Continue"
5. For "Role", select "Cloud Speech-to-Text User"
6. Click "Continue" and then "Done"

## Step 4: Download Credentials

1. In the service accounts list, click on your new service account
2. Go to "Keys" tab
3. Click "Add Key" > "Create new key"
4. Choose "JSON" format
5. Click "Create"
6. The JSON file will download automatically

## Step 5: Set Up Credentials

### Option A: Environment Variable (Recommended)

1. Move the downloaded JSON file to a secure location
2. Set the environment variable:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\credentials.json"
```

**Windows (Command Prompt):**
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\credentials.json
```

**Linux/Mac:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```

### Option B: Python Code

You can also set the credentials in your Python code:

```python
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your/credentials.json'
```

## Step 6: Install Dependencies

Install the required Python packages:

```bash
pip install google-cloud-speech google-auth pyaudio
```

## Step 7: Test the Setup

Run the transcription test:

```bash
python test_transcription.py
```

## Troubleshooting

### Common Issues

1. **"No module named 'google.cloud"**
   - Solution: `pip install google-cloud-speech`

2. **"No module named 'pyaudio"**
   - Solution: `pip install pyaudio`
   - On Windows, you might need: `pip install pipwin` then `pipwin install pyaudio`

3. **"Authentication failed"**
   - Check if the credentials file path is correct
   - Verify the environment variable is set
   - Ensure the service account has the correct permissions

4. **"API not enabled"**
   - Go to Google Cloud Console and enable the Speech-to-Text API
   - Wait a few minutes for the API to activate

### PyAudio Installation Issues

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

## Security Notes

1. **Never commit credentials to version control**
2. **Keep your credentials file secure**
3. **Use environment variables when possible**
4. **Consider using Google Cloud's built-in security features**

## Cost Information

- **Free Tier**: 60 minutes per month of audio transcription
- **Paid**: $0.006 per 15 seconds after free tier
- **Real-time**: Same pricing as batch transcription

## Next Steps

Once your credentials are set up:

1. Test the transcription system: `python test_transcription.py`
2. Run the complete interview monitor: `python src/interview_monitor.py`
3. Speak into your microphone to see live transcription

## Support

If you encounter issues:

1. Check the [Google Cloud Speech-to-Text documentation](https://cloud.google.com/speech-to-text/docs)
2. Verify your credentials and API access
3. Check the console logs for detailed error messages
4. Ensure your microphone is working and accessible

---

**Note**: The free tier is sufficient for testing and development. For production use, consider setting up billing alerts and monitoring usage.
