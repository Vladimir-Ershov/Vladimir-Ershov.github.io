# Quick Start Guide - Toy VLA System

## Overview
This is a toy Vision-Language-Action system that interprets visual scenes and executes action commands based on natural language input using Google Gemini.

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
cd /home/not7/src/Vladimir-Ershov.github.io/projects/ToyVLA

# Activate your environment
conda activate /mnt/z/wsl/env/behavior
# or: source /mnt/z/wsl/env/behavior/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Google API key
nano .env  # or use your favorite editor
```

Add your Google API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Create Test Images
```bash
# Generate sample test images
python create_test_image.py
```

This creates `test_images/scene_cubes.png` and `test_images/scene_bottles.png`

### 4. Run Tests (Optional)
```bash
pytest test_system.py -v
```

### 5. Start the Server
```bash
# Option 1: Using the helper script
./run_server.sh

# Option 2: Direct Python
python main.py
```

The server will start at `http://localhost:8000`

### 6. Test the API

**Option A: Using the test client**
```bash
# In a new terminal
python test_client.py
```

**Option B: Using curl**
```bash
# Check status
curl http://localhost:8000/status

# Execute a task
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "command": "Pick up the red cube",
    "use_structured_output": true
  }'
```

**Option C: Using the interactive API docs**
Open your browser to `http://localhost:8000/docs`

## 📝 Example Commands

Try these commands with the test images:

1. `"Pick up the red cube"`
2. `"Move the red cube next to the blue sphere"`
3. `"Count all the yellow objects"`
4. `"Push the green block to the left"`
5. `"Point to the blue sphere"`

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    REST API (FastAPI)                    │
│                      [main.py]                           │
└─────────────────┬───────────────────────┬───────────────┘
                  │                       │
                  ▼                       ▼
        ┌──────────────────┐    ┌──────────────────┐
        │ MemorySubsystem  │    │    AIFacade      │
        │                  │    │                  │
        │ - Image buffer   │    │ - Gemini API     │
        │ - Async reading  │    │ - Structured out │
        └──────────────────┘    └──────────────────┘
                  │                       │
                  ▼                       ▼
        ┌──────────────────┐    ┌──────────────────┐
        │  test_images/    │    │   Thesaurus      │
        │                  │    │                  │
        │ - scene_cubes    │    │ - ActionType     │
        │ - scene_bottles  │    │ - SkillType      │
        └──────────────────┘    │ - FailureType    │
                                 │ - ActionOutput   │
                                 └──────────────────┘
```

## 📦 File Structure

```
ToyVLA/
├── main.py                 # FastAPI server & main loop
├── memory_subsystem.py     # Async image buffer (30 rotating images)
├── ai_facade.py           # Gemini API integration
├── thesaurus.py           # Enums & structured output models
├── task.py                # Task object with context
├── test_system.py         # Comprehensive test suite
├── test_client.py         # API test client
├── create_test_image.py   # Generate test images
├── run_server.sh          # Server startup script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from .env.example)
├── .env.example          # Example environment config
├── .gitignore            # Git ignore rules
├── README.md             # Full documentation
└── QUICKSTART.md         # This file
```

## 🔧 Configuration

Edit `.env` to customize:

```bash
# Required
GOOGLE_API_KEY=your_key_here

# Optional (with defaults)
IMAGE_FOLDER_PATH=./test_images    # Where to read images from
MAX_BUFFER_SIZE=30                 # Max images in buffer
SCAN_INTERVAL=1.0                  # Folder scan interval (seconds)
API_HOST=0.0.0.0                   # API server host
API_PORT=8000                      # API server port
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest test_system.py -v

# Run specific test class
pytest test_system.py::TestMemorySubsystem -v
pytest test_system.py::TestAIFacade -v

# Run with coverage
pytest test_system.py --cov=. --cov-report=html
```

### Integration Test
```bash
# Start server in one terminal
python main.py

# Run test client in another terminal
python test_client.py
```

## 🎯 API Endpoints

### GET `/`
System information and available endpoints

### GET `/status`
Returns:
```json
{
  "status": "running",
  "buffer_size": 2,
  "api_key_configured": true
}
```

### POST `/execute`
Execute a VLA task

Request:
```json
{
  "command": "Pick up the red cube",
  "context": "Optional context string",
  "use_structured_output": true
}
```

Response:
```json
{
  "task_id": "uuid",
  "command": "Pick up the red cube",
  "context": null,
  "image_path": "/path/to/image.png",
  "output": {
    "action": "pick_up",
    "target": "red cube",
    "destination": null,
    "completion": 15.0,
    "current_skill": "object_detection",
    "failure": "none",
    "reasoning": "Red cube identified in the scene"
  },
  "error": null
}
```

### POST `/clear_buffer`
Clear the image buffer

## 🐛 Troubleshooting

### "No images available in buffer"
- Check that `test_images/` folder exists and contains images
- Wait 1-2 seconds for the memory subsystem to load images
- Check logs for image loading errors

### "GOOGLE_API_KEY not set"
- Ensure `.env` file exists and contains `GOOGLE_API_KEY=your_key`
- Verify the API key is valid
- Check that Gemini API is enabled in your Google Cloud project

### Import errors
- Activate the correct environment: `conda activate /mnt/z/wsl/env/behavior`
- Reinstall dependencies: `pip install -r requirements.txt`

### Port already in use
- Change `API_PORT` in `.env`
- Or stop the existing server

## 📚 Next Steps

1. **Add your own images**: Place images in `test_images/` folder
2. **Customize actions**: Edit enums in [thesaurus.py](thesaurus.py:9)
3. **Modify prompts**: Update system prompt in [ai_facade.py](ai_facade.py:20)
4. **Extend API**: Add endpoints in [main.py](main.py:50)

## 📄 License
MIT License - Educational/Demo purposes

---

**Ready to test?** Just run `./run_server.sh` and then `python test_client.py` in another terminal!
