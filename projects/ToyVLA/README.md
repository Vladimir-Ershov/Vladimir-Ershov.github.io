# Toy VLA (Vision-Language-Action) System

A toy system that interprets visual scenes and executes action commands based on natural language input, powered by Google Gemini Vision-Language Model.

## Features

- **Memory Subsystem**: Asynchronous image buffer that continuously monitors a folder for new images
- **AI Facade**: Integration with Google Gemini for vision-language understanding
- **Structured Output**: Type-safe action predictions with enums for actions, skills, and failure modes
- **REST API**: FastAPI-based endpoint for executing commands
- **Task Management**: Structured task objects with context and output tracking

## Architecture

### Components

1. **MemorySubsystem** (`memory_subsystem.py`)
   - Concurrent buffer with rotating deque (keeps 30 latest images)
   - Background thread asynchronously reads images from a folder
   - Thread-safe operations with locking
   - `get_latest()` method for retrieving the most recent image

2. **Thesaurus** (`thesaurus.py`)
   - Enums for ActionType, SkillType, FailureType
   - Pydantic models for structured outputs
   - Ensures mandatory fields: action, target, destination, completion, current_skill, failure

3. **AIFacade** (`ai_facade.py`)
   - Integration with Google Gemini API
   - Supports both regular and structured output prediction
   - Handles JSON parsing and validation
   - Pretends execution is in progress (toy example)

4. **Task** (`task.py`)
   - Encapsulates command, context, image, and output
   - Tracks task lifecycle (created_at, completed_at)
   - Serialization support for API responses

5. **Main Loop** (`main.py`)
   - FastAPI REST endpoint
   - Pulls latest image from memory
   - Sends to AI Facade for prediction
   - Returns structured action output

## Setup

### Prerequisites

- Python 3.9+
- Google API Key for Gemini

### Installation

1. Clone the repository and navigate to the project:
```bash
cd /home/not7/src/Vladimir-Ershov.github.io/projects/ToyVLA
```

2. Activate your environment:
```bash
conda activate /mnt/z/wsl/env/behavior
# or
source /mnt/z/wsl/env/behavior/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
```

5. Edit `.env` and add your Google API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
IMAGE_FOLDER_PATH=./test_images
```

6. Create test images directory:
```bash
mkdir -p test_images
```

## Usage

### Running the Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`

### API Endpoints

#### 1. Get System Status
```bash
curl http://localhost:8000/status
```

#### 2. Execute a Task
```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "command": "Pick up the red cube",
    "context": "There are multiple objects on the table",
    "use_structured_output": true
  }'
```

Response:
```json
{
  "task_id": "uuid",
  "command": "Pick up the red cube",
  "context": "There are multiple objects on the table",
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

#### 3. Clear Image Buffer
```bash
curl -X POST http://localhost:8000/clear_buffer
```

### Example Commands

- "Move the red cube next to the blue sphere."
- "Pick up the green block."
- "Count all the yellow objects."
- "Push the bottle to the left."

## Testing

Run the test suite:
```bash
pytest test_system.py -v
```

Run specific test class:
```bash
pytest test_system.py::TestMemorySubsystem -v
```

## Configuration

Environment variables (set in `.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `IMAGE_FOLDER_PATH` | Path to images folder | `./test_images` |
| `MAX_BUFFER_SIZE` | Max images in buffer | `30` |
| `SCAN_INTERVAL` | Folder scan interval (seconds) | `1.0` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |

## Project Structure

```
ToyVLA/
├── main.py                 # FastAPI application and main loop
├── memory_subsystem.py     # Asynchronous image buffer
├── ai_facade.py           # Gemini API integration
├── thesaurus.py           # Enums and structured outputs
├── task.py                # Task object definition
├── test_system.py         # Test suite
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment config
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Development Notes

### Adding New Actions

1. Add to `ActionType` enum in `thesaurus.py`
2. Update schema in `ai_facade.py`
3. Update prompt examples

### Adding New Skills or Failures

1. Add to respective enum in `thesaurus.py`
2. Update schema in `ai_facade.py`

### Using Structured Output

The system supports Gemini's structured output feature (response_schema). Set `use_structured_output: true` in the API request to use it. This provides more reliable JSON parsing.

## Troubleshooting

### No images in buffer
- Ensure `test_images` folder exists
- Add some test images to the folder
- Check logs for image loading errors

### API key errors
- Verify `GOOGLE_API_KEY` is set in `.env`
- Check API key is valid and has Gemini API enabled

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using the correct Python environment

## License

MIT License - Educational/Demo purposes
