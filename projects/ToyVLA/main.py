"""
Main application with REST API for the Toy VLA system.
"""
import os
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from memory_subsystem import MemorySubsystem
from ai_facade import AIFacade
from task import Task
from thesaurus import ActionOutput

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
memory_subsystem: Optional[MemorySubsystem] = None
ai_facade: Optional[AIFacade] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    global memory_subsystem, ai_facade

    # Startup
    logger.info("Starting Toy VLA System...")

    # Initialize Memory Subsystem
    image_folder = os.environ.get("IMAGE_FOLDER_PATH", "./test_images")
    max_buffer = int(os.environ.get("MAX_BUFFER_SIZE", "30"))
    scan_interval = float(os.environ.get("SCAN_INTERVAL", "1.0"))

    memory_subsystem = MemorySubsystem(
        image_folder_path=image_folder,
        max_buffer_size=max_buffer,
        scan_interval=scan_interval
    )
    memory_subsystem.start()

    # Initialize AI Facade
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set - AI predictions will fail!")
    else:
        logger.info("GOOGLE_API_KEY found, initializing AI Facade")

    ai_facade = AIFacade(api_key=api_key)

    logger.info("Toy VLA System started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Toy VLA System...")
    if memory_subsystem:
        memory_subsystem.stop()
    logger.info("Toy VLA System shut down")


app = FastAPI(
    title="Toy VLA System",
    description="A toy Vision-Language-Action system for robotic manipulation",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class TaskRequest(BaseModel):
    command: str
    context: Optional[str] = None
    use_structured_output: bool = True


class TaskResponse(BaseModel):
    task_id: str
    command: str
    context: Optional[str]
    image_path: Optional[str]
    output: Optional[dict]
    error: Optional[str]


class SystemStatus(BaseModel):
    status: str
    buffer_size: int
    api_key_configured: bool


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Toy VLA System",
        "version": "1.0.0",
        "description": "Vision-Language-Action system for robotic manipulation",
        "endpoints": {
            "POST /execute": "Execute a task with natural language command",
            "GET /status": "Get system status",
            "POST /clear_buffer": "Clear the image buffer"
        }
    }


@app.get("/status", tags=["System"], response_model=SystemStatus)
async def get_status():
    """Get system status."""
    if not memory_subsystem or not ai_facade:
        raise HTTPException(status_code=500, detail="System not initialized")

    return SystemStatus(
        status="running",
        buffer_size=memory_subsystem.get_buffer_size(),
        api_key_configured=bool(os.environ.get("GOOGLE_API_KEY"))
    )


@app.post("/execute", tags=["Tasks"], response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """
    Execute a VLA task with the given natural language command.

    The system will:
    1. Retrieve the latest image from the memory buffer
    2. Send the image and command to the AI model
    3. Return the structured action prediction
    """
    if not memory_subsystem or not ai_facade:
        raise HTTPException(status_code=500, detail="System not initialized")

    # Create task
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        command=request.command,
        context=request.context
    )

    logger.info(f"Executing task {task_id}: {request.command}")

    try:
        # Get latest image from memory
        latest = memory_subsystem.get_latest()
        if latest is None:
            raise HTTPException(
                status_code=404,
                detail="No images available in buffer. Please add images to the image folder."
            )

        image_path, image = latest
        task.image = image
        task.image_path = image_path

        logger.info(f"Using image: {image_path}")

        # Predict action using AI Facade
        if request.use_structured_output:
            output = ai_facade.predict_action_with_schema(
                image=image,
                command=request.command,
                context=request.context
            )
        else:
            output = ai_facade.predict_action(
                image=image,
                command=request.command,
                context=request.context
            )

        task.mark_completed(output)

        logger.info(f"Task {task_id} completed: {output.action} on {output.target}")

        return TaskResponse(
            task_id=task.task_id,
            command=task.command,
            context=task.context,
            image_path=task.image_path,
            output=output.model_dump(),
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing task {task_id}: {e}", exc_info=True)
        task.mark_failed(str(e))
        return TaskResponse(
            task_id=task.task_id,
            command=task.command,
            context=task.context,
            image_path=task.image_path,
            output=None,
            error=str(e)
        )


@app.post("/clear_buffer", tags=["System"])
async def clear_buffer():
    """Clear the image buffer."""
    if not memory_subsystem:
        raise HTTPException(status_code=500, detail="System not initialized")

    memory_subsystem.clear_buffer()
    return {"message": "Image buffer cleared successfully"}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
