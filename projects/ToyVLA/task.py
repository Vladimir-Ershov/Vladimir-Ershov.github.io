"""
Task object for encapsulating VLA task context and execution.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from PIL import Image
from thesaurus import ActionOutput


@dataclass
class Task:
    """
    Represents a vision-language-action task to be executed.

    Encapsulates the command, context, input image, and output prediction.
    """
    task_id: str
    command: str
    context: Optional[str] = None
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None
    output: Optional[ActionOutput] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def mark_completed(self, output: ActionOutput):
        """Mark task as completed with the given output."""
        self.output = output
        self.completed_at = datetime.now()

    def mark_failed(self, error: str):
        """Mark task as failed with the given error."""
        self.error = error
        self.completed_at = datetime.now()

    def to_dict(self):
        """Convert task to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "command": self.command,
            "context": self.context,
            "image_path": self.image_path,
            "output": self.output.model_dump() if self.output else None,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error
        }
