"""
Tests for the Toy VLA system.
"""
import os
import time
import tempfile
import shutil
from pathlib import Path
import pytest
from PIL import Image
from unittest.mock import Mock, patch

from memory_subsystem import MemorySubsystem
from thesaurus import ActionType, SkillType, FailureType, ActionOutput
from task import Task


class TestThesaurus:
    """Test the Thesaurus enums and ActionOutput."""

    def test_action_type_enum(self):
        """Test ActionType enum values."""
        assert ActionType.MOVE == "move"
        assert ActionType.PICK_UP == "pick_up"
        assert ActionType.COUNT == "count"

    def test_skill_type_enum(self):
        """Test SkillType enum values."""
        assert SkillType.OBJECT_DETECTION == "object_detection"
        assert SkillType.MANIPULATION == "manipulation"

    def test_failure_type_enum(self):
        """Test FailureType enum values."""
        assert FailureType.OBJECT_NOT_FOUND == "object_not_found"
        assert FailureType.NONE == "none"

    def test_action_output_validation(self):
        """Test ActionOutput creation and validation."""
        output = ActionOutput(
            action=ActionType.MOVE,
            target="red cube",
            destination="blue sphere",
            completion=50.0,
            current_skill=SkillType.MANIPULATION,
            failure=FailureType.NONE,
            reasoning="Moving object as requested"
        )

        assert output.action == ActionType.MOVE
        assert output.target == "red cube"
        assert output.completion == 50.0

    def test_action_output_completion_validation(self):
        """Test that completion percentage is validated."""
        # Valid completion
        output = ActionOutput(
            action=ActionType.PICK_UP,
            target="object",
            completion=75.0,
            current_skill=SkillType.OBJECT_DETECTION,
            failure=FailureType.NONE
        )
        assert output.completion == 75.0

        # Invalid completion (should raise validation error)
        with pytest.raises(Exception):  # Pydantic validation error
            ActionOutput(
                action=ActionType.PICK_UP,
                target="object",
                completion=150.0,  # Invalid: > 100
                current_skill=SkillType.OBJECT_DETECTION,
                failure=FailureType.NONE
            )


class TestTask:
    """Test the Task object."""

    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            task_id="test-123",
            command="Pick up the red cube"
        )

        assert task.task_id == "test-123"
        assert task.command == "Pick up the red cube"
        assert task.output is None
        assert task.error is None

    def test_task_mark_completed(self):
        """Test marking a task as completed."""
        task = Task(task_id="test-123", command="test")
        output = ActionOutput(
            action=ActionType.PICK_UP,
            target="red cube",
            completion=100.0,
            current_skill=SkillType.MANIPULATION,
            failure=FailureType.NONE
        )

        task.mark_completed(output)

        assert task.output == output
        assert task.completed_at is not None
        assert task.error is None

    def test_task_mark_failed(self):
        """Test marking a task as failed."""
        task = Task(task_id="test-123", command="test")
        task.mark_failed("Test error")

        assert task.error == "Test error"
        assert task.completed_at is not None
        assert task.output is None

    def test_task_to_dict(self):
        """Test converting task to dictionary."""
        task = Task(
            task_id="test-123",
            command="test command",
            context="test context"
        )
        output = ActionOutput(
            action=ActionType.MOVE,
            target="object",
            completion=50.0,
            current_skill=SkillType.SPATIAL_REASONING,
            failure=FailureType.NONE
        )
        task.mark_completed(output)

        task_dict = task.to_dict()

        assert task_dict["task_id"] == "test-123"
        assert task_dict["command"] == "test command"
        assert task_dict["output"]["action"] == "move"


class TestMemorySubsystem:
    """Test the MemorySubsystem."""

    @pytest.fixture
    def temp_image_dir(self):
        """Create a temporary directory for test images."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        img = Image.new('RGB', (100, 100), color='red')
        return img

    def test_memory_subsystem_initialization(self, temp_image_dir):
        """Test initializing the memory subsystem."""
        memory = MemorySubsystem(
            image_folder_path=temp_image_dir,
            max_buffer_size=30,
            scan_interval=1.0
        )

        assert memory.image_folder_path == Path(temp_image_dir)
        assert memory.max_buffer_size == 30
        assert memory.get_buffer_size() == 0

    def test_memory_subsystem_read_images(self, temp_image_dir, sample_image):
        """Test reading images from folder."""
        # Save a test image
        image_path = Path(temp_image_dir) / "test_image.png"
        sample_image.save(image_path)

        # Create and start memory subsystem
        memory = MemorySubsystem(
            image_folder_path=temp_image_dir,
            max_buffer_size=30,
            scan_interval=0.5
        )

        with memory:
            # Wait for image to be loaded
            time.sleep(1.5)

            # Check that image was loaded
            assert memory.get_buffer_size() > 0
            latest = memory.get_latest()
            assert latest is not None

            path, img = latest
            assert "test_image.png" in path
            assert img.size == (100, 100)

    def test_memory_subsystem_buffer_rotation(self, temp_image_dir, sample_image):
        """Test that buffer rotates when max size is reached."""
        # Create memory subsystem with small buffer
        memory = MemorySubsystem(
            image_folder_path=temp_image_dir,
            max_buffer_size=3,
            scan_interval=0.3
        )

        with memory:
            # Add multiple images
            for i in range(5):
                image_path = Path(temp_image_dir) / f"test_image_{i}.png"
                sample_image.save(image_path)
                time.sleep(0.1)

            # Wait for images to be loaded
            time.sleep(2.0)

            # Buffer should not exceed max size
            assert memory.get_buffer_size() <= 3

    def test_memory_subsystem_get_latest(self, temp_image_dir, sample_image):
        """Test getting the latest image."""
        memory = MemorySubsystem(
            image_folder_path=temp_image_dir,
            max_buffer_size=10,
            scan_interval=0.5
        )

        # No images yet
        assert memory.get_latest() is None

        with memory:
            # Add images
            for i in range(3):
                image_path = Path(temp_image_dir) / f"image_{i}.png"
                sample_image.save(image_path)
                time.sleep(0.2)

            # Wait for images to be loaded
            time.sleep(1.5)

            # Get latest should return the most recent image
            latest = memory.get_latest()
            assert latest is not None

    def test_memory_subsystem_clear_buffer(self, temp_image_dir, sample_image):
        """Test clearing the buffer."""
        image_path = Path(temp_image_dir) / "test.png"
        sample_image.save(image_path)

        memory = MemorySubsystem(
            image_folder_path=temp_image_dir,
            max_buffer_size=10,
            scan_interval=0.5
        )

        with memory:
            time.sleep(1.5)
            assert memory.get_buffer_size() > 0

            memory.clear_buffer()
            assert memory.get_buffer_size() == 0


class TestAIFacade:
    """Test the AI Facade (mocked, as we don't want to call real API in tests)."""

    @patch('ai_facade.genai')
    def test_ai_facade_initialization(self, mock_genai):
        """Test initializing the AI facade."""
        from ai_facade import AIFacade

        facade = AIFacade(api_key="test_key")
        assert facade.api_key == "test_key"
        mock_genai.configure.assert_called_once_with(api_key="test_key")

    def test_ai_facade_requires_api_key(self):
        """Test that AI facade requires an API key."""
        from ai_facade import AIFacade

        # Clear environment variable if set
        old_key = os.environ.get("GOOGLE_API_KEY")
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            AIFacade()

        # Restore
        if old_key:
            os.environ["GOOGLE_API_KEY"] = old_key

    @patch('ai_facade.genai')
    def test_ai_facade_predict_action_mock(self, mock_genai):
        """Test predicting action with mocked Gemini."""
        from ai_facade import AIFacade

        # Mock response
        mock_response = Mock()
        mock_response.text = '''```json
{
    "action": "pick_up",
    "target": "red cube",
    "destination": null,
    "completion": 10.0,
    "current_skill": "object_detection",
    "failure": "none",
    "reasoning": "Identified red cube in the scene"
}
```'''

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Test
        facade = AIFacade(api_key="test_key")
        image = Image.new('RGB', (100, 100), color='red')
        output = facade.predict_action(image, "Pick up the red cube")

        assert output.action == ActionType.PICK_UP
        assert output.target == "red cube"
        assert output.failure == FailureType.NONE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
