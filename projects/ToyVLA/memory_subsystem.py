"""
Memory Subsystem for managing a rotating buffer of images.
"""
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Deque
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemorySubsystem:
    """
    Manages a concurrent buffer of images with asynchronous reading from a folder.

    The system maintains a rotating buffer (deque) of the latest N images,
    with the most recent image accessible first.
    """

    def __init__(
        self,
        image_folder_path: str,
        max_buffer_size: int = 30,
        scan_interval: float = 1.0
    ):
        """
        Initialize the Memory Subsystem.

        Args:
            image_folder_path: Path to the folder containing images
            max_buffer_size: Maximum number of images to keep in buffer
            scan_interval: Time interval (seconds) between folder scans
        """
        self.image_folder_path = Path(image_folder_path)
        self.max_buffer_size = max_buffer_size
        self.scan_interval = scan_interval

        # Thread-safe deque with maxlen for automatic rotation
        self.image_buffer: Deque[tuple[str, Image.Image]] = deque(maxlen=max_buffer_size)

        # Threading control
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

        # Track processed files to avoid duplicates
        self._processed_files: set[str] = set()

        # Ensure folder exists
        self.image_folder_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"MemorySubsystem initialized with folder: {self.image_folder_path}")

    def start(self):
        """Start the background image reading thread."""
        if self._reader_thread is not None and self._reader_thread.is_alive():
            logger.warning("Reader thread already running")
            return

        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._read_images_loop, daemon=True)
        self._reader_thread.start()
        logger.info("Background image reader started")

    def stop(self):
        """Stop the background image reading thread."""
        if self._reader_thread is None:
            return

        self._stop_event.set()
        self._reader_thread.join(timeout=5.0)
        logger.info("Background image reader stopped")

    def _read_images_loop(self):
        """
        Background loop that continuously scans the folder for new images.
        """
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

        while not self._stop_event.is_set():
            try:
                # Get all image files sorted by modification time (newest first)
                image_files = []
                for file_path in self.image_folder_path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                        image_files.append(file_path)

                # Sort by modification time (newest first)
                image_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

                # Process new images
                for file_path in image_files:
                    file_key = f"{file_path.name}_{file_path.stat().st_mtime}"

                    if file_key not in self._processed_files:
                        try:
                            image = Image.open(file_path)
                            # Convert to RGB to ensure consistency
                            if image.mode != 'RGB':
                                image = image.convert('RGB')

                            with self._lock:
                                # Add to front of deque (most recent first)
                                self.image_buffer.appendleft((str(file_path), image.copy()))
                                self._processed_files.add(file_key)

                            logger.info(f"Loaded image: {file_path.name} (buffer size: {len(self.image_buffer)})")

                        except Exception as e:
                            logger.error(f"Error loading image {file_path}: {e}")

                # Clean up processed_files set if it gets too large
                if len(self._processed_files) > self.max_buffer_size * 2:
                    with self._lock:
                        current_files = {
                            f"{Path(path).name}_{Path(path).stat().st_mtime}"
                            for path, _ in self.image_buffer
                        }
                        self._processed_files = current_files

            except Exception as e:
                logger.error(f"Error in image reading loop: {e}")

            # Wait before next scan
            self._stop_event.wait(self.scan_interval)

    def get_latest(self) -> Optional[tuple[str, Image.Image]]:
        """
        Get the most recent image from the buffer.

        Returns:
            Tuple of (file_path, image) or None if buffer is empty
        """
        with self._lock:
            if len(self.image_buffer) > 0:
                return self.image_buffer[0]
            return None

    def get_buffer_size(self) -> int:
        """Get the current number of images in the buffer."""
        with self._lock:
            return len(self.image_buffer)

    def clear_buffer(self):
        """Clear all images from the buffer."""
        with self._lock:
            self.image_buffer.clear()
            self._processed_files.clear()
        logger.info("Image buffer cleared")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
