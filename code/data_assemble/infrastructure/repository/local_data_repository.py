import os
import json
import cv2
import time
import threading
import queue
from typing import Any
from pathlib import Path
from internal.interface.repository.data_repository_interface import DataRepositoryInterface
from internal.domain.value_object.telemetry import Telemetry
from app.config.settings import Settings

class LocalDataRepository(DataRepositoryInterface):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.out_dir = Path(settings.out_dir)
        self.image_dir = self.out_dir / "images"
        self.log_file = self.out_dir / "log.jsonl"

        # Ensure directories exist
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # Background worker for async saving
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def save(self, telemetry: Telemetry, image: Any) -> None:
        """Enqueues data for background saving."""
        # Note: We should copy the image to avoid it being overwritten by the next frame if needed,
        # but in this architecture DriveService captures a new frame each loop, so we're safe 
        # as long as we don't modify it in place.
        self._queue.put((telemetry, image))

    def _worker(self):
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                telemetry, image = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # 1. Save Image
            image_path = self.image_dir / telemetry.image_filename
            if image is not None:
                cv2.imwrite(str(image_path), image)

            # 2. Save Telemetry to JSONL
            with open(self.log_file, "a", encoding="utf-8") as f:
                line = json.dumps(telemetry.to_dict(), ensure_ascii=False)
                f.write(line + "\n")
            
            self._queue.task_done()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)
