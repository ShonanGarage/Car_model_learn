import os
import json
import csv
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
        self.csv_file = self.out_dir / "log.csv"

        # Ensure directories exist
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(self.out_dir, os.W_OK):
            raise PermissionError(
                f"Output directory is not writable: {self.out_dir}. "
                "Update settings.out_dir to a writable path."
            )
        # Touch log file to fail fast on permission issues
        with open(self.log_file, "a", encoding="utf-8"):
            pass
        # Touch csv file to fail fast on permission issues
        with open(self.csv_file, "a", encoding="utf-8"):
            pass

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

            # 3. Save Telemetry to CSV
            row = telemetry.to_dict()
            distances = row.pop("distances", [])
            for i, d in enumerate(distances):
                row[f"distance_{i}"] = d

            write_header = not self.csv_file.exists() or self.csv_file.stat().st_size == 0
            with open(self.csv_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            
            self._queue.task_done()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)
