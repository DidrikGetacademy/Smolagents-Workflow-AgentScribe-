import pickle
import os
import threading
from queue import Queue
from utility.log import log
import json
import os
import threading
from queue import Queue
from utility.log import log

class PersistentVideoQueue:
    def __init__(self, filepath):
        self.filepath = filepath
        self.queue = Queue()
        self.in_progress = {}  # Track items being processed
        self.lock = threading.Lock()
        self.load_from_disk()

    def put(self, item):
        with self.lock:
            self.queue.put(item)
            self._save_to_disk()
            log(f"➕ Added new task to queue. Total: {self.qsize()} items")

    def get(self, timeout=None):
        item = self.queue.get(timeout=timeout)
        with self.lock:
            # Mark item as in-progress instead of losing it
            task_id = id(item)  # Unique identifier for this task
            self.in_progress[task_id] = item
            self._save_to_disk()
            log(f"🔄 Moved task {task_id} to in-progress. Queue: {self.queue.qsize()}, In-progress: {len(self.in_progress)}")
        return item, task_id  # Return both item and ID

    def task_done(self, task_id=None):
        self.queue.task_done()
        if task_id is not None:
            with self.lock:
                # Remove from in-progress when completed
                if task_id in self.in_progress:
                    del self.in_progress[task_id]
                    self._save_to_disk()
                    log(f"✅ Task {task_id} completed and removed from storage. Remaining: {self.qsize()} items")
                else:
                    log(f"⚠️ Task {task_id} not found in in-progress (already completed?)")

    def empty(self):
        return self.queue.empty() and len(self.in_progress) == 0

    def qsize(self):
        return self.queue.qsize() + len(self.in_progress)

    def _save_to_disk(self):
        """Save both queued and in-progress items as JSON"""
        try:
            # Get queued items and convert tuples to lists
            queued_items = []
            temp_items = []

            while not self.queue.empty():
                try:
                    item = self.queue.get_nowait()
                    if item is not None:
                        # Convert tuple to list for JSON serialization
                        json_item = list(item) if isinstance(item, tuple) else item
                        queued_items.append(json_item)
                        temp_items.append(item)  # Keep original format for queue
                except:
                    break

            # Put original items back in queue
            for item in temp_items:
                self.queue.put(item)

            # Convert in_progress items to JSON format
            in_progress_items = []
            for task_id, item in self.in_progress.items():
                json_item = list(item) if isinstance(item, tuple) else item
                in_progress_items.append({
                    "task_id": task_id,
                    "data": json_item
                })

            # Create JSON structure
            all_items = {
                "metadata": {
                    "total_items": len(queued_items) + len(in_progress_items),
                    "queued_count": len(queued_items),
                    "in_progress_count": len(in_progress_items),
                    "format_version": "1.0"
                },
                "queued": queued_items,
                "in_progress": in_progress_items
            }

            total_items = len(queued_items) + len(in_progress_items)

            if total_items > 0:
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    json.dump(all_items, f, indent=2, ensure_ascii=False)
                log(f"💾 JSON backup updated: {len(queued_items)} queued + {len(in_progress_items)} in-progress")


        except Exception as e:
            log(f"❌ Error saving JSON queue to disk: {str(e)}")

    def load_from_disk(self):
        """Load queue state from JSON file"""
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                items_restored = 0

                # Handle new JSON format with metadata
                if isinstance(data, dict) and "queued" in data:
                    # Load queued items
                    for item_data in data.get("queued", []):
                        # Convert list back to tuple (video task format)
                        restored_item = tuple(item_data) if isinstance(item_data, list) else item_data
                        self.queue.put(restored_item)
                        items_restored += 1

                    # Load in-progress items back to queue (they weren't completed)
                    for in_progress_item in data.get("in_progress", []):
                        item_data = in_progress_item.get("data", in_progress_item)
                        restored_item = tuple(item_data) if isinstance(item_data, list) else item_data
                        self.queue.put(restored_item)
                        items_restored += 1

                # Handle old format (direct list)
                elif isinstance(data, list):
                    for item_data in data:
                        restored_item = tuple(item_data) if isinstance(item_data, list) else item_data
                        self.queue.put(restored_item)
                        items_restored += 1

                if items_restored > 0:
                    log(f"🔄 JSON: Restored {items_restored} video tasks from backup")

        except Exception as e:
            log(f"❌ Error loading JSON queue from disk: {str(e)}")

    def get_backup_summary(self):
        """Get a summary of what's in the backup file (for manual inspection)"""
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, dict) and "metadata" in data:
                    return data["metadata"]
                else:
                    return {"total_items": len(data) if isinstance(data, list) else 0}
            else:
                return {"total_items": 0, "file_exists": False}
        except Exception as e:
            return {"error": str(e)}
