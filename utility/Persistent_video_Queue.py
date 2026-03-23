import json
import os
import threading
import random
from queue import Queue
from utility.log import log

class PersistentVideoQueue:
    def __init__(self, filepath):
        self.filepath = filepath
        self.queue = Queue()
        self.in_progress = {}  # Track items being processed
        self.lock = threading.Lock()
        self.load_from_disk()

    # --------------------
    # Validation utilities
    # --------------------
    def _is_valid_task(self, item) -> bool:
        """Validate a video task tuple/list structure.
        Expected format: (video_path, audio_path, start, end, subtitle_text)
        - tuple/list of length 5
        - start/end are numbers (int/float)
        - video_path and audio_path are non-empty strings
        """
        try:
            if not isinstance(item, (list, tuple)):
                return False
            if len(item) != 5:
                return False
            video_path, audio_path, start, end, _subtitle = item
            if not isinstance(video_path, str) or not video_path:
                return False
            if not isinstance(audio_path, str) or not audio_path:
                return False
            if not isinstance(start, (int, float)):
                return False
            if not isinstance(end, (int, float)):
                return False
            return True
        except Exception:
            return False

    def put(self, item):
        with self.lock:
            self.queue.put(item)
            self._save_to_disk()
            log(f"➕ Added new task to queue. Total: {self.qsize()} items")

    def get(self, timeout=None):
        item = self.queue.get(timeout=timeout)
        # Do not track sentinel or blatantly invalid items as in-progress
        if item is None or not self._is_valid_task(item):
            with self.lock:
                self._save_to_disk()
            return item, None

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
                        # Convert tuple to list for JSON serialization, but only if valid
                        if self._is_valid_task(item):
                            json_item = list(item) if isinstance(item, tuple) else item
                            queued_items.append(json_item)
                        else:
                            log(f"⚠️ Skipping invalid queued item during save: {item}")
                        # Always re-queue the original item to preserve runtime semantics
                        temp_items.append(item)
                except:
                    break

            # Put original items back in queue
            for item in temp_items:
                self.queue.put(item)

            # Convert in_progress items to JSON format (filter invalid/sentinel)
            in_progress_items = []
            for task_id, item in list(self.in_progress.items()):
                if item is None or not self._is_valid_task(item):
                    # Do not persist invalid entries in file; keep runtime map as-is
                    log(f"⚠️ Skipping invalid in-progress item during save: task={task_id}")
                    continue
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

            # Always write file, even when empty, to avoid stale reloads later
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(all_items, f, indent=2, ensure_ascii=False)
            if total_items > 0:
                log(f"💾 JSON backup updated: {len(queued_items)} queued + {len(in_progress_items)} in-progress")
            else:
                log("🧹 JSON backup cleared (no queued or in-progress items)")


        except Exception as e:
            log(f"❌ Error saving JSON queue to disk: {str(e)}")

    def load_from_disk(self):
        """Load queue state from JSON file"""
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Collect items to restore
                queued_to_restore = []
                in_progress_to_restore = []
                seen_keys = set()  # Deduplicate across queued and in_progress

                # Handle new JSON format with metadata
                if isinstance(data, dict) and "queued" in data:
                    # Gather queued items
                    for item_data in data.get("queued", []):
                        # Skip empty lists or malformed entries
                        if isinstance(item_data, list) and len(item_data) == 0:
                            log("⚠️ Ignoring empty [] task from backup (queued)")
                            continue
                        restored_item = tuple(item_data) if isinstance(item_data, list) else item_data
                        if self._is_valid_task(restored_item):
                            key = tuple(restored_item) if isinstance(restored_item, (list, tuple)) else restored_item
                            if key not in seen_keys:
                                seen_keys.add(key)
                                queued_to_restore.append(restored_item)
                            else:
                                log(f"♻️ Skipping duplicate task from backup (queued): {item_data}")
                        else:
                            log(f"⚠️ Ignoring invalid task from backup (queued): {item_data}")

                    # Gather in-progress items (treat as queued since they weren't completed)
                    for in_progress_item in data.get("in_progress", []):
                        item_data = in_progress_item.get("data", in_progress_item)
                        if isinstance(item_data, list) and len(item_data) == 0:
                            log("⚠️ Ignoring empty [] task from backup (in_progress)")
                            continue
                        restored_item = tuple(item_data) if isinstance(item_data, list) else item_data
                        if self._is_valid_task(restored_item):
                            key = tuple(restored_item) if isinstance(restored_item, (list, tuple)) else restored_item
                            if key not in seen_keys:
                                seen_keys.add(key)
                                in_progress_to_restore.append(restored_item)
                            else:
                                log(f"♻️ Skipping duplicate task from backup (in_progress): {item_data}")
                        else:
                            log(f"⚠️ Ignoring invalid task from backup (in_progress): {item_data}")

                # Handle old format (direct list)
                elif isinstance(data, list):
                    for item_data in data:
                        if isinstance(item_data, list) and len(item_data) == 0:
                            log("⚠️ Ignoring empty [] task from legacy backup list")
                            continue
                        restored_item = tuple(item_data) if isinstance(item_data, list) else item_data
                        if self._is_valid_task(restored_item):
                            key = tuple(restored_item) if isinstance(restored_item, (list, tuple)) else restored_item
                            if key not in seen_keys:
                                seen_keys.add(key)
                                queued_to_restore.append(restored_item)
                            else:
                                log(f"♻️ Skipping duplicate task from legacy backup: {item_data}")
                        else:
                            log(f"⚠️ Ignoring invalid task from legacy backup: {item_data}")

                # Shuffle only queued items and enqueue; preserve handling for in-progress
                restored_count = 0
                if queued_to_restore:
                    random.shuffle(queued_to_restore)
                    for item in queued_to_restore:
                        self.queue.put(item)
                        restored_count += 1

                # Enqueue in-progress items in original order (no shuffle)
                if in_progress_to_restore:
                    for item in in_progress_to_restore:
                        self.queue.put(item)
                        restored_count += 1

                if restored_count:
                    # Note: Only queued items are shuffled to maintain existing logic semantics
                    log(f"🔄 JSON: Restored {restored_count} valid video tasks from backup (queued shuffled)")

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
