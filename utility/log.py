
log_file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\logs\log.txt"
import datetime
import threading
def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thread_name = threading.current_thread().name
    log_message = f"[{timestamp}][{thread_name}] {msg}"
    print(log_message)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

