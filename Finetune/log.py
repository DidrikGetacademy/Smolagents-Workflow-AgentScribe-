
from datetime import datetime
log_path_finetune = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\finetuning_general_log.txt"
log_messages = set()
def log(msg):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_messages.add(msg)
    print(msg)
    log_message = f"[{timestamp}] {msg}"
    with open(log_path_finetune, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")


log_path_merge = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\merge_and_unload_log.txt"
log_messages = set()
def merge_logger(msg):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_messages.add(msg)
    print(msg)
    log_message = f"[{timestamp}] {msg}"
    with open(log_path_merge, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

log_path_val = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\Manual_test_output.txt"
log_messages = set()
def validation_logger(msg):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_messages.add(msg)
    print(msg)
    log_message = f"[{timestamp}] {msg}"
    with open(log_path_val, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")