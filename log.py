# logger.py
import logging

# 1) Your log file
log_file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\debug_performance\log.txt"
log_stage_file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\debug_performance\log_stage.txt"

# # 2) Enumerate every entry-point or helper you’ll call with log_step()
# FLOW_STEPS = [
#     "transcribe_single_video",
#     "Transcript_Reasoning_AGENT",
#     "verify_saved_text_agent",
#     "gpu_worker",
#     "video_creation_worker",
#     "run_video_short_creation_thread",
#     "parse_multiline_block",
#     "SaveMotivationalText",
#     "create_motivationalshort",
#     "Delete_rejected_line",
#     "mix_audio",
#     "verify_start_time_end_time_text",
#     "get_current_textfile",
#     "clear_queue",
#     "change_saturation",
#     "enhance_detail_and_sharpness",
#     "sharpen_frame_naturally",
#     "group_subtitle_words_in_pairs",
#     "create_subtitles_from_pairs",
#     "detect_and_crop_frames_batch",
#     "create_short_video",
# ]

# # 3) Configure logging once
# LOG_FORMAT = (
#     "[%(asctime)s][%(filename)s:%(lineno)d - %(funcName)s()] "
#     "Step-%(step)d: %(message)s"
# )
# logging.basicConfig(
#     filename=log_file_path,
#     level=logging.DEBUG,
#     format=LOG_FORMAT,
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logger = logging.getLogger(__name__)


# def log_step(step_name: str, msg: str, level: int = logging.INFO):
#     """
#     Logs msg under the numbered step matching step_name (or 0 if unknown),
#     plus file/line/function context.
#     """
#     try:
#         step_num = FLOW_STEPS.index(step_name) + 1
#     except ValueError:
#         step_num = 0
#     logger.log(level, msg, extra={"step": step_num})


import datetime
import threading
def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thread_name = threading.current_thread().name
    log_message = f"[{timestamp}][{thread_name}] {msg}"
    print(log_message)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_message + "\n\n")

def log_Stage(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thread_name = threading.current_thread().name
    stage_log_message = f"[{timestamp}][{thread_name}] {msg}"
    print(stage_log_message)
    with open(log_stage_file_path, "a", encoding="utf-8") as f:
        f.write(stage_log_message + "\n\n")