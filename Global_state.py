
#-----------------------------------#
#Get/set Functions
#-----------------------------------#
_current_audio_url: str = None
_current_video_url: str = None
_current_youtube_channel: str = None
_current_agent_saving_file: str = None
_current_global_model: str = None
count: int = 0

def get_current_count() -> int:
    return count

def get_current_audio_path() -> str:
     return _current_audio_url

def get_current_videourl() -> str:
    return _current_video_url

def get_current_yt_channel()-> str:
     return _current_youtube_channel

def get_current_textfile() -> str:
    return _current_agent_saving_file

def get_current_global_model() -> str:
    return _current_global_model

def set_current_yt_channel(youtube_channel: str):
    global _current_youtube_channel 
    _current_youtube_channel = youtube_channel

def set_current_videourl(url: str):
    global _current_video_url
    _current_video_url = url

def set_current_textfile(url: str):
    global _current_agent_saving_file
    _current_agent_saving_file = url

def set_current_audio_path(url: str):
     global _current_audio_url
     _current_audio_url = url

def set_current_count(current_count: int):
    global count
    count = current_count 

def set_current_global_model(global_model):
    global _current_global_model
    _current_global_model = global_model


#---------------------------------------#
# Queue / Threads / Functions/ Variables
#---------------------------------------#
import threading
import queue
video_task_que = queue.Queue()
gpu_lock = threading.Lock()
transcript_queue = queue.Queue()
count_lock = threading.Lock()
chunk_proccesed_event = threading.Event()
