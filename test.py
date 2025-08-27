from App import run_video_short_creation_thread, clean_get_gpu_memory
import torch
import gc
from Global_state import set_current_audio_path
import os 

if __name__ == "__main__":
    clean_get_gpu_memory()
    set_current_audio_path(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\Discipline, Confidence & The Champion’s Mindset - Chris Bumstead (4K)\Discipline, Confidence & The Champion’s Mindset - Chris Bumstead (4K).wav")
    text="[327.15s - 333.81s] but better that's now the next minimum yeah and I've I've had conversation with many people about [333.81s - 339.07s] the possibility of not having that and just being able to enjoy and relieve the pressure on yourself"

    run_video_short_creation_thread(r"c:\Users\didri\Documents\Discipline, Confidence & The Champion’s Mindset - Chris Bumstead (4K).mp4",start_time=327.15, end_time=339.07,subtitle_text=text)
