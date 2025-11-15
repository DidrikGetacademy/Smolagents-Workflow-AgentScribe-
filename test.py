from App import run_video_short_creation_thread, clean_get_gpu_memory
import torch
import gc
import utility.Global_state as Global_state
import os
from App import create_short_video
#from Custom_Agent_Tools import  run_multi_Vision_model_conclusion
from Agent_AutoUpload.upload_Socialmedia import Get_latestUpload_date

# def open_work_file(work_queue_folder: str) -> str:
#      """A tool that returns motivational text  for each of the work folders saved by another agent.
#         work_queue_folder: (str): path to the work folder
#         Returns: (str) text from each folder
#      """
#      text_list = []
#      count = 1
#      import re
#      for subdir in os.listdir(work_queue_folder):
#           subdir_path = os.path.join(work_queue_folder,subdir)
#           if os.path.isdir(subdir_path):
#             copy_path = os.path.join(subdir_path, "agent_saving_path_copy.txt")
#             audio_path = os.path.join(subdir_path, f"{subdir}.wav")
#             video_path = os.path.join(subdir, f"c:/Users/didri/Documents/{subdir}.mp4")
#             if os.path.exists(copy_path):
#                  with open(copy_path, "r", encoding="utf-8") as r:
#                       content = r.read().strip()
#                       if content:
#                            blocks_pattern = r'===START_TEXT===(.*?)===END_TEXT==='
#                            blocks = re.findall(blocks_pattern, content, re.DOTALL)
#                            if blocks:
#                                cleaned_blocks = [block.strip() for block in blocks]
#                                numbered_block = []

#                                for block in cleaned_blocks:
#                                    numbered_block.append(f"Motivational snippet[{count}]:\n{block}")
#                                    count += 1

#                                cleaned_content = '\n\n'.join(numbered_block)

#                            else:
#                                cleaned_content = content

#                            title = subdir
#                            text_list.append(f"Video title: {title}\nvideo_path:{video_path}\nVideo Audio:{audio_path}\n--------------\n{cleaned_content}")
#             count = 1
#      return "\n\n".join(text_list) if text_list else "No files found."




if __name__ == "__main__":
    clean_get_gpu_memory()
    Global_state.set_current_yt_channel("LR_Youtube")
    audio_path = Global_state.set_current_audio_path(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p]\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p].wav")
    text="[2000.69s - 2012.88s] somebody who's got a lot of"


    run_video_short_creation_thread(r"c:\Users\didri\Documents\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p].mp4",audio_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p]\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p].wav", start_time=2000.69, end_time=2027.52,subtitle_text=text)
    # video_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Original.mp4"
    # audio_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\temp_audio.wav"
    # content = {
    #     "Transcript": "If you are happy and you",
    #     "emotion": "sad",
    #     "videolist": "jingle bell"
    # }
    # run_multi_Vision_model_conclusion(video_path=video_path, audio_path=audio_path, Additional_content=content)


    # content = open_work_file(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder")
    # print(content)
    # create_short_video(video_path=r"c:\Users\didri\Documents\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p].mp4", audio_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p]\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p].wav", start_time=7674.53, end_time=7676.00, video_name="hey", subtitle_text=[{'word': 'my', 'start': 0.00, 'end': 3.02}, {'word': 'strategy', 'start': 1.0, 'end': 1.50}],Video_output_path=None, apply_lut=False)
