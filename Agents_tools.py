from smolagents import tool,Tool,SpeechToTextTool
from tkinter.scrolledtext import ScrolledText
import os
from googleapiclient.discovery import build
import subprocess
import datetime
import tkinter as tk
from dotenv import load_dotenv
import wave
import contextlib
####FETCH MORE DETAILS TOO PROVIDE AGENT WITH MORE INFORMATION####

@tool
def Fetch_top_trending_youtube_videos(Search_Query: str) -> dict:
    """
        A tool for  Fetch enriched metadata + stats for the top trending YouTube videos for a query, including category names, tags, duration, views, likes, comments, and channel stats.
        Args:
        Search_Query (str): Topic or keywords to search (e.g. “Motivational”, “Tech Reviews”).

        Returns:
        dict: A YouTube API response containing for each video:
        - snippet: title, description, channelTitle, publishTime, thumbnails
        - statistics: viewCount, likeCount, commentCount
    """

    load_dotenv()       
    Api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build("youtube", "v3", developerKey=Api_key)

    #Searches for videos related too the (search query) retrieves basic info of each video. (20 results)
    search_resp = youtube.search().list(
                    part="snippet",
                    q=Search_Query,
                    type="video",
                    maxResults=20
                ).execute()

    #Extracts the videoId of each video in items
    video_ids = [item["id"]["videoId"] for item in  search_resp.get("items",[])]

    #Early exit if no videos is found!
    if not video_ids:
        return {"items": []}
    
    
    #Fetches snippet + statistics + contentdetails --> fetches mote stats and details --> (title, stats,duration) using the video id's 
    stats_resp = youtube.videos().list(
        part="snippet,statistics,contentdetails",
        id=",".join(video_ids)
    ).execute()


    # Extracts all unique categoryID from videos
    category_ids = list({item["snippet"]["categoryId"] for item in stats_resp.get("items", [])})
    fetch_category_names = youtube.videoCategories().list(
        part="snippet",
        id=",".join(category_ids),
        regionCode="US"
    ).execute()


    #Looksup human redable category names (music, motivation, education) for each categoryId
    category_map = {
        item["id"]: item["snippet"]["title"]
        for item in fetch_category_names.get("items",[])
        }
        #Results --> {    "1": "Film & Animation" ..}



    #Retrieve statistics for each youtube channel too the video vi have found.
    channel_ids = list({item["snippet"]["channelId"] for item in stats_resp.get("items",[])})
    
    #We use the list of channel-ids to retrieve channel statistics like (subscriber count)
    channel_response = youtube.channels().list(
        part="statistics",
        id=",".join(channel_ids)
    ).execute()


    #we map the channel_ids  -  amount of subscribers. 
    channel_map = {
        item["id"]: item["statistics"]["subscriberCount"] 
        for item in channel_response.get("items",[])
        }
        #Results --> {"UCabc123": "50000" ...}



    #Summarize and attach channel stats
    #loops through all videos in stats_resp pulls out each videos' channelId
    channel_ids = list({item["snippet"]["channelId"] for item in stats_resp.get("items",[])})

    #call too get stats like(subscriber count) for each channel
    channel_response = youtube.channels().list(
        part="statistics",
        id=",".join(channel_ids)
    ).execute()

    #dict that maps each channels' id to its subscriber count.
    channel_map = {item["id"]: item["statistics"]["subscriberCount"] for item in channel_response.get("items",[])}

    enriched = []
    for vid in stats_resp.get("items",[]):
        snippet = vid["snippet"]
        statistics = vid.get("statistics", {})
        content = vid.get("contentDetails", {})

        enriched.append({
            "videoId": vid["id"],
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "tags": snippet.get("tags", []),
            "channelTitle": snippet.get("channelTitle"),
            "subscriberCount": channel_map.get(snippet.get("channelId")),
            "category": category_map.get(snippet.get("categoryId")),
            "publishedAt": snippet.get("publishedAt"),
            "duration": content.get("duration"),
            "viewCount": statistics.get("viewCount"),
            "likeCount": statistics.get("likeCount"),
            "commentCount": statistics.get("commentCount"),
           })
        return {"items": enriched}




class ChunkLimiterTool(Tool):
    name = "chunk_limiter"
    description = (
        "Call this tool as a function using: chunk = chunk_limiter(file_path=..., max_chars=...) "
        "It returns one chunk of transcript text per call. You must only call it once per reasoning step, if you have run this function before, you must call `chunk_limiter.reset()` in the code block."
        "This tool keeps track of remaining transcript content internally, and will return the next chunk each time it's called. "
        "When it returns an empty string, the full transcript has been processed. "
        "If 'file_path' is omitted in future calls, it will reuse the last known value automatically."
    )

    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the transcript.txt file",
        },
        "max_chars": {
            "type": "integer",
            "description": "Maximum number of characters per chunk (suggested 1000)",
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.called = False
        self.saved_file_path = None

    def reset(self):
        self.called = False

    def forward(self, file_path: str, max_chars: int) -> str:
        if self.called:
            raise Exception(
                "ChunkLimiterTool was already called in this reasoning step. "
                "You must wait for the ReasoningAgent to finish processing before calling this tool again."
            )
        self.called = True

        if file_path:
            self.saved_file_path = file_path
        elif not self.saved_file_path:
            raise ValueError("file_path must be provided the first time ChunkLimiterTool is used.")
        
        with open(self.saved_file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            return ""
        
        split_idx = text.rfind("\n", 0, max_chars)
        if split_idx == -1: 
            split_idx = min(len(text),max_chars)

        chunk = text[:split_idx].strip()
        remaing = text[split_idx:].lstrip()

        with open(self.saved_file_path, "w", encoding="utf-8") as f:
                f.write(remaing)
            
        return f"{chunk}"



class Chunk_line_LimiterTool(Tool):
    name = "chunk_limiter_by_line"
    description = (
        "Call this tool as a function using: chunk = chunk_limiter(file_path=..., max_chars=...) "
        "It returns one chunk of transcript text per call. You must only call it once per reasoning step, if you have run this function before, you must call `chunk_limiter.reset()` in the code block."
        "This tool keeps track of remaining transcript content internally, and will return the next chunk each time it's called. "
        "When it returns an empty string, the full transcript has been processed. "
        "If 'file_path' is omitted in future calls, it will reuse the last known value automatically."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the transcript.txt file",
           "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.called = False
        self.saved_file_path = None
        self.current_line = 0

    def reset(self):
        self.called = False
        self.current_line = 0

    def forward(self, file_path: str = None):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        block = []
        in_block = False
        for line in lines[self.current_line:]:
            self.current_line += 1
            if "===START_QUOTE===" in line:
                in_block = True
                block = []
            elif "===END_QUOTE===" in line and in_block:
                in_block = False
                return ''.join(block).strip()
            elif in_block:
                block.append(line)
        return ""





@tool
def ExtractAudioFromVideo(video_path: str) -> str:
    """Extracts  mono 16kHz WAV audio from a video using ffmpeg.
        Args:
            video_path (str): The full path to the video file.

        Returns:
            str:  the path to the extracted audio file.
    """
    audio_path = "temp_audio.wav"
    command = [
        "ffmpeg",
        "-y",  #
        "-i", video_path,
        "-ac", "1", 
        "-ar", "16000",  
        "-vn", 
        "-f", "wav",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    print(f"[LOG] Extracted audio duration: {duration:.2f} seconds (~{duration/60:.2f} minutes)")

    return audio_path



@tool
def Log_Agent_Progress(chat_display: ScrolledText, stage: str, message: str) -> str:
    """A tool for logging agent thoughts, actions, and reflections during task execution.
        Args:
            chat_display(str):  you have access too the chat_display variable just do chat_display=chat_display when you call this function.
            stage (str): One of "info", "action", "reflection".
            message (str): A descriptive message of what the agent is doing or thinking.

        Returns:
            str: Confirmation that the message has been logged.
    """
    valid_stages = {"info", "action", "reflection"}
    if stage not in valid_stages:
        stage = "info"  

    log_entry = {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "stage": stage,
                "message": message
            }

    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, f"{log_entry['timestamp']} [{stage.upper()}] {message}\n")
    chat_display.config(state=tk.DISABLED)
    chat_display.see(tk.END)

    print(f"[{stage.upper()}] {log_entry['timestamp']}: {message}")
    return "Log recorded successfully."





# @tool
# def Upload_video_to_socialMedia_platform(title: str, description: str, time: int):
#                 """
                
#                 """
#                 return   
            



# @tool 
# def add_text_to_video():
#                 """
                
#                 """
#                 return





# @tool
# def add_audio_to_video():
#                 """
                
#                 """
#                 return





# @tool
# def add_filter_to_video():
#                 """
                
#                 """
#                 return 


def transcribe_audio_to_txt(video_paths):
    tool = SpeechToTextTool()
    tool.setup()

    for video_path in video_paths:
        if not os.path.isfile(video_path):
            print(f"File not found: {video_path}")
            continue

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        folder = os.path.dirname(video_path)
        audio_path = os.path.join(folder, f"{base_name}.wav")
        txt_output_path = os.path.join(folder, f"{base_name}.txt")

        # Extract audio using ffmpeg
        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output if exists
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # WAV format
                audio_path
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Extracted audio to: {audio_path}")
        except subprocess.CalledProcessError:
            print(f"Failed to extract audio from {video_path}")
            continue

        # Transcribe the audio
        try:
            result_txt_path = tool.forward({"audio": audio_path})
            # Optionally rename the transcript to desired name
            if result_txt_path != txt_output_path:
                os.rename(result_txt_path, txt_output_path)
            print(f"Transcript saved to: {txt_output_path}")
        except Exception as e:
            print(f"Transcription failed for {audio_path}: {e}")
