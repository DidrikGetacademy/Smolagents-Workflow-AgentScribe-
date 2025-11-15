import os
import sys
REPO_ROOT = os.path.abspath('.')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
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
from smolagents.tools import PipelineTool
from faster_whisper import WhisperModel
import torch
import time
from neon.log import log
import utility.Global_state as Global_state
import gc
import tempfile
import yt_dlp
import requests
import json
from typing import List ,Dict
cookie_file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Secrets\youtube.com_cookies.txt"



####FETCH MORE DETAILS TOO PROVIDE AGENT WITH MORE INFORMATION####
@tool
def Fetch_top_trending_youtube_videos(Search_Query: str) -> dict:
    """
        A tool for Fetching enriched metadata + stats for the top trending YouTube videos for a query, including category names, tags, duration, views, likes, comments, and channel stats.
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
    if not Api_key:
        raise ValueError(f"error api key is not in enviorment variables")

    #Searches for videos related too the (search query) retrieves basic info of each video. (20 results)
    search_resp = youtube.search().list(
            part="snippet",
            q=Search_Query,
            type="video",
            regionCode="US",
            order="viewCount",
            maxResults=5
        ).execute()


    #Extracts the videoId of each video in items
    video_ids = [item["id"]["videoId"] for item in  search_resp.get("items",[])]

    #Early exit if no videos is found!
    if not video_ids:
        return {"items": []}


    #Fetches snippet + statistics + contentdetails --> fetches mote stats and details --> (title, stats,duration) using the video id's
    stats_resp = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=",".join(video_ids)
    ).execute()

    # Extracts all unique categoryID from videos
    category_ids = list({item["snippet"]["categoryId"] for item in stats_resp.get("items", [])})
    fetch_category_names = youtube.videoCategories().list(
        part="snippet",
        id=",".join(category_ids),
    ).execute()


    #Looksup human redable category names (music, motivation, education) for each categoryId
    category_map = {
        item["id"]: item["snippet"]["title"]
        for item in fetch_category_names.get("items",[])
        }




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



    enriched = []
    count = 0
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
            raise Exception("ChunkLimiterTool was already called in this reasoning step.")
        self.called = True

        if file_path:
            self.saved_file_path = file_path
        elif not self.saved_file_path:
            raise ValueError("file_path must be provided the first time ChunkLimiterTool is used.")

        with open(self.saved_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return ""

        chunk_lines = []
        total_len = 0
        i = 0

        while i < len(lines):
            line = lines[i]
            line_len = len(line)


            if total_len + line_len > max_chars and chunk_lines:
                break

            chunk_lines.append(line)
            total_len += line_len
            i += 1

        chunk = "".join(chunk_lines)

        remainder = "".join(lines[i:])
        with open(self.saved_file_path, "w", encoding="utf-8") as f:
            f.write(remainder)

        return chunk




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
            str: the path to the extracted audio file.
    """
    audio_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-f", "wav",
        audio_path
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not created: {audio_path}")

        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        log(f"[LOG] Extracted audio duration: {duration:.2f} seconds (~{duration/60:.2f} minutes)")

    except Exception as e:
        print(f"Error during audio extraction: {e}")
        raise

    return audio_path




def transcribe_audio_to_txt(video_paths):
    tool = SpeechToTextTool()
    tool.setup()

    for video_path in video_paths:
        if not os.path.isfile(video_path):
            log(f"File not found: {video_path}")
            continue

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        folder = os.path.dirname(video_path)
        audio_path = os.path.join(folder, f"{base_name}.wav")
        txt_output_path = os.path.join(folder, f"{base_name}.txt")


        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                audio_path
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"Extracted audio to: {audio_path}")
        except subprocess.CalledProcessError:
            log(f"Failed to extract audio from {video_path}")
            continue


        try:
            result_txt_path = tool.forward({"audio": audio_path})

            if result_txt_path != txt_output_path:
                os.rename(result_txt_path, txt_output_path)
            log(f"Transcript saved to: {txt_output_path}")
        except Exception as e:
            log(f"Transcription failed for {audio_path}: {e}")




@tool
def Read_transcript(transcript_path: str, start_count: int = 0) -> str:
    """Reads up to 1000 characters from a transcript starting at a given position, note: If more content exists, a message is added to indicate that you must call the function again.
    Args:
        transcript_path (str): The path to the transcript file.
        start_count (int): The position in the file to start reading from.

    Returns:
        str: A chunk of the transcript (max 1000 chars) and a message if there's more content.
    """
    chunk_size = 1000
    with open(transcript_path , "r") as file:
        file.seek(start_count)
        content = file.read(chunk_size + 1)

        if len(content) > chunk_size:
            output = content[:chunk_size] + "\n\nThe transcript has more content please run the `Read_transcript` tool call again"
        else: output = content

        return output






class SpeechToTextTool(PipelineTool):
    default_checkpoint = r"c:\Users\didri\Desktop\LLM-models\Audio-Models\faster-whisper-large-v3-turbo-int8float16"
    description = "Fast tool that transcribes audio into text using faster-whisper. It returns the path to the transcript file"
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, a URL, or a tensor.",
        },
        "text_path": {
            "type": "string",
             "description": "The path to save the transcript to.",
        },
        "video_path": {
            "type": "string",
            "description": "The path to the video to transcribe. only for info logging",
        }
    }
    output_type = "string"
    def setup(self):

        self.model = WhisperModel(
                model_size_or_path=self.default_checkpoint,
                device="cuda",
                compute_type="int8_float16"
                )
    def forward(self, inputs):
        audio_path = inputs["audio"]
        text_path = inputs["text_path"]
        video_path = inputs["video_path"]
        segments, info = self.model.transcribe(
            audio_path,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
        log(f"\n🔊 Using Whisper on device: {self.device}, \ntranscribing video: {video_path} \n   with inputs: {self.inputs}")
        log(f"[INFO] Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
        log(f"[INFO] Audio Duration: {info.duration:.2f} seconds")

        try:
            with open(text_path, "w", encoding="utf-8") as f:
                log(f"opening txt_path on: {text_path} device: {self.device}")
                start_write_time = time.time()
                for segment in segments:
                        f.write(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text.strip()}\n")
                end_write_time = time.time()
        finally:
            total_write_time = end_write_time - start_write_time
            num_segments = len(segments)
            avg_segment_write_time = total_write_time / num_segments if num_segments > 0 else 0
            log(f"Finished writing {num_segments} segments.")
            log(f"Total write time: {total_write_time:.4f} seconds")
            log(f"Average write time per segment: {avg_segment_write_time:.6f} seconds")
            log(f"transcription complete ! device  {self.device}")
            del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return text_path

    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs


def parse_multiline_block(block_text):#DIDRIK
            """
            1) Splitter blokken i linjer.
            2) Finner ALLE '[start - end] tekst' — uansett hvordan de står.
            3) Bygger NY tekst med én linje per subtitle.
            4) Returnerer start_time (første), end_time (siste) OG den nye teksten.
            """
            import re
            log(f"block_text: {block_text}")

            pattern = re.compile(r"\[(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\]\s*([^\[]+)")
            matches = pattern.findall(block_text)

            if not matches:
                return None, None, ""
            new_text = "\n".join(
                f"[{start}s - {end}s] {text.strip()}"
                for start, end, text in matches
            )
            Video_start_time = float(matches[0][0])
            Video_end_time = float(matches[-1][1])
            log(f"[parse_multiline_block] start_time: {Video_start_time}, [parse_multiline_block] end_time: {Video_end_time},[parse_multiline_block] new_text: {new_text}")

            return Video_start_time, Video_end_time, new_text




@tool
def create_motivationalshort(text: str) -> None:
        """
        Tool that creates a motivational shorts video.
        You must Explicitly call correct Python syntax for string arguments in tool calls here is an exsample:

        This is correct:
            create_motivationalshort(text="....")

        This is invalid:
            create_motivationalshort(text="..."])
        Args:
            text (str): The complete input text for the motivational short.
                The text must include the entire message — as one complete
                thought, quote, or motivational statement — along with
                timestamps for each line. The text must be enclosed within
                the markers (===START_TEXT===) and (===END_TEXT===) in the
                following format:
                    ===START_TEXT===
                    ...
                    [start_time - end_time] Line 1
                    [start_time - end_time] Line 2
                    ...
                    ===END_TEXT===

                All content between START_TEXT and END_TEXT will be treated
                as a single cohesive message and analyzed as one unit.

        """
        with open (r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p]\verify_agent_verified.txt", "a", encoding="utf-8") as f:
            f.write(f"{text}\n")
        try:
            start_time, end_time, new_text = parse_multiline_block(text)
        except Exception as e:
             log(f"Error during [parse_multiline_block]: {str(e)}")


        if start_time is None or end_time is None:
             raise ValueError(f"start_time or end_time is None, start_time: {start_time}, end_time: {end_time}")

        video_url = Global_state.get_current_videourl()
        audio_path = Global_state.get_current_audio_path()
        try:
            log(f"\n now Added work to QUEUE: \n video url: {video_url}\n, audio_path: {audio_path} \n start_time: {start_time}s \n end_time {end_time}s\nOriginal text: {text}")

            Global_state.video_task_que.put((video_url, audio_path, start_time, end_time, new_text))
            count = Global_state.get_current_count()
            count +=1
            log(f"Added VideoWork to videotask Queue:\n{video_url}\nAudio_path:{audio_path}\n{start_time}\n {end_time}\n {new_text}\n  Amount of added videowork to queue: {count}\n ")
            Delete_rejected_line(text)
            log(f"Current videos added to que for proccessing: {count}")
            Global_state.set_current_count(count)


        except Exception as e:
             log(f"Error addng to queue: {str(e)}")


@tool
def SaveMotivationalText(text: str, text_file: str) -> None:
    """A function/tool that saves qualifying motivational texts for a punchy 15-30 second shorts video.
    Args:
        text (str): The complete motivational text block to save.
            - The text must evoke emotional energy (e.g., hope, empathy, vulnerability) alongside inspiration, making it suitable for grabbing a listener's attention in a short video.
            - The text must be a self-contained, inspirational passage that encourages action, resilience, discipline, perseverance, or personal growth, suitable for a short video without relying on prior context or external references.
            - Every line must include the exact timestamp range from the original chunk (e.g., '[start1 - end1] Text1 [start2 - end2] Text2').
            - You must provide the text exactly as it appears in the chunk, preserving every word, space, and punctuation, with no rephrasing, omissions, or alterations, except that leading conjunctions or transitional words (e.g., 'And,' 'So,' 'But') may be removed if doing so makes the text fully self-contained without changing the core motivational message.
            - If the saved text begins or ends in the middle of a line, the timestamp from that line must still be included.
            - The number of lines is not fixed; include all lines (with timestamps) that the text spans, whether it is one line or many.
            - Wrap the entire block in triple quotes if it contains commas, quotes, or line breaks to ensure proper handling in Python.
        text_file (str): The path to the text file where the motivational text will be saved.
    """

    with open(text_file, "a", encoding="utf-8") as f:
                f.write("===START_TEXT===")
                f.write(text.strip())
                f.write("===END_TEXT===\n")
                log(f"text: {text}")






@tool
def Delete_rejected_line(text: str) -> None:
        """  Deletes a block from the text file that contains the given inner text,
             by removing the whole block: ===START_TEXT=== text... ===END_TEXT===
        Args:
            text: The line to delete (i.e., considered rejected/not valid) Format:
              ===START_TEXT===
              [start_time - end_time] actual text here... [start_time - end_time] ....
              ===END_TEXT===
        """
        import re
        log(f"\n[Delete_rejected_line] text into func: {text}")
        text_file = Global_state.get_current_textfile()
        log(f"[Delete_rejected_line] path to textfile: {text_file}")

        with open(text_file, 'r', encoding="utf-8") as f:
                    content = f.read()

        escaped_text = re.escape(text.strip())

        pattern = rf'===START_TEXT\s*.*?{escaped_text}.*?\s*===END_TEXT===\s*'

        new_content, num_subs = re.subn(pattern, '', content, flags=re.DOTALL)

        if num_subs == 0:
            log("[Delete_rejected_line] No matching block found — nothing deleted.")
            return

        with open(text_file, 'w', encoding="utf-8") as f:
            f.write(new_content)

        with open (r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p]\verify_agent_declined.txt", "a", encoding="utf-8") as f:
            f.write(f"{text}\n")
        log(f"[Delete_rejected_line] Deleted {num_subs} block(s) containing the text.")




@tool
def open_work_file(work_queue_folder: str) -> str:
     """A tool that returns motivational text  for each of the work folders saved by another agent.
    Args:
        work_queue_folder: The path or name of the folder containing queued work.
     """
     text_list = []
     count = 1
     import re
     for subdir in os.listdir(work_queue_folder):
          subdir_path = os.path.join(work_queue_folder,subdir)
          if os.path.isdir(subdir_path):
            copy_path = os.path.join(subdir_path, "agent_saving_path_copy.txt")
            audio_path = os.path.join(subdir_path, f"{subdir}.wav")
            video_path = os.path.join(subdir, f"c:/Users/didri/Documents/{subdir}.mp4")
            if os.path.exists(copy_path):
                 with open(copy_path, "r", encoding="utf-8") as r:
                      content = r.read().strip()
                      if content:
                           blocks_pattern = r'===START_TEXT===(.*?)===END_TEXT==='
                           blocks = re.findall(blocks_pattern, content, re.DOTALL)
                           if blocks:
                               cleaned_blocks = [block.strip() for block in blocks]
                               numbered_block = []

                               for block in cleaned_blocks:
                                   numbered_block.append(f"Motivational snippet[{count}]:\n{block}")
                                   count += 1

                               cleaned_content = '\n\n'.join(numbered_block)

                           else:
                               cleaned_content = content

                           title = subdir
                           text_list.append(f"Video title: {title}\n--------------\nvideo_path:{video_path}\nVideo Audio:{audio_path}\n{cleaned_content}")
            count = 1
     return "\n\n".join(text_list) if text_list else "No files found."









### later i need to add possibility of more clips in middle if video_paths are > 3.
@tool
def montage_short_creation_tool(montage_list: List[Dict[str,str]]) -> str:
        """A tool that creates a montage-style short video by combining words, sentences, or complete snippets from multiple input sources.
           The tool must intelligently select the content from each video title snippet—either a few words, a full sentence, or the entire snippet—so that, when ordered in start–middle–ending sequence, it forms a smooth, coherent, and complete motivational message suitable for a short motivational video.
           Duration constraints:
            - Per-clip: each selected segment from its video_url must be between 3 and 7 seconds long (hard limit; do not exceed 7s per clip).
            - Combined: aim for a total runtime between 15 and 30 seconds; never exceed 30 seconds.
                Prefer concise word-level or single-sentence excerpts to stay within limits.
                Prefer words over a full snippet unless a full snippet is exceptionally valuable.
          Args:
            montage_list (List[Dict[str, str]]): A list of dictionaries, where each dictionary
                represents a video segment with the following required keys:

                - "video_url" (str): Path to the source video file.
                - "audio_path" (str): Path to the audio file associated with the clip.
                - "order" (str): Position of the clip in the montage sequence ("start", "middle", "ending").
                - "text" (str): Annotated text with embedded timestamps indicating the portion of the clip to extract.
                - "ID" (float): A unique identifier.
                    - The letter N represents the montage sequence number (1 = first montage, 2 = second montage, 3 = third montage, etc.).
                    - IDs per `run_montage_short_creation` tool call must be N.1, N.2, N.3, where:
                        • N.1 = start
                        • N.2 = middle
                        • N.3 = ending
                    -Never emit .4 or any number beyond 3.
                    -IDs must follow the N.1, N.2, N.3 format per call.
                - reason (str): A concise explanation of why the chosen text (words, a full sentence, or the entire snippet) was selected, focusing on its narrative coherence, emotional impact, motivational relevance, or contribution to the montage's flow.
                - YT_channel (str): The YouTube channel choosen for uploading the montage short. MR_Youtube, LRS_Youtube, LM_Youtube, or  LR_Youtube

         Example correct output:
            run_montage_short_creation([{"audio_path": r"...","order": "start","video_url": r"...","text": "...","ID": 1.1},{"audio_path": r"...","order": "middle","video_url": r"...","text": "...","ID": 1.2},{"audio_path": r"...","order": "ending","video_url": r"...","text": "...","ID": 1.3}])
            run_montage_short_creation([{"audio_path": r"...","order": "start","video_url": r"...","text": "...","ID": 2.1},{"audio_path": r"...","order": "middle","video_url": r"...","text": "...","ID": 2.2},{"audio_path": r"...","order": "ending","video_url": r"...","text": "...","ID": 2.3}])
            run_montage_short_creation([{"audio_path": r"...","order": "start","video_url": r"...","text": "...","ID": 3.1},{"audio_path": r"...","order": "middle","video_url": r"...","text": "...","ID": 3.2},{"audio_path": r"...","order": "ending","video_url": r"...","text": "...","ID": 3.3}])
        """
        items_queued = 0
        for item in montage_list:
             video_path = item["video_url"]
             audio = item["audio_path"]
             order = item["order"]
             text = item["text"]
             Montage_ID = item["ID"]
             YT_channel = item["YT_channel"]
             log(f"\n[montage_short_creation_tool] Item in montage_list:\n video_path: {video_path}, \n audio: {audio}, \n order: {order}, \n text: {text}, \n Montage_ID: {Montage_ID}, \n YT_channel: {YT_channel}")

             start_time, end_time, new_text = parse_multiline_block(text)
             if order not in ("start", "middle", "ending"):
                        raise ValueError(f"order must be one of ['start','middle','ending'], got: {order}")

             if start_time is None or end_time is None:
                raise ValueError(f"start_time or end_time is None, start_time: {start_time}, end_time: {end_time}")


             try:
                Global_state.Montage_clip_task_Que.put((
                    video_path,
                    audio,
                    start_time,
                    end_time,
                    new_text,
                    order,
                    Montage_ID,
                    YT_channel
                ))
                items_queued += 1
             except Exception as e:
                log(f"Error adding to queue.")

             log(f"Added to Montage_clip_task_Que: video_path: {video_path}, audio: {audio}, start_time: {start_time}, end_time: {end_time}, order: {order}, Montage_ID: {Montage_ID}")
        log(f"items queued: {items_queued}")











class SpeechToTextTool_viral_agent(PipelineTool):
    default_checkpoint = r"c:\Users\didri\Desktop\LLM-models\Audio-Models\faster-whisper-large-int8-ct2"
    description = "Fast tool that transcribes audio into text using faster-whisper. It returns transcript text of the audio"
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, a URL, or a tensor.",
        },

    }
    output_type = "string"
    def setup(self):
        self.model = WhisperModel(
                model_size_or_path=self.default_checkpoint,
                device="cpu",
                compute_type="int8",

                    )

    def forward(self, inputs):
        audio_path_viral_agent = inputs["audio"]
        segments,_ = self.model.transcribe(
            audio_path_viral_agent,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )


        try:
            result = []
            for segment in segments:
                    result.append(f"{segment.text.strip()}\n")


        except Exception as e:
            log(f"error during transcribing: {str(e)}")
        finally:
            os.remove(audio_path_viral_agent)
            del self.model
            if self.device == "cpu":
                del self.device
                torch.cuda.empty_cache()
            else:
                import gc
                gc.collect()

        return " ".join(result)

    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs





class SpeechToTextToolCPU_Custom(PipelineTool):
    default_checkpoint = r"c:\Users\didri\Desktop\LLM-models\Audio-Models\faster-whisper-large-v3-turbo-int8float16"
    description = "Fast tool that transcribes audio into text using faster-whisper. It returns the path to the transcript file"
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, a URL, or a tensor.",
        },

    }
    output_type = "string"
    def setup(self):
        self.model = WhisperModel(
                model_size_or_path=self.default_checkpoint,
                device="auto",
                compute_type="int8_float16",

                    )



    def forward(self, inputs):
        audio_path = inputs["audio"]
        segments = self.model.transcribe(
            audio_path,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True
        )


        try:
            words = []
            for segment in segments:
                    for word in segment.words:
                        words.append({
                            "word": word.word.strip(),
                            "start": word.start,
                            "end": word.end
                        })
            return words
        except Exception as e:
            log(f"error during transcribing {str(e)}")
            return []



        finally:
            del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()


    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs









class SpeechToTextToolCPU(PipelineTool):
    default_checkpoint = r"c:\Users\didri\Desktop\LLM-models\Audio-Models\faster-whisper-large-int8-ct2"
    description = "Fast tool that transcribes audio into text using faster-whisper. It returns the path to the transcript file"
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, a URL, or a tensor.",
        },
        "text_path": {
            "type": "string",
             "description": "The path to save the transcript to.",
        },
        "video_path": {
            "type": "string",
            "description": "The path to the video to transcribe. only for info logging",
        }
    }
    output_type = "string"
    def setup(self):
        self.model = WhisperModel(
                model_size_or_path=self.default_checkpoint,
                device="cpu",
                compute_type="int8",
                cpu_threads=6
                    )

    def forward(self, inputs):
        audio_path = inputs["audio"]
        text_path = inputs["text_path"]
        video_path = inputs["video_path"]
        segments, info = self.model.transcribe(
            audio_path,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
        log(f"\n🔊 Using Whisper on device: {self.device}, \ntranscribing video: {video_path} \n   with inputs: {self.inputs}")
        log(f"[INFO] Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
        log(f"[INFO] Audio Duration: {info.duration:.2f} seconds")

        try:
            with open(text_path, "w", encoding="utf-8") as f:
                log(f"opening txt_path on: {text_path} device: {self.device}")
                start_write_time = time.time()
                for segment in segments:
                        f.write(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text.strip()}\n")
                end_write_time = time.time()
        finally:
            total_write_time = end_write_time - start_write_time
            num_segments = len(segments)
            avg_segment_write_time = total_write_time / num_segments if num_segments > 0 else 0
            log(f"Finished writing {num_segments} segments.")
            log(f"Total write time: {total_write_time:.4f} seconds")
            log(f"Average write time per segment: {avg_segment_write_time:.6f} seconds")
            log(f"transcription complete ! device  {self.device}")
            del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return text_path

    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs




class SpeechToText_short_creation_thread(PipelineTool):
    default_checkpoint = r"c:\Users\didri\Desktop\LLM-models\Audio-Models\faster-whisper-large-v3-turbo-int8float16"
    description = "Fast tool that transcribes audio into text using faster-whisper. It returns the matched words with timestamps."
    name = "transcriber"
    def __init__(self, device="cuda"):
        self.device = device
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, a URL, or a tensor.",
        },
        "subtitle_text": {
            "type": "string",
            "description": "The correct subtitle text to match to",
        },
        "padding": {
            "type": "number",
            "description": "padding"
        },
        "original_start_time":
        {
            "type": "number",
            "description": "Extended end time for video"
        },

        "original_end_time": {
            "type": "number",
            "description": "extended start time for video"
        }
    }
    output_type = "string"

    def setup(self):
        self.model = WhisperModel(
            model_size_or_path=self.default_checkpoint,
            device=self.device,
            compute_type="int8_float16"
        )

    @staticmethod
    def clean_word(word):
        import re
        return re.sub(r"[^\w']", "", word).lower()

    def forward(self, inputs):
        audio_path = inputs["audio"]
        subtitle_text = inputs["subtitle_text"]
        original_start_time = round(inputs["original_start_time"],2)

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language="en",
                word_timestamps=True,
                temperature=0.0,
                beam_size=10,
                patience=1.5,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.6,                 # more strict VAD
                    "min_silence_duration_ms": 400,   # require a real pause
                    "min_speech_duration_ms": 250,    # stabilize speech regions
                    "speech_pad_ms": 100,
                    },
                no_speech_threshold=0.6,

            )

            log(f"[INFO] Audio Duration: {info.duration:.2f} seconds Detected Language: {info.language} (confidence: {info.language_probability:.2f})")

            all_words = []
            for segment in segments:
                all_words.extend(segment.words)


            subtitle_tokens = [self.clean_word(w) for w in subtitle_text.replace("\n", " ").split()]
            transcribed_tokens = [self.clean_word(w.word) for w in all_words]
            log(f"S{subtitle_tokens}\n")
            log(f"T{transcribed_tokens}\n")
            start_phrase = subtitle_tokens[:3]
            end_phrase = subtitle_tokens[-3:]
            log(f"start_phrase subtitletokens: {start_phrase}\n end_phrase subtitletokens: {end_phrase}")


            match_start = -1
            for i in range(len(transcribed_tokens) - len(start_phrase) + 1):
                    if transcribed_tokens[i:i+3] == start_phrase:
                        match_start = i
                        log(f"match start transcribetokens: {match_start}")
                        break

            match_end = -1
            for j in range(len(transcribed_tokens) - len(end_phrase) + 1):
                    if transcribed_tokens[j:j+3] == end_phrase:
                        match_end = j + 3
                        log(f"match end transcribe tokens: {match_end}")
                        break

            if match_start == -1 or match_end == -1 or match_end <= match_start:
                    log("Did not find a start/end in transcript")
                    raise ValueError("Did not find a start/end in transcript")


            window_tokens = transcribed_tokens[match_start:match_end]
            log(f"[WINDOW] Index {match_start}-{match_end}: {' '.join(window_tokens)}")

            commonwords = set(subtitle_tokens) & set(window_tokens)
            similarity = len(commonwords) /len(set(subtitle_tokens)) if subtitle_tokens else 0
            log(f"Word-based similarity: {similarity:.2f}")

            if similarity < 0.6:
                log(f"Match found but similarity is too low: {similarity:.2f}")
                raise ValueError(f"Match found but similarity is too low: {similarity:.2f}")

            matched_words = [
                        {
                            "word": self.clean_word(all_words[i].word),
                            "start": float(all_words[i].start),
                            "end": float(all_words[i].end),
                        }
                        for i in range(match_start, match_end)
                    ]

            log(f"[MATCH] Found exact match: {[w['word'] for w in matched_words]}")

            final_start_time = original_start_time + float(all_words[match_start].start)
            log(f"final_start_time: {final_start_time}")


            final_end_time = original_start_time + float(matched_words[-1]["end"]) + 0.5
            log(f"final_end_time: {final_end_time}")
            log(f"matched_words: {matched_words}\n start : {final_start_time}\n end : {final_end_time}\n")

            return {
                    "matched_words": matched_words,
                    "video_start_time": final_start_time,
                    "video_end_time": final_end_time,
                   }



        except Exception as e:
            log(f"[ERROR] during transcription: {str(e)}")
            raise ValueError(f"Logic in speectotext_short_creation_thread NEEDS FIXING")

        finally:
            log(f"Transcription complete | device: {self.device}")

            if self.device == "cuda":
                torch.cuda.empty_cache()

    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs


class SpeechToTextToolCUDA(PipelineTool):
    default_checkpoint = r"c:\Users\didri\Desktop\LLM-models\Audio-Models\faster-whisper-large-v3-turbo-int8float16"
    description = "Fast tool that transcribes audio into text using faster-whisper. It returns the path to the transcript file"
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, a URL, or a tensor.",
        },
        "text_path": {
            "type": "string",
             "description": "The path to save the transcript to.",
        },
        "video_path": {
            "type": "string",
            "description": "The path to the video to transcribe. only for info printging",
        }
    }
    output_type = "string"
    def setup(self):

        self.model = WhisperModel(
                model_size_or_path=self.default_checkpoint,
                device="cuda",
                compute_type="int8_float16"
                )
    def forward(self, inputs):
        audio_path = inputs["audio"]
        text_path = inputs["text_path"]
        video_path = inputs["video_path"]
        segments, info = self.model.transcribe(
            audio_path,
            language="en",
            temperature=0.0,
            vad_filter=True,
            initial_prompt="Motivational podcast",
            condition_on_previous_text=True,

        )
        print(f"\n🔊 Using Whisper on device: {self.device}, \ntranscribing video: {video_path} \n   with inputs: {self.inputs}")
        print(f"[INFO] Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
        print(f"[INFO] Audio Duration: {info.duration:.2f} seconds")

        try:
            with open(text_path, "a", encoding="utf-8") as f:
                print(f"opening txt_path on: {text_path} device: {self.device}")

                try:
                    for segment in segments:
                         f.write(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text.strip()}\n")
                except Exception as e:
                    print(f"error during segments: {str(e)}")

        finally:
            print(f"transcription complete ! device  {self.device}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                del self.model

        return text_path

    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs





class SpeechToTextToolTEST(PipelineTool):
    default_checkpoint = r"c:\Users\didri\Desktop\LLM-models\Audio-Models\faster-whisper-large-v3-turbo-int8float16"
    description = "Fast tool that transcribes audio into text using faster-whisper. It returns the path to the transcript file"
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, a URL, or a tensor.",
        },


    }
    output_type = "string"
    def setup(self):

        self.model = WhisperModel(
                model_size_or_path=self.default_checkpoint,
                device="cuda",
                compute_type="int8_float16"
                )
    def forward(self, inputs):
        audio_path = inputs["audio"]
        print(f"Audio_path speech to text tool: {audio_path}")


        segments, info = self.model.transcribe(
            audio_path,
            initial_prompt="Motivational podcast"
        )


        Transcript = ""
        try:
            for segment in segments:
                if hasattr(segment, "text"):
                    Transcript += segment.text + " "
                else:
                    Transcript += str(segment) + " "


        except Exception as e:
                    print(f"error during segments: {str(e)}")

        finally:
            print(f"transcription complete ! device  {self.device}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()

        return Transcript.strip()

    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs
















































#################################
# Background Music Automation  #
#################################
def run_multi_Vision_model_conclusion(video_path , Additional_content,Global_model, already_uploaded_videos) -> str:
    """
    Generates a detailed motivational summary of a podcast using both text and audiovisual inputs.
    The function leverages a Vision-Audio model to process the video and audio content,
    and a text summarization pipeline to produce a concise, motivational summary.

    Args:
        _text: The transcript or textual content of the podcast to summarize.
        video_path: Path to the video file that is being uploaded of the podcast, used to extract visual frames for context.
        audio_path: Path to the audio file that is being uploaded of the podcast, used for extracting audio features for context.

    Returns:
        A string containing a detailed summary of the podcast, highlighting motivational points, emotions,
        and actionable advice. If summarization fails, a message indicating the failure is returned.
    """
    from smolagents import CodeAgent,FinalAnswerTool
    import cv2
    from PIL import Image
    import yaml

    from PIL import Image

    frames = []
    cap = cv2.VideoCapture(video_path)
    while len(frames) < 10:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    log(f"Extracted {len(frames)} frames (capped at 64).")


    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\system_prompt_background_music_analytic.yaml", "r", encoding="utf-8") as r:
            prompt = yaml.safe_load(r)

    with open(already_uploaded_videos, "r", encoding="utf-8") as f:
            uploaded_videos_content = f.read()



    agent = CodeAgent(
        model=Global_model,
        add_base_tools=True,
        verbosity_level=1,
        prompt_templates=prompt,
        tools=[FinalAnswerTool()],
        additional_authorized_imports=['typing'],
        max_steps=3,
    )

    user_message = f"""Analyze the following inputs to select the best background music:
    - Podcast Transcript: ({Additional_content["Transcript"]}) -  (Use this to identify the story, key motivational points, and overall arc.)

    - Detected Speech Emotion: ({Additional_content["emotion"]})  (Raw tone from audio model; integrate but prioritize motivational intent if conflicting.)

    - Video Frames and Audio: Observe all the  provided images for visual style, speaker energy, and pacing. Listen to the audio for vocal delivery, pauses, and intensity.

    - Candidate Background Tracks: ({Additional_content["videolist"]})  (List with metadata; select exactly one by its number.)

    - Previously Used Tracks/songs: \n({uploaded_videos_content})\n  (Avoid repeating any tracks already used in prior videos. the limit before you are allowed to reuse is 3 videos ago.)
    Follow the reasoning process step-by-step in your thinking, but output only the JSON with your final choice. Ensure the music enhances clarity, emotion, and motivation for the audience.

    -Respond ONLY with valid JSON  object inside your `final_answer` tool
        - Structure:  "path": "<exact file path from videolist>",  "song_name": "The name of the song choosen","reason": "<concise explanation, 1-3 sentences>", "editing_notes": "<optional suggestions, e.g., 'Fade in at 5s, duck volume during key speeches'>", "lut_path": <The path for choosen lut for the shorts video>"
        - The "path" field must match the "path" value from one of the tracks in the provided videolist exactly.
    """
    response =  agent.run(task=user_message, images=frames)

    return response




def predict_emotion(audio_path):
    import librosa
    import torch
    from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2ForSequenceClassification
    model_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Audio models\FadQ\results"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)

    try:
        audio, rate = librosa.load(audio_path, sr=16000)
        inputs = feature_extractor(
            audio,
            sampling_rate=rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16000 * 25
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            predictions = torch.nn.functional.softmax(logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=-1).item()
            emotion = model.config.id2label[predicted_label]
        return emotion

    except Exception as e:
        log(f"Error processing audio: {str(e)}")
        return None




def Background_Audio_Decision_Model(audio_file: str, video_path: str,already_uploaded_videos: str):
    from utility.Global_state import videolist
    from App import Reload_and_change_model

    tool = SpeechToTextToolTEST()
    tool.device = "cuda"
    tool.setup()
    try:
        Transcript = tool.forward({"audio": audio_file})
        print(f"Transcript: {Transcript}")
    except Exception as e:
        log(f"Error during SpeechToTextTool")

    try:
        emotion = predict_emotion(audio_file)
        log(f"Emotion recieved from [predict_emotion] --> {emotion}")
    except Exception as e:
        log(f"Error during prediction of emotion: {str(e)}")

    def format_videolist(videolist):
        formatted = "Candidate Background Tracks: \n"
        for i, track in enumerate(videolist):
                formatted += f"Track {i}:\n"
                formatted += f"song name: {track['song_name']}\n"
                formatted += f"  Path: {track['path']}\n"
                formatted += f"  lut_path: {track['lut_path']}\n"
                formatted += f"  Description: {track['description']}\n"
                formatted += f"  Mood: {', '.join([f'{k}: {v}' for k, v in track['mood'].items()])}\n"
                formatted += f"  Tempo: {track['tempo']}\n"
                formatted += f"  Energy: Valence={track['energy']['💖Valence']}, Arousal={track['energy']['⚡Arousal']}\n"
                formatted += f"  Genre: {track['genre']}\n"
                formatted += f"  Tags: {', '.join(track['Tags'])}\n\n"
        return formatted

    _videolist = format_videolist(videolist)

    content = {
        "Transcript": Transcript,
        "emotion": emotion,
        "videolist": _videolist
    }
    try:
         Global_model = Reload_and_change_model(model_name="gpt-5-minimal",message="Loading model: gpt-5-minimal from [Background_Audio_Decision_Model]")
         log("Starting [run_multi_Vision_model_conclusion]")
         response = run_multi_Vision_model_conclusion(
            video_path=video_path,
            Additional_content=content,
            Global_model=Global_model,
             already_uploaded_videos=already_uploaded_videos
             )
         log(f"Response from [run_multi_Vision_model_conclusion]: {response}")
    except Exception as e:
        log(f"Error during [run_multi_Vision_model_conclusion]: {str(e)}")
    from utility.clean_memory import clean_get_gpu_memory
    clean_get_gpu_memory(threshold=0.1)
    del Global_model


    return response































































# async def  detect_music_in_video(audio_file: str):
#     from shazamio import Shazam
#     shazam = Shazam()
#     try:
#         result = await shazam.recognize(audio_file)
#         track = result.get('track')

#         if track:
#             title = track.get("title", "Unknown")
#             artist = track.get("artist", "Unknown")
#             log(f"shazam: Title: {title} \n Artist: {artist}")
#             return title, artist

#         elif track is None or track.get("title") is None:
#             log("SHAZAM failed, falling back to YouTube search using video title")
#             music_title = "Unknown"
#             music_artist = "Unknown"
#             return music_title, music_artist
#     except Exception as e:
#         log(f"Error inside: [detect_music_in_video] -> {str(e)}")





# def isolate_audiofile(audio_file: str, save_folder: str = None):
#     """
#     Demucs model that seperate the vocals and music, and returns the music
#     """
#     from demucs.pretrained import get_model
#     from demucs.audio import AudioFile
#     from demucs.separate import apply_model
#     import soundfile as sf
#     import numpy as np
#     model = get_model('mdx')
#     model.cpu()
#     model.eval()
#     try:
#         wav = AudioFile(audio_file).read(streams=0, samplerate=model.samplerate)
#         wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

#         with torch.no_grad():
#             sources = apply_model(model, wav, device='cpu')

#         accompaniment = sources[0, [0, 1, 2]].sum(dim=0).cpu().numpy()
#         output_file = os.path.join(save_folder,"accompaniment.wav")
#         sf.write(output_file, accompaniment.T, model.samplerate)
#     except Exception as e:
#         log(f"Error inside [isolate_audiofile] -> {str(e)}")
#     return output_file





# def Download_Music_from_youtube(music_info: dict):
#     if isinstance(music_info, tuple):
#         title, artist = music_info
#     else:
#         title = music_info.get("title")
#         artist = music_info.get("artist")

#     if title == "Unknown":
#         print("Shazam failed, skipping YouTube music download")
#         return None

#     query = title
#     if artist:
#         query += f" {artist}"

#     output_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio"
#     ytdl =  {
#             "outtmpl": f'{output_path}/%(title)s.%(ext)s',
#             "cookiefile": cookie_file_path,
#             'format': f"bestaudio/best",
#             'nocheckcertificate': True,
#             "restrictfilenames": True,
#             'quiet': False,
#             '--no-playlist': True,
#             "default_search": "ytsearch1",
#             'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'wav',
#             'preferredquality': '0',
#         }]
#         }
#     try:
#         with yt_dlp.YoutubeDL(ytdl) as ydl:
#             info = ydl.extract_info(query, download=True)
#             if 'requested_downloads' in info:

#                  file_path = info['requested_downloads'][0]['filepath']
#             elif 'entries' in info and len(info['entries']) > 0:
#                 file_path = info['entries'][0]['requested_downloads'][0]['filepath']

#             else:
#                 raise ValueError("Could not find downloaded file path ")
#             log(f"Music downloaded from YouTube to: {file_path}")
#             return file_path
#     except yt_dlp.utils.DownloadError as e:
#             log(f"[YT_dlp] ERROR: {str(e)}")
#             return None


# def detect_Music_with_Audd(audio_file: str,original_url = None, verbose: bool = True):
#     load_dotenv()
#     api_key = os.getenv("AAUDD_APIKEY")
#     if not api_key:
#         raise ValueError("Api key is missing!")

#     url = "https://api.audd.io/"

#     data = {
#         'api_token': api_key,
#         'return': 'apple_music,spotify,is_instrumental'
#     }

#     files = None
#     if audio_file:
#         files = {'file': open(audio_file, 'rb')}
#     if original_url:
#         data['url'] = original_url

#     response = requests.post(url, data=data, files=files)
#     result = response.json()

#     if verbose:
#         log("=== AUD raw response===")
#         log(json.dumps(result, indent=2))

#     if result['status'] == 'success' and result.get('result'):
#         data = result['result']
#         title = data.get('title', 'Unknown')
#         artist = data.get('artist', 'Unknown')
#         instrumental = data.get('is_instrumental', False)
#         spotify_link = data.get('spotify', {}).get('external_urls', {}).get('spotify')
#         apple_link = data.get('apple_music', {}).get('url')

#         if verbose:
#             log("=== AudD Parsed Result ===")
#             log(f"Title: {title}")
#             log(f"Artist: {artist}")
#             log(f"Is instrumental? {instrumental}")
#             log(f"Spotify: {spotify_link}")
#             log(f"Apple Music: {apple_link}")
#         return data
#     else:
#         if verbose:
#            log("Song not found or recognition failed")
#         return None





# def download_youtube_Music_Audio(Youtube_url: str,save_folder: str):
#     """Downloads the Audio file from a youtube video and detects the music name. Downloads the music and returns it
#         Args:
#             Youtube_url (str): Path to the youtube video

#     """
#     ytdl =  {
#             "outtmpl": os.path.join(save_folder, "%(title)s.%(ext)s"),
#             "cookiefile": cookie_file_path,
#             'format': f"bestaudio/best",
#             "restrictfilenames": True,
#             'nocheckcertificate': True,
#             'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'wav',
#             'preferredquality': '0',
#         }]
#         }
#     try:
#         with yt_dlp.YoutubeDL(ytdl) as ydl:
#                info = ydl.extract_info(Youtube_url,download=True)
#                audio_file_path = info['requested_downloads'][0]['filepath']
#                print(f"Video audio downloaded to: {audio_file_path}")
#     except yt_dlp.utils.DownloadError as e:
#             log(f"[YT_dlp] ERROR: {str(e)}")
#             return None

#     try:
#         accompaniment_path = isolate_audiofile(audio_file_path, save_folder)
#         log(f"only music/intstrumental path: {accompaniment_path}")
#     except Exception as e:
#         log(f"Error during: [isolate_audiofile]: {str(e)}")


#     import asyncio
#     log("Trying to detect music with Shazam.")
#     try:
#         music_title, music_artist = asyncio.run(detect_music_in_video(accompaniment_path))
#     except Exception as e:
#         log(f"Error during [detect_music_in_video]: {str(e)}")

#     if not music_title:
#         log("Shazam could not detect music. Skipping music download.")
#         return audio_file_path, accompaniment_path, None
#     else:
#         music_info_dict = { "title": music_title, "music_artist": music_artist }


#     try:
#        music_file_path  = Download_Music_from_youtube(music_info_dict)
#     except Exception as e:
#         log(f"Error during [Download_Music_from_youtube]: {str(e)}")

#     if music_file_path  is None:
#         log("Trying to detect with (AUUD) now...")
#         try:
#             Auud_music = detect_Music_with_Audd(accompaniment_path,Youtube_url)
#             if Auud_music is None:
#                 music_file_path  = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\way down we go (instrumental) - kaleo [edit audio] [mp3].mp3"
#         except Exception as e:
#             log(f"Error during [detect_Music_with_Audd]: {str(e)}")
#             music_file_path  = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\way down we go (instrumental) - kaleo [edit audio] [mp3].mp3"


#     log(f"Final Music path: {music_file_path}")

#     return  music_file_path



# @tool
# def Read_already_uploaded_video_publishedat(file_path: str) -> str:
#     """A tool that returns information about all videos that are published already. data like (title, description, tags, PublishedAt).
#         This tool is useful too gather information about future video PublishedAt/Time scheduling .
#         Args:
#         file_path (str): The path to already_uploaded file
#         Returns: str "string"
#     """
#     try:

#         with open(file_path, "r", encoding="utf-8") as f:
#             content = f.read()
#         return content
#     except FileNotFoundError as e:
#         return "No uploaded video data found."
#     except Exception as e:
#             return f"Error reading uploaded video data: {str(e)}"


