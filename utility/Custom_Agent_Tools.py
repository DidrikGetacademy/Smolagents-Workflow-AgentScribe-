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
from utility.log import log
import utility.Global_state as Global_state
import gc
import tempfile
import yt_dlp
import requests
import json
from typing import List ,Dict
cookie_file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Secrets\youtube.com_cookies.txt"
load_dotenv()

@tool
def read_file(text_file: str) -> str:
    """
    A tool that reads and returns the content of a text file.
    Args:
        text_file (str): The path to the text file.
    Return:
        str: Publihing date of the latest video uploaded.
    """
    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read()[:3000]
    return content





####FETCH MORE DETAILS TOO PROVIDE AGENT WITH MORE INFORMATION####
@tool
def Fetch_top_trending_youtube_videos(Search_Query: str) -> dict:
    """
        A tool for Fetching enriched metadata + stats for the top trending YouTube videos for a query, including category names, tags, duration, views, likes, comments, and channel stats.
        IMPORTANT: Avoid overly specific or long queries with many distinct concepts.
        If the search returns no results, it means the query was too specific.
        The tool will return a message indicating this, and you should retry with a broader, simpler query.

        BEST PRACTICES FOR QUERY:
        - Keep it short (1-3 keywords).
        - Focus on the core topic from the intent of the transcript (e.g.,"Motivation", "Discipline").
        - Avoid combining unrelated concepts (e.g., "motivation proof over validation discipline action bias").
        - Broad queries yield the best trending results.

        Args:
        Search_Query (str): Topic or keywords to search (e.g. “Motivational”, “Tech Reviews”).

        Returns:
        dict: A YouTube API response containing for each video:
        - snippet: title, description, channelTitle, publishTime, thumbnails
        - statistics: viewCount, likeCount, commentCount
    """

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
            maxResults=6
        ).execute()

    items = search_resp.get("items", [])


    #Extracts the videoId of each video in items
    video_ids = [item["id"]["videoId"] for item in items]

    #Early exit if no videos is found!
    if not video_ids:
        return {"items": [], "message": f"No videos found for query: '{Search_Query}'. The query may be too specific. Try call the tool again but  reduce the number of keywords."}


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




    def forward(self, file_path: str, max_chars: int) -> str:


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
        chunk = ""
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
def Read_transcript(transcript_path: str, start_count: int = 0, chunk_size: int = 1000) -> dict:
    """Reads a chunk from a transcript starting at a given position.
    Args:
        transcript_path (str): Path to the transcript file.
        start_count (int): Byte offset to start reading from.
        chunk_size (int): Number of characters to read (default 1000).

    Returns:
        dict: {
            "text": str,                 # the transcript chunk
            "next_start": int,           # position to use for next call
            "has_more": bool             # whether more content remains
        }
    """
    with open(transcript_path, "r", encoding="utf-8") as file:
        file.seek(start_count)
        content = file.read(chunk_size + 1)

    has_more = len(content) > chunk_size
    text = content[:chunk_size] if has_more else content
    next_start = start_count + len(text)

    return {
        "text": text,
        "next_start": next_start,
        "has_more": has_more,
    }






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
def create_motivationalshort(text: str,text_file: str) -> None:
        """
        Save a single, self-contained motivational passage for short-video creation.

        Rules (summary):
        - Verbatim only: copy text exactly as in the transcript; no paraphrasing, trimming, or cleanup.
        - Timestamps: include every "[SSSS.SSs - SSSS.SSs]" segment in order. If multi-line, merge into one string preserving order and spacing.
        - One call per passage: do not split a coherent passage across multiple calls.
        - Pass `text_file` by variable name (e.g., text_file=text_file), not a literal path.

        Example:
            create_motivationalshort(
                text="[1478.88s - 1483.00s] Line 1 [1483.58s - 1489.74s] Line 2",
                text_file=text_file,
            )

        Args:
            text (str): Complete passage including all required timestamps
            text_file (str): Path to the text file where the passage will be appended.

        """
        with open (text_file, "a", encoding="utf-8") as f:
            log(f"Writing saved motivational text from agent to: {text_file}\n text: {text}")
            f.write(f"===START_TEXT===")
            f.write(f"{text}")
            f.write(f"===END_TEXT===\n")
        try:
            start_time, end_time, new_text = parse_multiline_block(text)
            log(f"start time: {start_time}, end_time: {end_time}, new_text: {new_text}")
        except Exception as e:
             log(f"Error during [parse_multiline_block]: {str(e)}")


        if start_time is None or end_time is None:
             raise ValueError(f"start_time or end_time is None, start_time: {start_time}, end_time: {end_time}")

        video_url = Global_state.get_current_videourl()
        audio_path = Global_state.get_current_audio_path()
        try:
            log(f"\n now Added work to QUEUE: \n video url: {video_url}\n, audio_path: {audio_path} \n start_time: {start_time}s \n end_time {end_time}s\nOriginal text: {text}")

         #   Global_state.video_task_que.put((video_url, audio_path, start_time, end_time, new_text))
            count = Global_state.get_current_count()
            count +=1
            log(f"Added VideoWork to videotask Queue:\n{video_url}\nAudio_path:{audio_path}\n{start_time}\n {end_time}\n {new_text}\n  Amount of added videowork to queue: {count}\n ")

            log(f"Current videos added to que for proccessing: {count}")
            Global_state.set_current_count(count)


        except Exception as e:
             log(f"Error addng to queue: {str(e)}")















@tool
def open_work_file(work_queue_folder: str) -> str:
     """A tool that returns motivational text  for each of the work folders saved by another agent.
    Args:
        work_queue_folder: The path or name of the folder containing queued work.
     """
     text_list = []
     count = 1
     import re
     text_list.append(f"Below are the motivational snippets found in the work queue folder. : {work_queue_folder}")
     for subdir in os.listdir(work_queue_folder):
          subdir_path = os.path.join(work_queue_folder,subdir)
          if os.path.isdir(subdir_path):
            copy_path = os.path.join(subdir_path, "agent_saving_path.txt")
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
                - "middle_order" (int, optional): If `order` is "middle" and there are multiple middle clips, this indicates the sequence of the middle clips (1 for first middle clip, 2 for second, etc.). Omit or set to None if not applicable.
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
            run_montage_short_creation([{"audio_path": r"...","order": "start","video_url": r"...","text": "...","ID": 1.1},{"audio_path": r"...","order": "middle","video_url": r"...","text": "...","ID": 1.2,"middle_order": 1},{"audio_path": r"...","order": "middle","video_url": r"...","text": "...","ID": 3.2,"middle_order": 2},{"audio_path": r"...","order": "ending","video_url": r"...","text": "...","ID": 1.3}])
            run_montage_short_creation([{"audio_path": r"...","order": "start","video_url": r"...","text": "...","ID": 2.1},{"audio_path": r"...","order": "middle","video_url": r"...","text": "...","ID": 2.2, "middle_order": 1},{"audio_path": r"...","order": "middle","video_url": r"...","text": "...","ID": 3.2,"middle_order": 2},{"audio_path": r"...","order": "ending","video_url": r"...","text": "...","ID": 2.3}])
            run_montage_short_creation([{"audio_path": r"...","order": "start","video_url": r"...","text": "...","ID": 3.1},{"audio_path": r"...","order": "middle","video_url": r"...","text": "...","ID": 3.2,"middle_order": 1},{"audio_path": r"...","order": "middle","video_url": r"...","text": "...","ID": 3.2,"middle_order": 2},{"audio_path": r"...","order": "ending","video_url": r"...","text": "...","ID": 3.3}])
        """
        items_queued = 0
        for item in montage_list:
             video_path = item["video_url"]
             middle_order = item["middle_order"]
             audio = item["audio_path"]
             order = item["order"]
             text = item["text"]
             Montage_ID = item["ID"]
             YT_channel = item["YT_channel"]
             log(f"\n[montage_short_creation_tool] Item in montage_list:\n video_path: {video_path}, \n audio: {audio}, \n order: {order}, \n text: {text}, \n Montage_ID: {Montage_ID}, \n YT_channel: {YT_channel}, middle_order: {middle_order} ")

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
                    YT_channel,
                    middle_order
                ))
                items_queued += 1
             except Exception as e:
                log(f"Error adding to queue: {str(e)}")

             log(f"Added to Montage_clip_task_Que: video_path: {video_path}, audio: {audio}, start_time: {start_time}, end_time: {end_time}, order: {order}, Montage_ID: {Montage_ID}")
        log(f"items queued: {items_queued}")



@tool
def Check_Already_used_videos(video_id: str, channel_name: str) -> bool:
    """
    A tool that checks if a YouTube video ID has already been used by reading from a JSON file.

    Args:
        video_id (str): The YouTube video ID to check.
        channel_name (str): The name of the channel.
    Returns:
        bool: True if the video ID has been used, False otherwise.
    """
    import json
    youtube_folder = os.path.join(REPO_ROOT, "Youtube", channel_name)
    json_file = os.path.join(youtube_folder, "used_videos.json")

    if not os.path.exists(json_file):
        return False

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            used_videos = json.load(f)

        for video in used_videos:
            if video.get("video_id") == video_id:
                return True
    except Exception as e:
        log(f"Error reading used_videos.json: {e}")
        return False

    return False


@tool
def update_used_videos(video_id: str, channel_name: str, title: str, published_at: str, description: str) -> None:
    """
    A tool that updates the record of used YouTube video IDs in a JSON file.

    Args:
        video_id (str): The YouTube video ID to add to the used list.
        channel_name (str): The name of the channel.
        title (str): The title of the video.
        published_at (str): The publication date.
        description (str): The video description.
    Returns:
        None
    """
    import json
    import time

    youtube_folder = os.path.join(REPO_ROOT, "Youtube", channel_name)
    os.makedirs(youtube_folder, exist_ok=True)

    json_file = os.path.join(youtube_folder, "used_videos.json")

    used_videos = []
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                used_videos = json.load(f)
        except Exception:
            used_videos = []


    if any(v.get("video_id") == video_id for v in used_videos):
        log(f"Video {video_id} already exists in records for {channel_name}.")
        return

    new_entry = {
        "video_id": video_id,
        "title": title,
        "published_at": published_at,
        "description": description,
        "added_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    used_videos.append(new_entry)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(used_videos, f, indent=4, ensure_ascii=False)

    log(f"Updated used_videos.json for {channel_name} with video {video_id}")







@tool
def youtube_Searcher(channel_name: str, search_query: str = None, max_results: int = 10, order: str = "relevance") -> dict:
    """
    A tool that retrieves information about videos from a specific YouTube channel.
    It can either list the latest uploads or search for videos matching a query within the channel.

    Args:
        channel_name (str): The name of the channel to look up (e.g., "Chris_williamson", "Jay_shetty", "Mel Robbins").
        search_query (str, optional): A keyword or topic to search for within the channel's videos. If None, fetches latest uploads (unless order is 'viewCount' or 'rating').
        max_results (int, optional): The maximum number of videos to return. Defaults to 10.
        order (str, optional): The order of the search results. Allowed values: 'date', 'rating', 'relevance', 'title', 'videoCount', 'viewCount'.
                               Defaults to 'relevance'. Use 'viewCount' to find the most popular videos.

    Returns:
        dict: A dictionary containing channel statistics and a list of videos (with titles, descriptions, etc.).
    """
    Youtube_channels = {
        "Chris_williamson": 'UCIaH-gZIVC432YRjNVvnyCA',
        "Jay_shetty": 'UCbk_QsfaFZG6PdQeCvaYXJQ',
        'Mel_Robbins': 'UCk2U-Oqn7RXf-ydPqfSxG5g',
        'Diary_ceo': 'UCGq-a57w-aPwyi3pW7XLiHw',
    }

    if channel_name not in Youtube_channels:
        return {"error": f"Channel '{channel_name}' not found in the available list. Available channels: {list(Youtube_channels.keys())}"}

    channel_id = Youtube_channels[channel_name]

    Api_key = os.getenv("YOUTUBE_API_KEY")
    if not Api_key:
        raise ValueError("YOUTUBE_API_KEY is not set in environment variables.")

    youtube = build("youtube", "v3", developerKey=Api_key)

    channel_response = youtube.channels().list(
        part="snippet,statistics,contentDetails",
        id=channel_id
    ).execute()

    if not channel_response.get("items"):
        return {"error": f"No details found for channel ID {channel_id}"}

    channel_info = channel_response["items"][0]
    uploads_playlist_id = channel_info["contentDetails"]["relatedPlaylists"]["uploads"]

    stats = {
        "title": channel_info["snippet"]["title"],
        "description": channel_info["snippet"]["description"],
        "subscriberCount": channel_info["statistics"]["subscriberCount"],
        "videoCount": channel_info["statistics"]["videoCount"],
        "viewCount": channel_info["statistics"]["viewCount"]
    }

    videos = []


    use_search_api = (search_query is not None) or (order in ["viewCount", "rating", "title", "videoCount"])

    if use_search_api:
        # Search within the channel
        search_params = {
            "part": "snippet",
            "channelId": channel_id,
            "maxResults": max_results,
            "order": order,
            "type": "video"
        }
        if search_query:
            search_params["q"] = search_query

        search_response = youtube.search().list(**search_params).execute()

        for item in search_response.get("items", []):
            videos.append({
                "title": item["snippet"]["title"],
                "videoId": item["id"]["videoId"],
                "publishedAt": item["snippet"]["publishedAt"],
                "description": item["snippet"]["description"]
            })
    else:

        playlist_response = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=max_results
        ).execute()

        for item in playlist_response.get("items", []):
            videos.append({
                "title": item["snippet"]["title"],
                "videoId": item["contentDetails"]["videoId"],
                "publishedAt": item["snippet"]["publishedAt"],
                "description": item["snippet"]["description"]
            })

    return {
        "channel_statistics": stats,
        "videos": videos
    }





@tool
def youtube_downloader(VideoId: int) -> str:
    """
    A tool that Downloads video based on a YouTube Video ID.

    Args:
        VideoId (int): The ID of the YouTube video to download.

    Returns:
        str: The path to the downloaded video file.
    """
    yt_dlp_opts = {
        'quiet': False,
        'nocheckcertificate': True,
        'format':'best',
        'debuge': True,
        "cookiefile": '../youtube.com_cookies.txt',
        'cookiesfrombrowser': True,
        'extractor_args':{
            'youtube': {
                'player_client': ['default','mweb'],
            }
        }
    }
    with yt_dlp.YoutubeDL(yt_dlp_opts) as ytdlp:
        try:
            video_url = f"https://www.youtube.com/watch?v={VideoId}"
            info_dict = ytdlp.extract_info(video_url, download=True)
            video_title = info_dict.get('title', None)
            sanitized_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '_', '-')).rstrip()
            output_path = ytdlp.prepare_filename(info_dict)
            new_output_path = os.path.join(os.path.dirname(output_path), f"{sanitized_title}.mp4")
            os.rename(output_path, new_output_path)
            log(f"Downloaded video to: {new_output_path}")
            return new_output_path
        except ytdlp.utils.DownloadError as e:
            log(f"DownloadError: {str(e)}")
            return None







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
                    "threshold": 0.6,
                    "min_silence_duration_ms": 400,
                    "min_speech_duration_ms": 250,
                    "speech_pad_ms": 100,
                    },
                no_speech_threshold=0.6,
                initial_prompt=subtitle_text
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

            matched_words = [
                        {
                            "word": self.clean_word(all_words[i].word),
                            "start": float(all_words[i].start),
                            "end": float(all_words[i].end),
                        }
                        for i in range(match_start, match_end)
                    ]

            log(f"[MATCH] Found exact match: {[w['word'] for w in matched_words]}")

            final_start_time = original_start_time + float(all_words[match_start].start) - 0.2
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
            device="auto",
            compute_type="int8_float16"
            )


    def forward(self, inputs):
        audio_path = inputs["audio"]
        print(f"Audio_path speech to text tool: {audio_path}")

        try:
            segments, _= self.model.transcribe(
                audio_path,
                initial_prompt="Motivational podcast",
                language="en",
                temperature=0.0,
                vad_filter=True,
            )


            Transcript = ""
            for segment in segments:
                log(f" text: {segment.text.strip()}")
                if hasattr(segment, "text"):
                    Transcript += segment.text + " "
                else:
                    print(f"Segment missing text attribute: {segment}")
        except Exception as e:
             log(f"Error during transcription: {str(e)}")

        finally:
            log(f"transcription complete")
            log(f"Transcript: {Transcript}")
            del self.model

        return Transcript.strip()

    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs







def extract_window_frames(video_path: str, start_time: float, end_time: float, max_frames: int = 5):
    import cv2
    from PIL import Image
    import numpy as np
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start_f = int(start_time * fps)
    end_f = int(end_time * fps)
    if end_f <= start_f:
        cap.release()
        return []
    indices = np.linspace(start_f, end_f - 1, num=max_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    log(f"Extracted {len(frames)} frames in window [{start_time:.2f}s - {end_time:.2f}s].")
    return frames




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




def Background_Audio_Decision_Model(audio_file: str, video_path: str,already_uploaded_videos: str,start_time: float,end_time: float):
    from utility.Global_state import Music_list
    from App import Reload_and_change_model

    try:
        tool = SpeechToTextToolTEST()
        tool.setup()
        Transcript = tool.forward({"audio": audio_file})
        del tool
        log(f"Transcript: {Transcript}")
    except Exception as e:
        log(f"Error during SpeechToTextTool {str(e)}")

    try:
        emotion = predict_emotion(audio_file)
        log(f"Emotion recieved from [predict_emotion] --> {emotion}")
    except Exception as e:
        log(f"Error during prediction of emotion: {str(e)}")

    def format_videolist(Music_list):
        formatted = "Candidate Background Tracks: \n"
        for i, track in enumerate(Music_list):
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

    _videolist = format_videolist(Music_list)

    content = {
        "Transcript": Transcript,
        "emotion": emotion,
        "videolist": _videolist
    }
    response = {}
    try:
         Global_model = Reload_and_change_model(model_name="gpt-5-minimal",message="Loading model: gpt-5-minimal from [Background_Audio_Decision_Model]")
         log("Starting [run_multi_Vision_model_conclusion]")
         from Agents.Vision_agent import run_multi_Vision_model_agent
         response = run_multi_Vision_model_agent(
            video_path=video_path,
            Additional_content=content,
            Global_model=Global_model,
             already_uploaded_videos=already_uploaded_videos,
             start_time=start_time,
             end_time=end_time
             )
         log(f"Response from [run_multi_Vision_model_conclusion]: {response}")
    except Exception as e:
        log(f"[Background_Audio_Decision_Model] Error during run_multi_Vision_model_agent: {str(e)}")
    from utility.clean_memory import clean_get_gpu_memory
    clean_get_gpu_memory(threshold=0.1)
    if 'Global_model' in locals():
        del Global_model


    log(f"Finished [Background_Audio_Decision_Model] with response: {response}")


    return response








async def detect_music_in_video(audio_file: str):
    from shazamio import Shazam
    shazam = Shazam()
    try:
        result = await shazam.recognize(audio_file)
        track = result.get('track')

        if track:
            title = track.get("title", "Unknown")
            artist = track.get("artist", "Unknown")
            log(f"shazam: Title: {title} \n Artist: {artist}")
            return title, artist

        elif track is None or track.get("title") is None:
            log("SHAZAM failed, falling back to YouTube search using video title")
            music_title = "Unknown"
            music_artist = "Unknown"
            return music_title, music_artist
    except Exception as e:
        log(f"Error inside: [detect_music_in_video] -> {str(e)}")





def isolate_audiofile(audio_file: str, save_folder: str = None):
    """
    Demucs model that seperate the vocals and music, and returns the music
    """
    from demucs.pretrained import get_model
    from demucs.audio import AudioFile
    from demucs.separate import apply_model
    import soundfile as sf
    import numpy as np
    model = get_model('mdx')
    model.cpu()
    model.eval()
    try:
        wav = AudioFile(audio_file).read(streams=0, samplerate=model.samplerate)
        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            sources = apply_model(model, wav, device='cpu')

        accompaniment = sources[0, [0, 1, 2]].sum(dim=0).cpu().numpy()
        output_file = os.path.join(save_folder,"accompaniment.wav")
        sf.write(output_file, accompaniment.T, model.samplerate)
    except Exception as e:
        log(f"Error inside [isolate_audiofile] -> {str(e)}")
    return output_file





def Download_Music_from_youtube(music_info: dict):
    if isinstance(music_info, tuple):
        title, artist = music_info
    else:
        title = music_info.get("title")
        artist = music_info.get("artist")

    if title == "Unknown":
        print("Shazam failed, skipping YouTube music download")
        return None

    query = title
    if artist:
        query += f" {artist}"

    output_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio"
    ytdl =  {
            "outtmpl": f'{output_path}/%(title)s.%(ext)s',
            "cookiefile": cookie_file_path,
            'format': f"bestaudio/best",
            'nocheckcertificate': True,
            "restrictfilenames": True,
            'quiet': False,
            '--no-playlist': True,
            "default_search": "ytsearch1",
            'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }]
        }
    try:
        with yt_dlp.YoutubeDL(ytdl) as ydl:
            info = ydl.extract_info(query, download=True)
            if 'requested_downloads' in info:

                 file_path = info['requested_downloads'][0]['filepath']
            elif 'entries' in info and len(info['entries']) > 0:
                file_path = info['entries'][0]['requested_downloads'][0]['filepath']

            else:
                raise ValueError("Could not find downloaded file path ")
            log(f"Music downloaded from YouTube to: {file_path}")
            return file_path
    except yt_dlp.utils.DownloadError as e:
            log(f"[YT_dlp] ERROR: {str(e)}")
            return None


def detect_Music_with_Audd(audio_file: str,original_url = None, verbose: bool = True):
    load_dotenv()
    api_key = os.getenv("AAUDD_APIKEY")
    if not api_key:
        raise ValueError("Api key is missing!")

    url = "https://api.audd.io/"

    data = {
        'api_token': api_key,
        'return': 'apple_music,spotify,is_instrumental'
    }

    files = None
    if audio_file:
        files = {'file': open(audio_file, 'rb')}
    if original_url:
        data['url'] = original_url

    response = requests.post(url, data=data, files=files)
    result = response.json()

    if verbose:
        log("=== AUD raw response===")
        log(json.dumps(result, indent=2))

    if result['status'] == 'success' and result.get('result'):
        data = result['result']
        title = data.get('title', 'Unknown')
        artist = data.get('artist', 'Unknown')
        instrumental = data.get('is_instrumental', False)
        spotify_link = data.get('spotify', {}).get('external_urls', {}).get('spotify')
        apple_link = data.get('apple_music', {}).get('url')

        if verbose:
            log("=== AudD Parsed Result ===")
            log(f"Title: {title}")
            log(f"Artist: {artist}")
            log(f"Is instrumental? {instrumental}")
            log(f"Spotify: {spotify_link}")
            log(f"Apple Music: {apple_link}")
        return data
    else:
        if verbose:
           log("Song not found or recognition failed")
        return None





def download_youtube_Music_Audio(Youtube_url: str,save_folder: str):
    """Downloads the Audio file from a youtube video and detects the music name. Downloads the music and returns it
        Args:
            Youtube_url (str): Path to the youtube video

    """
    ytdl =  {
            "outtmpl": os.path.join(save_folder, "%(title)s.%(ext)s"),
            "cookiefile": cookie_file_path,
            'format': f"bestaudio/best",
            "restrictfilenames": True,
            'nocheckcertificate': True,
            'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }]
        }
    try:
        with yt_dlp.YoutubeDL(ytdl) as ydl:
               info = ydl.extract_info(Youtube_url,download=True)
               audio_file_path = info['requested_downloads'][0]['filepath']
               print(f"Video audio downloaded to: {audio_file_path}")
    except yt_dlp.utils.DownloadError as e:
            log(f"[YT_dlp] ERROR: {str(e)}")
            return None

    try:
        accompaniment_path = isolate_audiofile(audio_file_path, save_folder)
        log(f"only music/intstrumental path: {accompaniment_path}")
    except Exception as e:
        log(f"Error during: [isolate_audiofile]: {str(e)}")


    import asyncio
    log("Trying to detect music with Shazam.")
    try:
        music_title, music_artist = asyncio.run(detect_music_in_video(accompaniment_path))
    except Exception as e:
        log(f"Error during [detect_music_in_video]: {str(e)}")

    if not music_title:
        log("Shazam could not detect music. Skipping music download.")
        return audio_file_path, accompaniment_path, None
    else:
        music_info_dict = { "title": music_title, "music_artist": music_artist }


    try:
       music_file_path  = Download_Music_from_youtube(music_info_dict)
    except Exception as e:
        log(f"Error during [Download_Music_from_youtube]: {str(e)}")

    if music_file_path  is None:
        log("Trying to detect with (AUUD) now...")
        try:
            Auud_music = detect_Music_with_Audd(accompaniment_path,Youtube_url)
            if Auud_music is None:
                music_file_path  = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\way down we go (instrumental) - kaleo [edit audio] [mp3].mp3"
        except Exception as e:
            log(f"Error during [detect_Music_with_Audd]: {str(e)}")
            music_file_path  = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\way down we go (instrumental) - kaleo [edit audio] [mp3].mp3"


    log(f"Final Music path: {music_file_path}")

    return  music_file_path



@tool
def Read_already_uploaded_video_publishedat(file_path: str) -> str:
    """A tool that returns information about all videos that are published already. data like (title, description, tags, PublishedAt).
        This tool is useful too gather information about future video PublishedAt/Time scheduling .
        Args:
        file_path (str): The path to already_uploaded file
        Returns: str "string"
    """
    try:

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError as e:
        return "No uploaded video data found."
    except Exception as e:
            return f"Error reading uploaded video data: {str(e)}"


@tool
def check_video_source_exists(video_name: str) -> str:
    """
    Checks if a video source name exists in the completed videos database.
    This tool searches the Videosources_completed.json file to see if a specific video has already been used.

    Args:
        video_name (str): The exact name of the video to check (e.g., "Motivational Speech.mp4")

    Returns:
        str: A message indicating whether the video exists and its metadata if found.
    """
    json_path = r"c:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\Videosources_completed.json"

    try:
        if not os.path.exists(json_path):
            return f"Database file not found. Video '{video_name}' does not exist in database."

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = video_name.strip()

        if video_name in data:
            metadata = data[video_name]
            used_date = metadata.get("used_date", "unknown")
            clips_created = metadata.get("clips_created", 0)
            return f"Video '{video_name}' EXISTS in database. Used on: {used_date}, Clips created: {clips_created}"
        else:
            return f"Video '{video_name}' does NOT exist in database."

    except json.JSONDecodeError as e:
        return f"Error parsing database file: {str(e)}"
    except Exception as e:
        return f"Error checking video source: {str(e)}"

