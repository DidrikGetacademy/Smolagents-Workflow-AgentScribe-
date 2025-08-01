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
from log import log 





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
        raise ValueError(f"error api key is not in enviorment variables: {str(e)}")

    #Searches for videos related too the (search query) retrieves basic info of each video. (20 results)
    search_resp = youtube.search().list(
            part="snippet",
            q=Search_Query,
            type="video",
            regionCode="US",
            order="viewCount",   # Sort by most viewed = trending within the query
            maxResults=1
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

            # If adding this line would exceed the limit AND we already have at least one line, stop
            if total_len + line_len > max_chars and chunk_lines:
                break

            # Otherwise, add the line
            chunk_lines.append(line)
            total_len += line_len
            i += 1

        chunk = "".join(chunk_lines)

        # Save remaining lines back to file
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
            log(f"Extracted audio to: {audio_path}")
        except subprocess.CalledProcessError:
            log(f"Failed to extract audio from {video_path}")
            continue

        # Transcribe the audio
        try:
            result_txt_path = tool.forward({"audio": audio_path})
            # Optionally rename the transcript to desired name
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
        content = file.read(chunk_size + 1)  # Read 1 extra char to check if more content exists

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
    




class SpeechToTextTool_viral_agent(PipelineTool):

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
                device="cpu",
                compute_type="int8_float16",
    
                    )              



    def forward(self, inputs):
        audio_path = inputs["audio"]
        segments = self.model.transcribe(
            audio_path,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )


        try:
            result = []
            for segment in segments:
                    result.append(f"{segment.text.strip()}\n")
        except Exception as e:
            log(f"error during transcribing")      
        finally:
            del self.model 
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return result

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
                device="cuda",
                compute_type="int8_float16",
    
                    )              



    def forward(self, inputs):
        audio_path = inputs["audio"]
        segments = self.model.transcribe(
            audio_path,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 100},
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
                cpu_threads=4 
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
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
        print(f"\n🔊 Using Whisper on device: {self.device}, \ntranscribing video: {video_path} \n   with inputs: {self.inputs}")
        print(f"[INFO] Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
        print(f"[INFO] Audio Duration: {info.duration:.2f} seconds")
        
        try:
            with open(text_path, "w", encoding="utf-8") as f:
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
        from log import log

        try:
            segments, info = self.model.transcribe(
                audio_path,
                word_timestamps=True
            )

            print(f"[INFO] Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
            print(f"[INFO] Audio Duration: {info.duration:.2f} seconds")

            # Flatten all words
            last_word_match = False
            all_words = []
            for segment in segments:
                all_words.extend(segment.words)

            # Cleaned text
            subtitle_tokens = [self.clean_word(w) for w in subtitle_text.replace("\n", " ").split()]
            transcribed_tokens = [self.clean_word(w.word) for w in all_words]

            # Sliding window exact match
            match_start = -1
            for i in range(len(transcribed_tokens) - len(subtitle_tokens) + 1):
                window = transcribed_tokens[i:i + len(subtitle_tokens)]
                if window == subtitle_tokens:
                    match_start = i
                    break

            if match_start != -1:
                match_end = match_start + len(subtitle_tokens)
                matched_words = [
                    {
                        "word": all_words[i].word,
                        "start": float(all_words[i].start),
                        "end": float(all_words[i].end)
                    }
                    for i in range(match_start, match_end)
                ]
                print(f"[MATCH] Found exact match: {[w['word'] for w in matched_words]}")

                new_video_start_time = float(matched_words[0]["start"])
                new_video_end_time = float(matched_words[-1]["end"])

                subtitle_last_word  = self.clean_word(subtitle_tokens[-1])
                audio_transcript_last_word = self.clean_word(transcribed_tokens[-1])



                if subtitle_last_word == audio_transcript_last_word:
                   last_word_match = True

            else:
                raise ValueError("ERRROOOOOOOOOOOOOORR")

            log(f"Matched words from speectotexttool: {matched_words}\n\n")
            return {
                "matched_words": matched_words,
                "video_start_time": new_video_start_time,
                "video_end_time": new_video_end_time,
                "last_word_match": last_word_match
            }

        except Exception as e:
            log(f"[ERROR] during transcription: {str(e)}")
            return {
                "matched_words": [],
                "video_start_time": None,
                "video_end_time": None,
                "last_word_match": False
            }
        finally:
            log(f"Transcription complete | device: {self.device}")

            if self.device == "cuda":
                torch.cuda.empty_cache()

    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs 