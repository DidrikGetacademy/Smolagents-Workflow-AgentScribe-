# Full-Agent-Flow Video Editing

## Overview

Full-Agent-Flow Video Editing is an end-to-end autonomous system that turns long-form motivational videos (podcasts, speeches, interviews) into fully produced YouTube Shorts and montage videos.

The pipeline is built around:
- **Multi-agent reasoning** (via `smolagents`) to read transcripts, detect high-impact quotes, and generate SEO metadata.
- **GPU-heavy video processing** (YOLOv8-face ONNX, GPEN, Blender, RIFE) to crop, enhance, and interpolate clips.
- **Background music selection and upload automation** for multiple YouTube channels (and optional TikTok/Instagram posting).

You can run it either headless (via `App.py`) or through a simple Tkinter GUI (`Gui.py`).

## High-Level Architecture

- **Entry points**
  - `App.py`: Main orchestration of transcription, reasoning, short creation, montage creation, and uploads. Contains worker threads, queues, and the full short pipeline (`create_short_video`).
  - `Gui.py`: Tkinter front-end to start the engine and monitor basic status logs.

- **Agents** (`Agents/`)
  - `Motivational_agent.py`: Reads transcript chunks and decides which passages become shorts.
  - `Montage_agent.py`: Builds cross-video motivational montages, then triggers montage rendering and upload.
  - `Youtube_agent.py`: Generates SEO metadata (title, description, tags, hashtags, category, schedule time) for uploads.
  - `Youtube_Download_agent.py`: (Optional) Automates downloading/queuing of new source videos.
  - `Vision_agent.py`: (Optional) For future visual reasoning or video selection workflows.

- **Core utilities** (`utility/`)
  - `Global_state.py`: Central runtime state (current paths, selected channel/model) and process-wide queues.
  - `Custom_Agent_Tools.py`: All tools that agents can call (transcription, YouTube search, work-queue tools, etc.).
  - `create_montage_short.py`: Rendering of montage parts and composition helpers.
  - `RIFE_FPS.py`: Wrapper to run the external RIFE frame interpolation script.
  - `upload_Socialmedia.py`: Upload logic for YouTube (and optional TikTok/Instagram via Ayrshare).
  - `clean_memory.py`, `log.py`, `reload_model.py`, `Persistent_video_Queue.py`: Supporting infrastructure.

- **Media modules**
  - `GPEN/`: Face enhancement models and utilities (used via `FaceEnhancement`).
  - `RIFE/`: Interpolation model and CLI script consumed by `RIFE_FPS.py`.
  - `blender/blender.py`: High-detail frame enhancement using Blender.
  - `neon/`: Neon subtitle styles and visual effects.
  - `Utils-Video_creation/`: LUTs, fonts, and style assets.

## Data Flow: End-to-End Short Creation

This section walks through how a single short is produced from a long-form source video.

### 1. Source video selection

- **Via main script**: At the bottom of `App.py` (inside `if __name__ == "__main__":`), a hard-coded `video_paths` list defines input files.
- **Via GUI**: In `Gui.AutoGensis.Add_Video_tolist`, the user selects `.mp4/.avi/.mkv/.mov` files, which are added to an internal list; the engine can be wired to use these paths.

### 2. Transcription (`transcribe_single_video` in `App.py`)

For each input video path:
- Audio is extracted using `ffmpeg` to a `.wav` file in a per-video `work_queue_folder/in_progress/<video_name>/` directory.
- A `SpeechToTextToolCUDA` instance (from `Custom_Agent_Tools`) is created with device `cuda` by default.
  - A GPU lock (`Global_state.gpu_lock`) ensures only one GPU-heavy transcription at a time.
- The tool writes a transcript `.txt` file and a backup copy.
- The tuple `(video_path, transcript_txt_path, agent_text_saving_path, audio_path)` is pushed to `Global_state.transcript_queue`.

### 3. Transcript reasoning (`gpu_worker` and `Motivational_agent`)

- `gpu_worker` runs as a dedicated thread:
  - It blocks on `Global_state.transcript_queue.get()`.
  - For each item, it updates `Global_state` with the current audio and video paths, and the agent-savings file path.
  - It then calls `Motivational_analytic_agent` from `Agents/Motivational_agent.py`.

- `Motivational_analytic_agent`:
  - Reloads an LLM via `Reload_and_change_model("gpt-5-minimal")`.
  - Loads the motivational system prompt from
    `Agents/Prompt_templates/Motivational Analytic Agent/System_prompt.yaml`.
  - Constructs a `CodeAgent` with tools:
    - `create_motivationalshort` (from `Custom_Agent_Tools`) to enqueue new short-creation tasks.
    - `FinalAnswerTool` for signaling a rejected chunk.
  - Uses `ChunkLimiterTool` to iterate through the transcript file in large character-limited chunks.
  - For each chunk, builds a very detailed task prompt describing what counts as a standalone, impactful, motivational passage and when to reject.
  - If no more text remains, it:
    - Sets `Global_state.chunk_proccesed_event` to signal that all tasks from this transcript are enqueued.
    - Waits for the video worker queue to drain (`wait_for_proccessed_video_complete`).

### 4. Enqueuing short-creation tasks (`create_motivationalshort` tool)

Inside `Custom_Agent_Tools` (not fully shown above, but central to the workflow):
- `create_motivationalshort` parses reasoning-agent output text of the form:
  - `[10.00s - 15.50s] YOU DON’T NEED TO FEEL READY ...`
- It converts timestamps to floats, pairs them with the original source video and audio, and enqueues tasks into `Global_state.video_task_que`.
- Each queue item roughly looks like:
  - `(video_url, audio_path, final_start_time, final_end_time, subtitle_words)`.

`PersistentVideoQueue` is used here, meaning tasks are backed to `Video_taskQueue_backup.json` so crashes or restarts can resume work.

### 5. Short rendering (`video_creation_worker` and `run_video_short_creation_thread`)

- `video_creation_worker` (in `App.py`) is a long-running thread that:
  - Waits on `Global_state.chunk_proccesed_event` (ensures tasks are ready).
  - `get()`s items from `video_task_que` with a timeout.
  - Calls `run_video_short_creation_thread(video_url, audio_path, start, end, subtitle_text)`.

- `run_video_short_creation_thread`:
  - Increments `Video_count` (global) so truncated audio filenames are unique.
  - Probes the full audio file duration using `ffmpeg` and validates the requested range.
  - Uses `truncate_audio` to cut just the needed segment with additional padding (default 4s before and after), both for better transcription alignment and background music selection.
  - Cleans brackets like `[10.1s - 17.3s]` out of the text and compresses whitespace.
  - Runs `SpeechToText_short_creation_thread` with the truncated audio and cleaned text to tightly align subtitle words with exact timestamps:
    - Returns `matched_words` (word-level timing), new refined `video_start_time`, and `video_end_time`.
  - Calls `create_short_video` with:
    - `video_path` (original long-form video file),
    - `audio_path` (truncated segment),
    - refined start/end times,
    - a generated or provided video title name,
    - `subtitle_text` as word/timing list.

### 6. Short building (`create_short_video` in `App.py`)

`create_short_video` is the heart of the rendering pipeline:

1. **Metadata & probe**
   - Uses `ffmpeg.probe` to inspect codecs, bitrate, and stream layout.

2. **Subtitle grouping & clip creation**
   - Groups word-level subtitles into small chunks (max 5 words, with pause detection) using `group_subtitle_words_in_triplets`.
   - Creates `TextClip` overlays per chunk (uppercase, stroke, centered, timed precisely with `.with_start()` and `duration`).

3. **Frame extraction**
   - Opens the full source `VideoFileClip`.
   - Subclips `[start_time, end_time]` into `clip`.
   - Iterates `clip.iter_frames()` and collects frames into a list.

4. **Face detection & cropping**
   - Calls `detect_and_crop_frames_batch`:
     - Runs YOLOv8-face ONNX (`yolov8x-face-lindevs_cuda.onnx`) via `onnxruntime` with CUDA provider.
     - Processes frames in batches; resizes to `928x928`, normalizes, and feeds to the detector.
     - Applies NMS (`ultralytics.utils.ops.non_max_suppression`).
     - Focuses on max-area detection, tracks center (`cx,cy`) with smoothing, and crops a 9:16 (1080x1920) region.

5. **Face enhancement**
   - (Optionally) passes frames through Blender for additional sharpening (`enhance_frames_bpy`).
   - Constructs a `FaceEnhancement` object from GPEN with model `GPEN-BFR-2048` and RealESRGAN upscaler.
   - Converts frames to BGR, calls `.process`, returns enhanced frames back to RGB.

6. **Clip reconstruction**
   - Builds `ImageSequenceClip(FaceEnhancement_frames, fps=clip.fps)` and forces `.with_duration(clip.duration)`.

7. **Logo overlay**
   - Reads `Video_clips/Youtube_Upload_folder/latest_uploaded.txt` to rotate through YouTube channels via `Global_state.set_current_yt_channel`.
   - Chooses an alpha logo clip per channel and subclips it to `clip.duration`.
   - Applies mask and centers on screen in `CompositeVideoClip`.

8. **Subtitle overlay**
   - Constructs `CompositeVideoClip` combining processed video, subtitle clips, and logo.
   - Applies a `CrossFadeIn` and final `FadeIn`/`FadeOut` effects.

9. **Background audio selection and mixing**
   - Calls `Background_Audio_Decision_Model` from `Custom_Agent_Tools` with:
     - `audio_file`, `video_path`, `already_uploaded_videos`, `start_time`, `end_time`.
   - Chooses a track from `Global_state.Music_list` based on mood/tempo/usage history.
   - If a background path is available, uses `mix_audio` to combine original clip audio + background, with volume and looping logic.

10. **Write-out + FPS interpolation**
    - Writes a temporary H.264 clip (`libx264`, CRF 8, `yuv420p`, `fps=30`, `minterpolate` filter, etc.).
    - Calls `run_rife` from `utility.RIFE_FPS` to generate a `<name>_rife.mp4` with interpolated frames.
    - Deletes the pre-RIFE file and frees GPU memory.

11. **YouTube upload**
    - Uses `Reload_and_change_model("gpt-5-minimal")` to (re)load an LLM model for metadata generation.
    - Retrieves `YT_channel` from `Global_state.get_current_yt_channel()`.
    - Calls `upload_video` from `utility.upload_Socialmedia` with:
      - The current model, `output_video` path, `subtitle_text`, `YT_channel`, `Background_Audio_Reason`, `song_name`, and final clip duration.

12. **Post-upload cleanup**
    - Deletes the RIFE input file (`out_path`), truncated audio file, and cleans GPU memory.

## Montage Workflow

Montage clips are created from multiple videos and assembled into a single motivational storyline.

### 1. Montage agent (`Run_short_montage_agent` in `Agents/Montage_agent.py`)

- Reloads an LLM via `Reload_and_change_model("gpt-5")`.
- Loads a montage-specific prompt template from
  `Agents/Prompt_templates/Montage short Agent/system_prompt.yaml`.
- Constructs a `CodeAgent` with tools:
  - `open_work_file` to read saved snippet files.
  - `montage_short_creation_tool` to enqueue montage tasks.
- Task logic:
  - Reads 3–8 video titles and associated snippets.
  - Groups them into:
    - Opening, one or more middle segments, and ending.
  - Emits 5 montage short creation requests (one per YouTube channel).

### 2. Montage job queue and worker (`Montage_short_worker` in `App.py`)

- Consumes items from `Global_state.Montage_clip_task_Que`:
  - Each item: `(video_path, audio_path, start_time, end_time, subtitle_text, order, montage_id, YT_channel, middle_order)`.
- For each part:
  - Truncates the audio (`truncate_audio`).
  - Re-aligns subtitles via `SpeechToText_short_creation_thread`.
  - Calls `_create_montage_short_func` from `utility/create_montage_short.py` with part data.
  - Waits for RIFE to generate a `<basename>_rife.mp4` and keeps that as the part path.
- Tracks readiness per group (`montage_id` prefix):
  - Requires:
    - A `start` part,
    - One or more `middle` parts with contiguous `middle_order` indices,
    - An `ending` part.
- Once ready, calls `compose_montage_clips` (in `create_montage_short.py`) to stitch `start + middles + ending` into a single output in `Video_clips/Montage_clips/montage_<group>.mp4`.
- Calls `upload_MontageClip` from `utility.upload_Socialmedia` using a freshly reloaded model.

### 3. `_create_montage_short_func` details

- Similar to `create_short_video` but specialized for montage:
  - Tighter subtitle chunking and larger font size.
  - Color/saturation adjustment via `change_saturation` from `App.py`.
  - Blender + GPEN face enhancement pipeline.
  - Channel-specific logos (e.g., different logos for `LR_Youtube` start vs. non-start parts).
  - `FadeIn`/`FadeOut` and static `.fps = clip.fps` with a `fps=30` ffmpeg filter.

## Upload and Social Media Automation

All YouTube upload logic lives in `utility/upload_Socialmedia.py`.

- **Authentication**
  - `get_authenticated_service(YT_channel)` loads OAuth credentials per channel from `Video_clips/Secrets/<Channel>/client_secret_*.json`.
  - Uses `youtube_token.pickle` per channel for refresh tokens.

- **Metadata generation**
  - `get_automatic_data_from_agent(model, input_video)` (for regular shorts) and
    `get_automatic_data_from_agent_montage(model, input_video, YT_channel)` (for montages):
    - Both create a `CodeAgent` with tools:
      - `ExtractAudioFromVideo`, `SpeechToTextTool_viral_agent` for transcript.
      - `Fetch_top_trending_youtube_videos` to inspect competitor metadata.
      - Web-search tools (`GoogleSearchTool`, `DuckDuckGoSearchTool`, `VisitWebpageTool`) and `PythonInterpreterTool`.
      - Optional `read_file` for reading already uploaded logs.
    - They compute `previous_publishAt` via `Get_latestUpload_date(YT_channel)`.
    - The agent returns a JSON object with `title`, `description`, `hashtags`, `tags`, `categoryId`, `publishAt`.

- **Upload**
  - `upload_video`:
    - Builds the request body from agent output.
    - Uploads as `private` with scheduled `publishAt`.
    - Adds the video to the channel’s uploads playlist (via `get_single_playlist_id`).
    - Logs detailed metadata and subtitles to `already_uploaded.txt` for that channel.
    - Deletes the truncated audio file used for the short (`Global_state.get_current_truncated_audio_path()`).
  - `upload_MontageClip` performs the same process for montage outputs.

- **Optional TikTok/Instagram**
  - `upload_tiktok_Instagram_API` uses Ayrshare to cross-post the same rendered short.

## Global State and Queues

`utility/Global_state.py` is the glue for all runtime state:

- **Current values**
  - `_current_audio_url`, `_current_video_url`, `_current_youtube_channel`, `_current_agent_saving_file`, `_current_truncated_audio_path`, `_current_global_model`, `count`.
  - Accessed via `get_current_*` and `set_current_*` helpers.

- **Queues & locks**
  - `video_task_que = PersistentVideoQueue(...)`: durable queue for short-creation tasks.
  - `transcript_queue = queue.Queue()`: in-memory queue for `(video_path, transcript_path, agent_path, audio_path)`.
  - `Montage_clip_task_Que = queue.Queue()`: jobs for montage parts.
  - `gpu_lock = threading.Lock()`: ensures only one GPU-heavy transcription runs at a time.
  - `chunk_proccesed_event = threading.Event()`: used to gate `video_creation_worker` so it only runs when transcripts are fully processed.

- **Background music catalog**
  - `Music_list`: a curated list of tracks with path, song name, mood tags, valence/arousal, and example quote suggestions.
  - `Background_Audio_Decision_Model` (in `Custom_Agent_Tools`) reasons over this structure to pick a matching track and avoid duplicates.

## Agent Tools (`utility/Custom_Agent_Tools.py`)

Main categories of tools provided to agents:

- **File and transcript tools**
  - `read_file(text_file)`: reads first 3000 chars of a text file.
  - `ChunkLimiterTool`: stateful chunk reader by character count (used by `Motivational_agent`).
  - `Chunk_line_LimiterTool`: stateful chunk reader based on quote blocks (between `===START_QUOTE===` and `===END_QUOTE===`).
  - `Read_transcript`: stateless chunk reader by file offset.

- **Audio/video helpers**
  - `ExtractAudioFromVideo(video_path)`: creates a `temp_audio.wav` at mono 16kHz.
  - `transcribe_audio_to_txt(video_paths)`: offline batch transcriber using the generic `SpeechToTextTool` wrapper.

- **Transcription models**
  - `SpeechToTextTool` (PipelineTool): wraps `faster-whisper` for generic transcription.
  - `SpeechToTextToolCUDA` and `SpeechToText_short_creation_thread`: specialized for GPU and short alignment (defined further down in the same file).
  - `SpeechToTextTool_viral_agent`: used by the YouTube SEO agent for full-video transcription.

- **YouTube discovery**
  - `Fetch_top_trending_youtube_videos(Search_Query)`: uses the YouTube Data API v3 to fetch trending videos, then enriches with categories, statistics, and channel subscriber counts.

- **Montage and work tools**
  - `open_work_file`, `montage_short_creation_tool`: allow agents to read saved snippet files and push multi-part montage jobs into `Global_state.Montage_clip_task_Que`.

## GUI (`Gui.py`)

The GUI is intentionally simple and can be extended:

- Class `AutoGensis`:
  - Shows:
    - A list of added video files.
    - A list of videos in progress.
    - Basic GPU/CPU info.
    - An agent log window (`ScrolledText`) for status messages.
  - Buttons:
    - **Add Video to list**: lets the user select videos from disk.
    - **Run Engine**: clears logs, frees GPU memory, and starts `video_creation_worker` in a background thread.
    - **Stop Engine**: currently a stub; can be extended to set sentinel values on queues.

## Running the Project

### 1. Environment and dependencies

- OS: Windows (absolute Windows paths are hard-coded throughout the repo).
- GPU: NVIDIA GPU with CUDA, sufficient VRAM for ONNX face detection + GPEN + RIFE.
- Python environment:
  - Install base requirements:
    - `pip install -r requirements.txt`
  - Additional model/tool requirements:
    - `smolagents`
    - `faster-whisper`
    - `onnxruntime-gpu`
    - `torch`, `torchvision`
    - `moviepy>=2`
    - `pydub`, `ffmpeg` (system binary), `google-api-python-client`, `google-auth-oauthlib`

Make sure:
- Your local model folders (`LLM-models`, `Face-Detection-Models`, Whisper checkpoints, GPEN, RIFE) match the hard-coded paths in `App.py`, `Custom_Agent_Tools.py`, `RIFE_FPS.py`.
- Secret files for each YouTube channel exist under `Video_clips/Secrets/<Channel>/`.
- `YOUTUBE_API_KEY`, `SERPAPI_API_KEY`, `OPENAI_APIKEY`, and Ayrshare keys (for TikTok/Instagram) are set in your `.env`.

### 2. Running via CLI

```bash
python App.py
```

This will:
- Clean log files.
- Start `video_creation_worker` and `gpu_worker` threads.
- Transcribe all `video_paths` listed at the bottom of `App.py`.
- Run the motivational reasoning agent, enqueue shorts, render, RIFE-interpolate, and upload.
- Launch the montage agent afterward and process any queued montage jobs.

### 3. Running via GUI

```bash
python Gui.py
```

- Use **Add Video to list** to point at long-form files.
- Press **Run Engine** to start `video_creation_worker`.
- Watch the log window for status updates.

> Note: in the current code, `Gui.py` does not yet wire selected GUI video paths into the transcription/agent pipeline. That connection can be implemented by passing `Uploaded_videos_tolist` into a wrapper that calls `transcribe_single_video` for each path.

## Extending the System

A few common extension points:

- **Add a new agent tool**
  - Implement it in `utility/Custom_Agent_Tools.py` with `@tool` or as a `Tool` subclass.
  - Document function inputs/outputs in the docstring.
  - Register it in the relevant agent file (`Motivational_agent.py`, `Montage_agent.py`, or `Youtube_agent.py`).

- **Change background-music policy**
  - Update `Global_state.Music_list` and `Background_Audio_Decision_Model` logic to adjust how tracks are matched to content.

- **Insert new visual processing steps**
  - Modify `create_short_video` or `_create_montage_short_func`:
    - Add a transformation step between cropping and face enhancement.
    - Respect RGB/BGR conversions and free intermediate tensors (`del` + `clean_get_gpu_memory`).

- **Support new social platforms**
  - Follow the pattern in `upload_tiktok_Instagram_API` to call other REST APIs, reusing the SEO metadata from the YouTube agent.

## Notes and Caveats

- The project relies heavily on absolute Windows paths; if you move the project folder or models, update the paths in the relevant files.
- GPU memory management is critical; new features should:
  - Avoid keeping large arrays/tensors alive longer than necessary.
  - Use `clean_get_gpu_memory` and `torch.cuda.empty_cache()` where appropriate.
- Many components assume MoviePy v2-style APIs (`with_effects`, `subclipped` etc.). Mixing versions may break the pipeline.

This README captures the overall structure and data flow of the AI agent system so you (and AI assistants like GitHub Copilot) can safely extend, debug, and operate the full-agent video editing pipeline.
