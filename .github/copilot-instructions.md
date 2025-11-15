# Copilot Instructions for Full-Agent-Flow_VideoEditing

Purpose
- Help AI agents work productively in this repo: automated motivational shorts creation, enhancement, and YouTube upload.

Architecture overview
- Entry points
  - `gui.py`: Tkinter UI that starts the engine (spawns `video_creation_worker`).
  - `App.py`: Core pipeline and worker threads. Builds short clips from a source video segment + transcript words. Handles GPU-heavy steps, LUTs, audio mix, FPS interpolation, and YouTube upload orchestration.
  - `Agent_AutoUpload/Upload_youtube.py`: Generates SEO metadata via `smolagents` and uploads to YouTube (OAuth per channel).
- State & queues
  - `Global_state.py`: Central runtime state (current video/audio paths, yt channel, selected LLM), and work queues:
    - `video_task_que`: items `(video_url, audio_path, start, end, subtitle_words)` processed by `video_creation_worker`.
    - `Montage_clip_task_Que`: items for montage worker.
    - `chunk_proccesed_event`: gates video worker after transcript processing.
- Agent tools
  - `Custom_Agent_Tools.py`: Tooling used by `smolagents`:
    - Transcript tools: `SpeechToTextTool`, `ChunkLimiterTool`, `Chunk_line_LimiterTool`, `Read_transcript`.
    - Video creation enqueue: `create_motivationalshort`, `Delete_rejected_line`, `open_work_file`, `montage_short_creation_tool`.
    - YouTube/data: `Fetch_top_trending_youtube_videos`, `ExtractAudioFromVideo`, `SpeechToTextTool_viral_agent`.
- Media enhancement
  - Face detection (ONNX) and crop in `App.py` (`detect_and_crop_frames_batch` → YOLOv8-face ONNX with CUDA provider).
  - Subtitles rendering via `neon/dynamic_glow_fade.py`.
  - Face restoration via `GPEN/face_enhancement.py`.
  - FPS interpolation via `RIFE_FPS.run_rife` (wraps `RIFE/inference_video.py`).

Critical workflows
- Run GUI
  - From `gui.py`, button “Run Engine” calls `video_creation_worker` in a thread. Expect CUDA; uses paths from `Global_state` and the queues.
- End-to-end short creation
  1) An agent validates/saves text blocks to an agent file.
  2) `verify_saved_text_agent()` (in `App.py`) loads prompt `Prompt_templates/verify_agent_system_prompt.yaml`, drives a `CodeAgent` with tools `create_motivationalshort`/`Delete_rejected_line`.
  3) `create_motivationalshort(text=...)` enqueues work to `Global_state.video_task_que` with timestamps parsed from `[Ss.s - Ee.e]` lines.
  4) `video_creation_worker()` consumes items, calls `run_video_short_creation_thread()` → `create_short_video()`.
  5) `create_short_video()` builds `ImageSequenceClip`, applies LUT, face enhancement, neon subtitles, mixes background music, writes H.264, then runs RIFE and uploads via `Upload_youtube.upload_video`.
- YouTube upload
  - Uses channel-specific OAuth in `Secrets/<Channel>/client_secret_*.json` via `get_authenticated_service()`. Video is set private, added to uploads playlist, and metadata comes from a `CodeAgent` in `Upload_youtube.get_automatic_data_from_agent()`.

Project conventions & gotchas
- Paths are Windows-absolute in many places. Avoid changing to relative unless you wire all call sites.
- Timestamped text format is strict. Tools parse blocks like:
  ===START_TEXT===\n[123.45s - 128.90s] line ... [..] line ...\n===END_TEXT===
  - `create_motivationalshort` must receive text WITHOUT the START/END markers; `Delete_rejected_line` expects them.
- Subtitle words format in `create_short_video` is list of dicts: `{word,start,end}`; pass through untouched.
- GPU memory hygiene matters. Use `clean_memory.clean_get_gpu_memory(threshold=...)` and free large tensors/arrays.
- ONNX face detector path/provider fixed in code (CUDA). Keep shapes 928×928 preproc and call `ultralytics.utils.ops.non_max_suppression` with existing thresholds.
- MoviePy v2 style is used (e.g., `with_effects`, `subclipped`, `ImageSequenceClip(...).with_duration(...)`). Stick to those APIs.
- RIFE integration: do not import RIFE as a lib; call `RIFE_FPS.run_rife(path)`; it builds output `<name>_rife.mp4`.
- Logs go to `debug_performance/log.txt` via `log.log()`; keep new logs short.

External dependencies
- Heavy models on local disk: Whisper, YOLOv8-face (ONNX), GPEN weights, RIFE train_log models, LUTs, fonts. Don’t hardcode new model paths without confirming environment.
- `smolagents` tools drive LLMs via `LiteLLMModel`. `GPT_5_API_KEY` is read from env. Respect max tokens and avoid loading multiple big models at once.

Common extension patterns
- Adding a new tool: implement in `Custom_Agent_Tools.py` with `@tool`, define inputs/output_type, keep side-effects minimal, and log succinctly.
- New processing step in the video pipeline: add after cropping/before writing in `App.create_short_video()`. Ensure RGB/BGR conversions are correct and free intermediates.
- New background-music policy: extend `Background_Audio_Decision_Model` in `Custom_Agent_Tools` to return `{path, reason, lut_path}`; `create_short_video` already consumes these.

Examples
- Enqueue a short (from an agent tool):
  create_motivationalshort(text="[10.0s - 15.0s] YOU DON’T NEED TO FEEL READY [15.0s - 20.0s] JUST START")
- Worker loop pattern (see `video_creation_worker`): get item with timeout, process, mark `task_done()`.

Safety/limits
- Don’t modify OAuth/Secrets paths in code.
- Don’t change fps/interpolation flags unless you also update `RIFE_FPS.run_rife` invocation.
- Keep `Global_state` as the single source of truth for current paths/counters.

Where to look in code
- Core pipeline: `App.py` (create_short_video, detect_and_crop_frames_batch, workers)
- Tools/agents: `Custom_Agent_Tools.py`, `Prompt_templates/*.yaml`
- Upload: `Agent_AutoUpload/Upload_youtube.py`
- Effects: `neon/dynamic_glow_fade`, `GPEN/*`, `RIFE_FPS.py`
