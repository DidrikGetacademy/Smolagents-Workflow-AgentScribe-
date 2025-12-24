# Copilot Instructions for Smolagents-Workflow-AgentScribe

## Repository Overview

**Purpose**: Automated motivational short video creation system using AI agents. Extracts audio from long-form content, transcribes it, identifies motivational segments using reasoning agents, creates professional short-form videos with effects/subtitles, and automatically uploads to YouTube/TikTok/Instagram.

**Size**: ~11MB, ~10,000 lines of Python code across 31 files  
**Languages**: Python 3.11+ (tested with 3.12.3)  
**Key Frameworks**: smolagents, PyTorch, MoviePy 2.x, faster-whisper, ONNX Runtime, OpenCV, Tkinter

## Critical Environment Requirements

### Required Dependencies (NOT in Standard Environment)
This project has **extensive GPU-dependent requirements** that will NOT be available in CI/standard environments:

1. **CUDA Toolkit**: PyTorch 2.6.0+cu126, torchvision, torchaudio with CUDA 12.6
2. **ONNX Runtime GPU**: Version 1.22.0 with CUDA provider
3. **External Model Directories** (NOT in repo):
   - `GPEN/` - Face enhancement models (referenced but not included)
   - `RIFE/` - FPS interpolation models (referenced but not included)  
   - `neon/` - Subtitle rendering (referenced but not included)
   - LUT files for color grading
   - YOLO face detection ONNX models
4. **Windows-Specific Paths**: Hardcoded `C:\Users\...` paths throughout (see App.py lines 65-81)
5. **External Video Files**: Work queue folder, video clips directory structure

### Environment Variables Required (in .env file)
```
OPENAI_APIKEY=<key>          # Required for LLM agents
YOUTUBE_API_KEY=<key>        # For trending video analysis
SERPAPI_API_KEY=<key>        # For search functionality
GPT_5_API_KEY=<key>          # Alternative LLM key
<Channel>_tiktok_instagram_api_key=<key>  # Per-channel social media keys
```

### Dependencies File Issues
- `requirements.txt` is **UTF-16 encoded** with CRLF line terminators (NOT standard UTF-8)
- Contains ~430 packages including many Windows/CUDA-specific wheels
- Two commented local wheel installs: flash-attn, Real-ESRGAN from git
- To read: `iconv -f UTF-16LE -t UTF-8 requirements.txt > requirements_utf8.txt`

## Project Architecture

### Entry Points
1. **`gui.py`** - Primary GUI application (Tkinter). Button "Run Engine" starts `video_creation_worker` thread.
2. **`App.py`** - Core video processing pipeline, worker threads, GPU operations.
3. **`test_motivational_agent.py`** - Standalone script to test transcript reasoning agent.

### Key Directories
```
Agent_AutoUpload/
  └── upload_Socialmedia.py      # YouTube/TikTok/Instagram upload with OAuth

Prompt_templates/                # YAML system prompts for different agents
  ├── verify_agent_system_prompt.yaml          # Validates motivational text
  ├── viral_agent_prompt.yaml                  # Analyzes trending content
  ├── system_prompt_background_music_analytic.yaml
  └── structured_output_prompt_TranscriptReasoning_gpt5.yaml

utility/
  ├── Custom_Agent_Tools.py      # @tool decorators for smolagents (speech-to-text, video creation)
  ├── Global_state.py            # Central state: queues, paths, counters
  ├── create_montage_short.py   # Multi-clip composition
  ├── clean_memory.py            # GPU memory management
  └── RIFE_FPS.py                # FPS interpolation wrapper

Utils-Video_creation/
  └── Fonts/                     # Font files for subtitles

Finetune/                        # Dataset & scripts for fine-tuning reasoning models
debug_performance/               # Log files (log.txt, VerifyAgentRun_data.txt)
```

### Core Workflow (App.py)
1. `video_creation_worker()` - Main loop consuming `Global_state.video_task_que`
2. `run_video_short_creation_thread()` - Spawns thread calling `create_short_video()`
3. `create_short_video()` - Full pipeline:
   - Extract video segment (start/end timestamps)
   - Detect faces with ONNX YOLOv8 (928x928 input, CUDA provider)
   - Crop & enhance faces with GPEN
   - Apply LUT color grading
   - Render dynamic subtitles (triplet grouping: 4 words per subtitle)
   - Mix background audio (selected by LLM agent)
   - Write H.264 video (libx264, CRF 8, yuv420p)
   - Run RIFE FPS interpolation (doubles FPS)
   - Upload to YouTube via `upload_Socialmedia.upload_video()`

### Agent System (smolagents)
- **Manager Agent** chunks transcript, feeds to **Reasoning Agent**
- **Reasoning Agent** identifies motivational quotes with timestamps using `verify_agent_system_prompt.yaml`
- Tools in `Custom_Agent_Tools.py`:
  - `SpeechToTextToolCUDA` - faster-whisper on GPU
  - `create_motivationalshort(text)` - Enqueues video task, parses `[Ss.s - Ee.e]` format
  - `Delete_rejected_line(text)` - Removes invalid segments
  - `Background_Audio_Decision_Model()` - LLM selects music & LUT
  - `Fetch_top_trending_youtube_videos()` - YouTube API metadata
- **Upload Agent** generates SEO metadata (title/description/tags) by analyzing trending videos

### Global State (`Global_state.py`)
- `video_task_que: PersistentVideoQueue` - Items: `(video_url, audio_path, start, end, subtitle_words)`
- `Montage_clip_task_Que: queue.Queue` - For multi-clip compositions
- `chunk_proccesed_event: threading.Event` - Signals transcript processing complete
- `gpu_lock: threading.Lock` - Serializes GPU operations
- Current paths: audio, video, YouTube channel, agent text file

### Face Detection (App.py lines 250-408)
- ONNX model: YOLOv8-face (928x928 input, fp16)
- Provider: `CUDAExecutionProvider` with device_id=0
- Preprocessing: letterbox resize, RGB->BGR, normalize [0,1], NCHW
- Postprocessing: `ultralytics.utils.ops.non_max_suppression(conf_thres=0.25, iou_thres=0.5)`
- Crops largest face (by area), centers & upscales to 9:16 for shorts

## Build & Run Instructions

### ⚠️ **CRITICAL: This Project CANNOT Run in Standard CI**
- **NO automated testing possible** - Requires Windows, CUDA GPU, external models, API keys
- **NO pip install** - Many dependencies are CUDA wheels that fail on CPU-only systems
- **NO linting setup** - No `.flake8`, `pyproject.toml`, or linting config files present
- **NO GitHub Actions** - No `.github/workflows/` directory exists

### Local Development Setup (Windows + NVIDIA GPU Only)
1. **Prerequisites**:
   - Windows 10/11
   - NVIDIA GPU with CUDA 12.6
   - Python 3.11 or 3.12
   - FFmpeg in PATH
   - Visual Studio Build Tools (for C++ extensions)

2. **Installation** (Expect 30-60 minutes):
   ```bash
   # Convert requirements.txt encoding first
   iconv -f UTF-16LE -t UTF-8 requirements.txt > requirements_utf8.txt
   
   # Create virtual environment
   python -m venv venv
   venv\Scripts\activate
   
   # Install dependencies (many will compile from source)
   pip install -r requirements_utf8.txt
   
   # Install smolagents with extras
   pip install smolagents[transformers,whisper]
   ```

3. **Manual Model Setup** (NOT automated):
   - Download GPEN weights to `GPEN/weights/`
   - Download RIFE models to `RIFE/train_log/`
   - Download YOLOv8-face ONNX to expected path
   - Create directory structure: `Video_clips/Youtube_Upload_folder/<channel>/`
   - Add LUT files to LUT directory

4. **Environment Configuration**:
   ```bash
   # Create .env file with all required API keys (see above)
   # Edit App.py lines 65-81 to update Windows paths to your system
   # Edit Global_state.py line 71 for queue backup path
   ```

5. **Smolagents Library Patches** (REQUIRED):
   - Per README: Modify `smolagents.model` classes for local model loading
   - Modify `smolagents.tools.SpeechToTextTool` for Whisper ONNX support
   - See `Smolagents_libary_Changes.txt` (referenced but not in repo)

6. **Run GUI**:
   ```bash
   python gui.py
   ```
   - Click "Add Video to list", select video file
   - Click "Run Engine" to start worker thread
   - Monitor `debug_performance/log.txt` for progress

7. **Run Standalone Agent Test**:
   ```bash
   python test_motivational_agent.py
   ```

### Common Issues & Workarounds
1. **Import errors (GPEN, RIFE, neon)**: These are external modules not in repo. Code expects them at `./GPEN/`, `./RIFE/`, `./neon/`. Either:
   - Clone/download these separately
   - Comment out imports if not using face enhancement/FPS boost
   
2. **Path errors**: All Windows absolute paths hardcoded. Search & replace:
   - `C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\` → your repo path
   
3. **GPU out of memory**: Adjust thresholds in `clean_get_gpu_memory()` calls throughout App.py

4. **MoviePy v2 API**: Code uses `with_effects`, `with_start`, `subclipped` (v2 style). Do NOT downgrade to v1.

5. **UTF-16 requirements.txt**: If pip fails, convert encoding first (see step 2 above)

## Code Conventions

### Style Guidelines
- **No linting enforced** - No config files present
- **Hardcoded paths everywhere** - Windows absolute paths in production code
- **Global state via module** - `Global_state.py` is imported and mutated globally
- **Threading without protection** - Many shared data structures accessed without locks (except `gpu_lock`)
- **Long functions** - `create_short_video()` is 600+ lines
- **Mixed logging** - `print()`, `log()` from `neon.log`, and `logging` module all used

### Critical Patterns to Follow
1. **GPU Memory Management**: ALWAYS call `clean_get_gpu_memory(threshold=X)` after GPU operations
   ```python
   # After model inference or video processing
   del model, tensors
   clean_get_gpu_memory(threshold=0.3)
   ```

2. **MoviePy v2 API** (Lines 22, 456, 476, 744-745):
   ```python
   clip = VideoFileClip(path).subclipped(start, end)
   clip = FadeIn(duration=0.1).apply(clip)
   text = TextClip(...).with_start(t).with_position(...)
   ```

3. **Timestamp Format** (Custom_Agent_Tools.py line 1287+):
   ```python
   # Text must be: [123.45s - 128.90s] MESSAGE HERE [128.90s - 134.20s] NEXT MESSAGE
   # Wrapped in: ===START_TEXT===\n...\n===END_TEXT===
   ```

4. **Tool Decorator** (Custom_Agent_Tools.py):
   ```python
   from smolagents import tool
   
   @tool
   def my_tool(param: str) -> dict:
       """Docstring is shown to LLM as tool description"""
       return {"result": value}
   ```

5. **Global State Access**:
   ```python
   import utility.Global_state as Global_state
   
   video_url = Global_state.get_current_videourl()
   Global_state.video_task_que.put((url, audio, start, end, words))
   ```

### File Paths Requiring Update for New Environments
- `App.py` lines 65-81 (log paths, model paths, saving paths)
- `Global_state.py` line 71 (queue backup path)
- `Custom_Agent_Tools.py` line 28 (cookie file path)
- All references to `C:\Users\didri\Desktop\...`

## Testing & Validation

### ⚠️ NO Automated Testing Available
- No test suite (test_*.py files are manual scripts, not pytest)
- No CI/CD configured
- No linting/formatting automation

### Manual Validation Steps
1. **Before committing changes to video processing**:
   - Run GUI, add test video, verify worker processes it
   - Check `debug_performance/log.txt` for errors
   - Verify output video exists in `Video_clips/Youtube_Upload_folder/`
   - Play output video to confirm quality

2. **Before committing changes to agent tools**:
   - Run `python test_motivational_agent.py`
   - Verify agent identifies motivational segments
   - Check text is saved with correct timestamp format

3. **GPU Memory Validation**:
   ```python
   import pynvml
   pynvml.nvmlInit()
   handle = pynvml.nvmlDeviceGetHandleByIndex(0)
   info = pynvml.nvmlDeviceGetMemoryInfo(handle)
   print(f"GPU memory: {info.used/1e9:.1f}GB / {info.total/1e9:.1f}GB")
   ```

## Key Files Reference

### Root Directory Files
```
.gitignore                    # Excludes venv, GPEN, RIFE, neon, Video_clips, Secrets, weights
App.py                        # Main pipeline (1,000+ lines)
gui.py                        # Tkinter GUI (177 lines)
requirements.txt              # UTF-16 encoded dependencies (430 packages)
test_motivational_agent.py    # Standalone agent test
apply_lut_to_image.py         # LUT color grading utility
blender.py                    # Video blending utility
copilot-instructions.md       # Existing brief instructions
Gjøremål.txt                  # Norwegian TODO notes
READMD.MD                     # Project README (note: typo in filename)
```

### Important .gitignore Entries
```
/GPEN, /RIFE, /neon          # External modules not in repo
/Video_clips, /Secrets        # API keys & output videos
*.pt, *.pickle                # Model weights, OAuth tokens
venv/, *env/                  # Virtual environments
```

## Working with This Codebase

### Before Making Changes
1. **Read README** (`READMD.MD`) for project context
2. **Check if external dependencies needed**: If modifying face detection/FPS/subtitles, you need GPEN/RIFE/neon
3. **Verify paths**: Most code assumes Windows with specific directory structure
4. **Check API keys**: Many features require `.env` with multiple keys

### When Adding New Features
1. **Custom Agent Tools**: Add `@tool` decorated functions to `Custom_Agent_Tools.py`
2. **Agent Prompts**: Create new YAML in `Prompt_templates/` following existing format
3. **Video Effects**: Modify `create_short_video()` in App.py (lines 417-870)
4. **GPU Operations**: Always add `clean_get_gpu_memory()` calls after allocation

### When Fixing Bugs
1. **Check logs first**: `debug_performance/log.txt` for runtime, `VerifyAgentRun_data.txt` for agent output
2. **Path issues**: Search for `C:\Users\didri\` and update to relative paths where possible
3. **GPU errors**: Increase memory cleaning thresholds in `clean_get_gpu_memory()` calls
4. **Agent failures**: Load corresponding YAML prompt, verify tool names match `Custom_Agent_Tools.py`

### Performance Considerations
- Video processing is **very slow**: 10-15 second clip can take 2-5 minutes with all effects
- RIFE interpolation: Adds 30-60 seconds per video
- Face enhancement: 0.5-1 second per frame
- YouTube upload: Agent metadata generation takes 30-60 seconds

## Trust These Instructions

These instructions were generated by thoroughly exploring the codebase:
- Analyzed all 31 Python files
- Reviewed 5 YAML prompt templates  
- Examined directory structure and dependencies
- Tested Python environment capabilities
- Verified file encodings and formats

**Only search beyond these instructions if**:
- You need specific implementation details of a function
- Instructions appear outdated (check git history)
- You're working on a component not covered here (e.g., Finetune directory)

---

**Last Updated**: Based on repository state as of commit with 31 Python files, ~10K lines of code.
