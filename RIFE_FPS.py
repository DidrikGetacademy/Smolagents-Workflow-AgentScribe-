# import os
# import subprocess
# import torch
# import gc

# def log(message):
#     print(message)

# def get_video_resolution(video_path):
#     """Get (width, height) of video using ffprobe"""
#     cmd = [
#         "ffprobe", "-v", "error",
#         "-select_streams", "v:0",
#         "-show_entries", "stream=width,height",
#         "-of", "csv=p=0:s=x",
#         video_path
#     ]
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"ffprobe failed: {result.stderr}")
#     width, height = map(int, result.stdout.strip().split('x'))
#     return width, height

# def get_video_fps(video_path):
#     """Get FPS of video using ffprobe"""
#     cmd = [
#         "ffprobe", "-v", "error",
#         "-select_streams", "v:0",
#         "-show_entries", "stream=r_frame_rate",
#         "-of", "default=noprint_wrappers=1:nokey=1",
#         video_path
#     ]
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"ffprobe failed: {result.stderr}")
#     fps_str = result.stdout.strip()
#     nums = fps_str.split('/')
#     if len(nums) == 2:
#         fps = float(nums[0]) / float(nums[1])
#     else:
#         fps = float(fps_str)
#     return fps

# def print_video_info(video_path, label="Video"):
#     width, height = get_video_resolution(video_path)
#     fps = get_video_fps(video_path)
#     log(f"{label} resolution: {width}x{height}")
#     log(f"{label} FPS: {fps:.2f}")
#     return width, height, fps

# def run_rife_interpolation(
#     rife_python_exe,
#     inference_script,
#     input_video,
#     output_video,
#     model_dir,
#     exp=1
# ):
#     try:
#         # Print input video info
#         print_video_info(input_video, label="Input video")

#         log(f"Checking input video dimensions for resizing...")

#         process = subprocess.Popen(
#             [
#                 rife_python_exe,
#                 inference_script,
#                 "--video", input_video,
#                 "--output", output_video,
#                 "--model", model_dir,
#                 "--exp", str(exp),
#                 "--ext", "mp4",
#                 "--scale", "1",
#             # "--UHD"
#             ],
#         )
#         process.wait()
      
#         if process.returncode != 0:
#             raise RuntimeError(f"RIFE failed with exit code {process.returncode}")

#         log(f"RIFE interpolation done: {output_video}")

#         # Print output video info
#         print_video_info(output_video, label="Output video")

#     except Exception as e:
#         log(f"RIFE interpolation failed: {e}")

#     finally:
#         torch.cuda.empty_cache()
#         gc.collect()
#         log("Cleared cache and collected garbage")

# if __name__ == "__main__":
#     # === ABSOLUTE PATH CONFIG ===
#     rife_python_exe = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\RIFE\RIFEVENV\Scripts\python.exe"
#     inference_script = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\RIFE\inference_video.py"
#     input_video = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\short11.mp4"
#     output_video = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\short11_rife.mp4"
#     model_dir = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\RIFE\train_log"

#     run_rife_interpolation(
#         rife_python_exe,
#         inference_script,
#         input_video,
#         output_video,
#         model_dir
#    )
import os
import subprocess
import torch
import gc

def log(message):
    print(message)

def get_video_resolution(video_path):
    """Get (width, height) of video using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    width, height = map(int, result.stdout.strip().split('x'))
    return width, height

def get_video_fps(video_path):
    """Get FPS of video using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    fps_str = result.stdout.strip()
    nums = fps_str.split('/')
    if len(nums) == 2:
        fps = float(nums[0]) / float(nums[1])
    else:
        fps = float(fps_str)
    return fps

def print_video_info(video_path, label="Video"):
    width, height = get_video_resolution(video_path)
    fps = get_video_fps(video_path)
    log(f"{label} resolution: {width}x{height}")
    log(f"{label} FPS: {fps:.2f}")
    return width, height, fps

def run_rife(input_video):
    """
    Run RIFE interpolation on the given input video.
    Only 'input_video' is needed. Other config is set here.
    """

    # === CONFIGURATION ===
    rife_python_exe = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\RIFE\RIFEVENV\Scripts\python.exe"
    inference_script = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\RIFE\inference_video.py"
    model_dir = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\RIFE\train_log"

    # Automatically generate output video path
    base, ext = os.path.splitext(input_video)
    output_video = f"{base}_rife{ext}"

    exp = 1  

    try:
        # Print input video info
        print_video_info(input_video, label="Input video")

        log(f"Running RIFE interpolation...")

        process = subprocess.Popen(
            [
                rife_python_exe,
                inference_script,
                "--video", input_video,
                "--output", output_video,
                "--model", model_dir,
                "--exp", str(exp),
                "--ext", "mp4",
                "--scale", "1",
            ],
        )
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"RIFE failed with exit code {process.returncode}")

        log(f"RIFE interpolation done: {output_video}")

        # Print output video info
        print_video_info(output_video, label="Output video")

        return output_video

    except Exception as e:
        log(f"RIFE interpolation failed: {e}")

    finally:
        torch.cuda.empty_cache()
        gc.collect()
        log("Cleared cache and collected garbage")
