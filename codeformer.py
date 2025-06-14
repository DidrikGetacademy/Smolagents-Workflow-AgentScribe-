import cv2
import os

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from tqdm import tqdm

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        # Convert BGR to RGB as expected by CodeFormer
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        success, frame = cap.read()
    cap.release()
    return frames

def save_frames_to_video(frames, original_video_path, output_path="output_video.mp4"):
    cap = cv2.VideoCapture(original_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Assuming frames are RGB, convert back to BGR for saving with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()

def run_codeformer_on_video(video_path):
    print("Extracting frames from video...")
    frames = extract_frames(video_path)
    from codeformer import CodeFormer
    print("Initializing models...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(
        model_path=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\gfpgan\weights\RealESRGAN_x2plus.pth",
        model=model,
        scale=2
    )

    codeformer = CodeFormer(
        weights=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\gfpgan\weights\codeformer.pth",
        fidelity_weight=0.7,
        upscale=1,
        has_aligned=False,
        only_center_face=True,
        bg_upsampler=bg_upsampler,
        face_enhance=True,
    )

    restored_frames = []
    print("Running CodeFormer on frames...")
    try:
        for frame in tqdm(frames, desc="CodeFormer Upscaling", unit="frame"):
            restored_img, _ = codeformer.face_enhance(frame, has_aligned=True)
            restored_frames.append(restored_img)
    except Exception as e:
        print(f"Error during upscaling: {str(e)}")

    output_path = "restored_video.mp4"
    save_frames_to_video(restored_frames, video_path, output_path)
    print(f"Restored video saved to: {output_path}")

# Example usage
run_codeformer_on_video(r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\airbrush_video.mov")
