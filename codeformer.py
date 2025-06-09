from codeformer import CodeFormer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from tqdm import tqdm

def run_codeformer(frames):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer( model_path=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\gfpgan\weights\RealESRGAN_x2plus.pth", model=model, scale=2)
    restored_frames = []
    codeformer = CodeFormer(
            weights=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\gfpgan\weights\codeformer.pth",
            fidelity_weight=0.7,
            upscale=1,
            has_aligned=True,
            only_center_face=True,
            bg_upsampler=bg_upsampler,
            face_enhance=True,
        )
    try:
        for frame in tqdm(frames, desc="CodeFormer Upscaling", unit="frame"):
                restored_img, _ = codeformer.face_enhance(frame, has_aligned=True)
                
                restored_frames.append(restored_img)
                print(f"appending upscaled frame")
    except Exception as e:
            print(f"Error during upscaling.. {str(e)}")
    
    return restored_frames