   
       
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
        for frame in tqdm(cropped_frames, desc="CodeFormer Upscaling", unit="frame"):
            restored_img, _ = codeformer.face_enhance(cropped_frames, has_aligned=True)
            
            restored_frames.append(restored_img)
            print(f"appending upscaled frame")
    except Exception as e:
        print(f"Error during upscaling.. {str(e)}")