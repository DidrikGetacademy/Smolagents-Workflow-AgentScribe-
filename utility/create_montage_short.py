from tqdm import tqdm
import torch
from moviepy import VideoFileClip, ImageSequenceClip, TextClip, CompositeVideoClip,vfx,AudioFileClip,afx,CompositeAudioClip,afx
import os
import cv2
import sys
import sys
import os


def _ffmpeg_minterpolate_to_fps(input_path: str, target_fps: int = 30) -> str:
    """Create a target-fps version of input using ffmpeg minterpolate. Returns new path."""
    import os
    import subprocess
    from neon.log import log

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_fps{int(target_fps)}{ext}"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "8",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "384k",
        "-ar", "48000",
        "-movflags", "+faststart",
        output_path,
    ]
    log(f"[minterpolate] {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log(f"[minterpolate] Failed to normalize FPS. Using original.\n{result.stderr[-800:]} ")
        return input_path
    log(f"[minterpolate] Wrote normalized clip: {output_path}")
    return output_path


def _create_montage_short_func(video_path, start_time, end_time, subtitle_text, video_name, order, Video_output_path=None, YT_channel=None, middle_order=None):
    print(f"YT_channel in create_montage_short: {YT_channel}, order: {order}, middle_order: {middle_order}")
    from neon.log import log; log(f"[create_montage_short] order={order}, middle_order={middle_order}")
    print(f"YT_channel in create_montage_short: {YT_channel}")
    import os
    import gc
    from utility.clean_memory import clean_get_gpu_memory
    import ffmpeg
    from neon.log import log
    from moviepy import VideoFileClip,ImageSequenceClip,CompositeVideoClip
    from moviepy.video.fx.CrossFadeIn import CrossFadeIn
    from moviepy.video.fx.FadeIn import FadeIn
    from moviepy.video.fx.FadeOut import FadeOut
    from App import detect_and_crop_frames_batch
    log(f"YT_channel: {YT_channel}")
    probe = ffmpeg.probe(video_path)
    log(probe)
    format_info = probe.get('format', {})
    bitrate = int(format_info.get('bit_rate', 0))
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    video_codec = video_streams[0]['codec_name'] if video_streams else None
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    audio_codec = audio_streams[0]['codec_name'] if audio_streams else None




    def group_subtitle_words_in_triplets(subtitle_words):
         chunks = []
         if not subtitle_words:
             return chunks

         offset = float(subtitle_words[0]['start'])
         segments = []
         current_segment = [subtitle_words[0]]
         PAUSE_THRESHOLD = 0.3

         for i in range(1, len(subtitle_words)):
              prev_word = subtitle_words[i-1]
              curr_word = subtitle_words[i]
              gap = float(curr_word['start']) - float(prev_word['end'])

              if gap > PAUSE_THRESHOLD:
                  current_segment[-1]['end'] += 0.2
                  segments.append(current_segment)
                  current_segment = []

              current_segment.append(curr_word)

         if current_segment:
              segments.append(current_segment)

         MAX_WORDS_PER_CHUNK = 4

         for segment in segments:
              for i in range(0, len(segment), MAX_WORDS_PER_CHUNK):
                   chunk_words = segment[i : i + MAX_WORDS_PER_CHUNK]

                   text_chunk = ''.join([w['word'].strip() + ' ' for w in chunk_words]).strip().upper()
                   start = float(chunk_words[0]['start']) - offset
                   end = float(chunk_words[-1]['end']) - offset
                   duration = max(0.0, end - start)
                   start = max(0.0, start)

                   chunks.append({'text': text_chunk, 'start': start, 'end': end, 'duration': duration})

         return chunks

    try:
       log(f"Subtitle text before grouping: {subtitle_text}")
       triplets = group_subtitle_words_in_triplets(subtitle_text)
    except Exception as e:
         log(f"[group_subtitle_words_in_triplets] Error during grouping of subtitles in triplets. {str(e)} ")


    def create_subtitles_from_triplets(triplets):
        text_clips = []

        for i, c in enumerate(triplets):



            txt_clip = TextClip(
                text=c['text'],
                font=r"C:\WINDOWS\FONTS\COPPERPLATECC-BOLD.TTF",
                font_size=50,
                margin=(10, 10),
                text_align="center",
                vertical_align="center",
                horizontal_align="center",
                color='white',
                stroke_color="black",
                stroke_width=4,
                size=(1400, 300),
                method="label",
                duration=c['duration']
            ).with_start(c['start']).with_position(('center', 0.50), relative=True)


            text_clips.append(txt_clip)
            log(f"Subtitle clip {i}: text='{c['text'][:20]}...', total_duration={c['duration']:.3f}s")
        return text_clips
    #------------------------------------------------------------------------#
    # CREATES THE (START & END) OF the currentclip from FULL ORIGINAL VIDEO
    #------------------------------------------------------------------------#
    full_video = VideoFileClip(video_path)
    clip = full_video.subclipped(start_time, end_time)
    log(f"clip duration: {clip.duration}, clip fps: {clip.fps}, clip width: {clip.w}, clip height: {clip.h}, start_time: {start_time}, end_time: {end_time}, video_path: {video_path}")
    log(f"Clip fps: {clip.fps}")




#--------------------------------------------------#
# Extrcting frames from original video to a LIST
#--------------------------------------------------#
    frames = []
    for  frame in clip.iter_frames():
        frames.append(frame)
        frame_height, frame_width = frame.shape[:2]
    log(f"[Extracting original video frames]  1. frames: {len(frames)} frames.\n [CLIP.ITER] Height: {frame_height}, Width: {frame_width}")









# --------------------------------#
# Yolo8/facedetection + Cropping
# --------------------------------#
    clean_get_gpu_memory(threshold=0.2)
    cropped_frames = detect_and_crop_frames_batch(frames=frames,batch_size=8)
    log(f"2. cropped frames length: {len(cropped_frames)}")
    del frames



    try:
        subtitle_clips = create_subtitles_from_triplets(triplets)
    except Exception as e:
            log(f"error during [create_subtitles_from_triplets]: {str(e)}")

# -----------------#
# # color/Adjustment
# #-----------------#
    log(f"\n\n[COLOR ADJUSTMENT] PROCCESS starting...")
    try:
        from App import change_saturation

        Color_corrected_frames = [change_saturation(frame,mode="Increase", amount=0.1) for frame in cropped_frames]
        log(f"3. Color_corrected_frames length: {len(Color_corrected_frames)}")
        del cropped_frames
    except Exception as e:
         log(f"Error during color correction: {str(e)}")



    from blender.blender import enhance_frames_bpy
    blender_frames = enhance_frames_bpy(Color_corrected_frames)
    del Color_corrected_frames
# ----------------------#
#   FACEENCHANCEMENT
# ----------------------#
    class Face_enchance_Args:
            model = 'GPEN-BFR-2048'
            task = 'FaceEnhancement'
            key = None
            in_size = 2048
            out_size = 0
            channel_multiplier = 2
            narrow = 1
            alpha = 0.4
            use_sr = True
            use_cuda = True
            save_face = False
            aligned = False
            sr_model = 'realesrnet'
            sr_scale = 2
            tile_size = 0
            ext = '.png'

    log(f"\n\n[FACEENCHACEMENT] PROCCESS starting...")

    from GPEN.face_enhancement import FaceEnhancement

    face_args = Face_enchance_Args()

    Skin_texture_enchancement = FaceEnhancement(
        face_args,
        in_size=face_args.in_size,
        model=face_args.model,
        use_sr=face_args.use_sr,
        device='cuda' if face_args.use_cuda else 'cpu'
    )

    FaceEnhancement_frames = []
    try:
         for enchanced_frame in tqdm(blender_frames, desc="[FaceEnhancement]  proccessing frames", unit="frame"):
              log(f"Input frame size: {frame.shape[1]} x {frame.shape[0]}")
              frame_height,frame_width = frame.shape[:2]
              enchanced_frame = cv2.cvtColor(enchanced_frame, cv2.COLOR_RGB2BGR)
              enchanced_frame, _, _ = Skin_texture_enchancement.process(enchanced_frame)
              log(f"Enhanced frame size: {enchanced_frame.shape[1]} x {enchanced_frame.shape[0]}")
              RGB_face_enchanced_frame = cv2.cvtColor(enchanced_frame, cv2.COLOR_BGR2RGB)
              FaceEnhancement_frames.append(RGB_face_enchanced_frame)

         log(f"[FaceEnhancement] 5. FaceEnhancement_frames length: {len(FaceEnhancement_frames)}")
         torch.cuda.empty_cache()
         gc.collect()

         log(f"Cleared cache and collected garbage")
    except Exception as e:
            log(f"[FaceEnhancement] Error: {str(e)}")




#-------------------------------#
# Creating videoclip from frames
#-------------------------------#
    try:
       processed_clip = ImageSequenceClip(FaceEnhancement_frames, fps=clip.fps).with_duration(clip.duration)
       gc.collect()
    except Exception as e:
         log(f"[processed_clip] ERROR: {str(e)}")
    del blender_frames



#-----------------------------------------------#
#     Adds Logo/overlay for YT_channel
#-----------------------------------------------#
    if YT_channel == "LR_Youtube":
            if order != "start":
                overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_MR_CUT.mp4",has_mask=True)
            else:
                overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LR.mp4",has_mask=True)

    elif YT_channel == "LRS_Youtube":
                overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LRS.mp4",has_mask=True)

    elif YT_channel == "MR_Youtube":
        overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_MR.mp4",has_mask=True)

    elif YT_channel == "LM_Youtube":
            overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LM.mp4",has_mask=True)
    else:
            raise ValueError(f"Error No {YT_channel} exists.")


    overlay_clip = overlay_clip.subclipped(0, clip.duration)
    logo_mask = overlay_clip.to_mask()
    logo_with_mask = overlay_clip.with_mask(logo_mask)

#-----------------------------------------------#
# Adds Subtitles + Logo to the video
#-----------------------------------------------#
    final_clip = CompositeVideoClip(
                [processed_clip.with_position('center')] + subtitle_clips + [logo_with_mask.with_position('center',0.85)] ,
                size=processed_clip.size
                )



    del logo_with_mask
    gc.collect()


    fade = CrossFadeIn(1.5)
    final_clip = fade.apply(final_clip)


    final_clip.audio = clip.audio






#-----------------------------------------------------#
#    Adds fade in/out to the video and sets the FPS
#------------------------------------------------------#
    final_clip = FadeIn(duration=0.1).apply(final_clip)
    final_clip = FadeOut(duration=0.05).apply(final_clip)
    final_clip.fps = clip.fps




#-----------------------------------------------------#
#    Writes the final Video
#------------------------------------------------------#
    output_dir = f"./Video_clips/Youtube_Upload_folder/{YT_channel}"
    os.makedirs(output_dir, exist_ok=True)





    finalclipduration = final_clip.duration
    log(f"FINAL CLIP DURATION: {finalclipduration}")

    final_clip.write_videofile(
    Video_output_path,
    logger='bar',
    codec="libx264",
    preset="slow",
    audio_codec="aac",
    threads=8,
    ffmpeg_params=[
        "-crf", "8",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-ar", "48000",
        "-vf", "fps=30",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-movflags", "+faststart",
    ],
    audio_bitrate="384k",
    remove_temp=True
        )


    full_video.close()
    clip.close()
    clean_get_gpu_memory(threshold=0.1)

    from utility.RIFE_FPS import run_rife
    #normalized_path = _ffmpeg_minterpolate_to_fps(Video_output_path, target_fps=30)
    try:
      output_video = run_rife(Video_output_path)
      import os
      stem, ext = os.path.splitext(Video_output_path)
      expected_rife = f"{stem}_rife{ext}"
      if os.path.abspath(output_video) != os.path.abspath(expected_rife):
          try:
              os.replace(output_video, expected_rife)
              output_video = expected_rife
              log(f"[rename] Normalized RIFE output to expected name: {output_video}")
          except Exception as _e:
              log(f"[rename] Could not rename RIFE output: {_e}")
      log(f"video is completed: output path : {Video_output_path}, video name: {video_name} video_fps: {clip.fps}, codec: {video_codec}, bitrate: {bitrate}, audio_codec: {audio_codec}, subtitles: {subtitle_text} \n Final video resolution (width x height): {final_clip.size[0]} x {final_clip.size[1]}")
      clean_get_gpu_memory(threshold=0.2)
      log(f"Interpolated video saved to: {output_video}")
    except Exception as e:
         log(f"[run_rife] ERROR: {str(e)}")
    finally:

        import os
        try:
            if os.path.exists(Video_output_path):
                os.remove(Video_output_path)
        except Exception:
            pass
        # try:
        #     if normalized_path != Video_output_path and os.path.exists(normalized_path):
        #         os.remove(normalized_path)
        # except Exception:
        #     pass

def compose_montage_clips(input_paths, output_path, YT_channel, audio_path, video_path, add_bg_music=True,  bg_music_volume=0.4):
    """
    Compose the provided clips in order into a single output video.
    Attempts fast concat via ffmpeg demuxer; falls back to MoviePy re-encode if needed.
    Optionally adds background music (randomly chosen) and an optional LUT to the final composed clip.
    Args:
        input_paths (List[str]): [start_path, middle_path, ending_path]
        output_path (str): final montage output file path
        add_bg_music (bool): If True, mix random background audio into the composed clip (chosen once for the composed video)
        bg_music_candidates (Optional[List[str]]): Candidate paths for background audio. If None, defaults are used.
        bg_music_volume (float): Background music volume multiplier.
        add_random_lut (bool): If True, randomly choose a LUT (or None) and apply to the composed video.
        lut_path_candidates (Optional[List[Optional[str]]]): Candidate LUT paths including None. If None, defaults to [None, <blackwhite1.cube>].
    Returns:
        str: output_path
    """
    import os
    import subprocess
    import tempfile
    from neon.log import log

    # Extra: probe stream signatures to decide if stream copy is safe
    def _probe_signature(path):
        try:
            import ffmpeg
            info = ffmpeg.probe(path)
            v = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), {})
            a = next((s for s in info.get('streams', []) if s.get('codec_type') == 'audio'), None)
            # fps as float
            fr = v.get('avg_frame_rate') or v.get('r_frame_rate') or '0/1'
            try:
                num, den = fr.split('/')
                fps = float(num) / float(den) if float(den) else float(num)
            except Exception:
                fps = None
            sig = {
                'vcodec': v.get('codec_name'),
                'pix_fmt': v.get('pix_fmt'),
                'w': v.get('width'),
                'h': v.get('height'),
                'fps': round(fps, 3) if fps else None,
                'acodec': a.get('codec_name') if a else None,
                'ar': int(a.get('sample_rate')) if a and a.get('sample_rate') else None,
                'achannels': a.get('channels') if a else None,
                'has_audio': a is not None,
            }
            return sig
        except Exception as e:
            log(f"[compose_montage_clips] ffprobe failed for '{path}': {e}")
            return None

    # Ensure all inputs exist; allow variable number of middles (>=2 total with start and ending)
    existing = [p for p in input_paths if p and os.path.exists(p)]
    if len(existing) < 2:
        raise ValueError(f"compose_montage_clips: Expected at least 2 valid paths (start and ending), got {len(existing)}: {input_paths}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    selected_bg = None
    background_audio_path = ""
    if add_bg_music:
        log("#######choosing Background Audio########\n")
        try:
            already_uploaded_videos = f"Video_clips/Youtube_Upload_folder/{YT_channel}/already_uploaded.txt"
            from utility.Custom_Agent_Tools import Background_Audio_Decision_Model
            result = Background_Audio_Decision_Model(audio_file=audio_path,video_path=video_path,already_uploaded_videos=already_uploaded_videos,start_time=start_time,end_time=end_time)

            background_audio_path = result.get("path", "")
            if background_audio_path:
                selected_bg = background_audio_path

        except Exception as e:
            log(f"error during: [Background_Audio_Decision_Model]: {str(e)}")





    selected_lut = None

    # Helper to build ffmpeg filter args for LUT
    def _lut_ffmpeg_args(lut_path):
        if not lut_path:
            return []
        # Normalize to forward slashes and escape ':' for ffmpeg filtergraph
        p = lut_path.replace('\\', '/')
        p = p.replace(':', r'\\:')
        # Escape single quotes just in case
        p = p.replace("'", r"\\'")
        filter_arg = f"lut3d=file={p}"
        return ["-vf", filter_arg]

    # Decide if we can safely stream-copy
    sigs = [_probe_signature(p) for p in existing]
    try:
        for i, (pth, sig) in enumerate(zip(existing, sigs), start=1):
            log(f"[compose_montage_clips] part{i} sig: {{'has_audio': {sig.get('has_audio') if sig else None}, 'fps': {sig.get('fps') if sig else None}, 'vcodec': {sig.get('vcodec') if sig else None}}}")
    except Exception:
        pass

    can_stream_copy = all(s is not None for s in sigs) and sigs.count(sigs[0]) == len(sigs)

    # If bg music or LUT is requested, force re-encode to guarantee application
    need_reencode = bool(selected_bg or selected_lut)
    do_stream_copy = can_stream_copy and not need_reencode
    if not do_stream_copy and can_stream_copy:
        log("[compose_montage_clips] Re-encode forced because bg music or LUT is requested.")

    if not can_stream_copy:
        try:
            log("[compose_montage_clips] Stream parameters differ across parts; skipping fast concat and re-encoding.")
            for i, (p, s) in enumerate(zip(existing, sigs), start=1):
                log(f"  part{i}: path={p}, sig={s}")
        except Exception:
            pass

    # Try ffmpeg concat demuxer (no re-encode) ONLY when identical and no extra processing is needed
    listfile = None
    if do_stream_copy:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                listfile = f.name
                for p in existing:
                    abs_p = os.path.abspath(p)
                    f.write(f"file '{abs_p}'\n")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", listfile,
                "-c", "copy",
                output_path,
            ]
            log(f"[compose_montage_clips] ffmpeg concat (stream copy): {' '.join(cmd)}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0 and os.path.exists(output_path):
                log(f"[compose_montage_clips] Success (stream copy): {output_path}")
                return output_path
            else:
                log(f"[compose_montage_clips] ffmpeg concat failed, fallback to re-encode.\n{result.stderr[-800:]} ")
        except Exception as e:
            log(f"[compose_montage_clips] Error in ffmpeg concat: {str(e)}")
        finally:
            if listfile and os.path.exists(listfile):
                try:
                    os.remove(listfile)
                except Exception:
                    pass

    # Fallback or forced: MoviePy compose (re-encode)
    try:
        from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip, afx
        clips = [VideoFileClip(p) for p in existing]
        # Use compose to handle different sizes; enforce a consistent output fps
        target_fps = None
        try:
            target_fps = clips[0].fps if getattr(clips[0], 'fps', None) else 30
        except Exception:
            target_fps = 30
        final = concatenate_videoclips(clips, method="compose")

        # If background music is requested, mix it before writing to avoid a second pass
        if selected_bg:
            try:
                from App import mix_audio
                if final.audio is None:
                    # No original audio: just use bg music, trimmed/looped to duration
                    bg = AudioFileClip(background_audio_path)
                    if getattr(final, 'duration', None):
                        if bg.duration < final.duration:
                            bg = afx.audio_loop(bg, duration=final.duration)
                        else:
                            bg = bg.subclipped(0, final.duration)
                    final.audio = bg.volumex(bg_music_volume)
                    log("[compose_montage_clips] Composed clip had no audio; using background music only.")
                else:
                    final.audio = mix_audio(final.audio, selected_bg, bg_music_volume=bg_music_volume)
                    log(f"[compose_montage_clips] Mixed background audio into composed clip before export.")
            except Exception as e:
                log(f"[compose_montage_clips] Failed to mix background audio in fallback path: {e}")

        ffmpeg_params = [
            "-crf", "8",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-ar", "48000",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-movflags", "+faststart",
        ]
        # Append LUT filter if selected
        ffmpeg_params.extend(_lut_ffmpeg_args(selected_lut))

        final.write_videofile(
            output_path,
            logger='bar',
            codec="libx264",
            preset="slow",
            audio_codec="aac",
            threads=8,
            fps=target_fps,
            ffmpeg_params=ffmpeg_params,
            audio_bitrate="384k",
            remove_temp=True
        )
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        final.close()
        log(f"[compose_montage_clips] Success (re-encode): {output_path}")
        return output_path
    except Exception as e:
        log(f"[compose_montage_clips] MoviePy fallback failed: {str(e)}")
        raise

