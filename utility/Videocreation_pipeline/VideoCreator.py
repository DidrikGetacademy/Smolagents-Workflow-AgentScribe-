
from ..log import log
from ..clean_memory import clean_get_gpu_memory
from moviepy import VideoFileClip, ImageSequenceClip, TextClip, CompositeVideoClip,vfx,AudioFileClip,afx,CompositeAudioClip,afx
import ffmpeg
from moviepy.video.fx.CrossFadeIn import CrossFadeIn
from moviepy.video.fx.CrossFadeOut import CrossFadeOut
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
import utility.Global_state as Global_state
from Agents.utility.Agent_tools import Background_Audio_Decision_Model
from .Utility import detect_and_crop_frames_batch,mix_audio
from utility.reload_model import Reload_and_change_model
from .blender  import enhance_frames_bpy
import os
import random
import gc
import sys
REPO_ROOT = os.path.abspath('.')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
GPEN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'GPEN'))
if GPEN_PATH not in sys.path:
    sys.path.insert(0, GPEN_PATH)
from GPEN.face_enhancement import FaceEnhancement
import GPEN.__init_paths
#------------------------------------------------------------------------------------------------------------------------#
# create_short_video --> Function takes (video, start time/end time for video, video name, subtitles for video) as input
#------------------------------------------------------------------------------------------------------------------------#
def create_short_video(video_path, audio_path, start_time, end_time, video_name, subtitle_text, order = None, Video_output_path=None, YT_channel = None, middle_order=None,Montage_Flag=False):
    probe = ffmpeg.probe(video_path)
    log(probe)
    format_info = probe.get('format', {})
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    bitrate = int(format_info.get('bit_rate', 0))
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
         PAUSE_THRESHOLD = 0.2

         for i in range(1, len(subtitle_words)):
              prev_word = subtitle_words[i-1]
              curr_word = subtitle_words[i]
              gap = float(curr_word['start']) - float(prev_word['end'])

              if gap > PAUSE_THRESHOLD:
                  current_segment[-1]['end'] += 0.2 # siste ordet blir rendera litt lengre.
                  segments.append(current_segment)
                  current_segment = []

              current_segment.append(curr_word)

         if current_segment:
              segments.append(current_segment)

         MAX_WORDS_PER_CHUNK = 5

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
       triplets = group_subtitle_words_in_triplets(subtitle_text)
    except Exception as e:
         log(f"[group_subtitle_words_in_triplets] Error during grouping of subtitles in triplets. {str(e)} ")


    def create_subtitles_from_triplets(triplets):
        text_clips = []

        for i, c in enumerate(triplets):

            _text = c['text']
            _text = _text.upper()
            log(f"text: {_text}")
            txt_clip = TextClip(
                text=_text,
                font=r"C:\WINDOWS\FONTS\COPPERPLATECC-BOLD.TTF",
                font_size=40,
                margin=(10, 10),
                text_align="center",
                vertical_align="center",
                horizontal_align="center",
                color='white',
                stroke_color="black",
                stroke_width=2,
                size=(1200, 400),
                method="label",
                duration=c['duration']
            ).with_start(c['start']).with_position(('center', 0.45), relative=True).with_effects([CrossFadeIn(0.15), CrossFadeOut(0.15)])



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
    log(f"[Extracting original video frames]  1. frames: {len(frames)} frames.\n  Height: {frame_height}, Width: {frame_width}")






# --------------------------------#
# Yolo8/facedetection + Cropping
# --------------------------------#
    cropped_frames = detect_and_crop_frames_batch(frames=frames,batch_size=8)
    clean_get_gpu_memory(threshold=0.1)
    del frames
    frame_height, frame_width = cropped_frames[0].shape[:2]
    log(f"[cropped_frames]  1. frames: {len(cropped_frames)} frames.\n Height: {frame_height}, Width: {frame_width}")




# --------------------------------------#
#  Blender version 5.0.0 detail enahcement
# ----------------------------------------#
    log(f"blender frames proccessing now.")
    gc.collect()
    gc.collect()
    blender_frames = enhance_frames_bpy(cropped_frames, batch_size=10,use_skin_mask=True)
    del cropped_frames
    clean_get_gpu_memory(threshold=0.1)






# ----------------------#
#   FACEENCHANCEMENT
# # ---------------------#
    class Face_enchance_Args:
            model = 'GPEN-BFR-2048'
            task = 'FaceEnhancement'
            key = None
            in_size = 2048
            out_size = 0
            channel_multiplier = 2
            narrow = 1
            alpha = 0.35
            use_sr = True
            use_cuda = True
            save_face = False
            aligned = False
            sr_model = 'realesrnet'
            sr_scale = 2
            tile_size = 512
            ext = '.png'
    face_args = Face_enchance_Args()
    Skin_texture_enchancement = FaceEnhancement(
        face_args,
        in_size=face_args.in_size,
        model=face_args.model,
        use_sr=face_args.use_sr,
        device='cuda'
    )
    FaceEnhancement_frames = []
    try:
         for f_frames in tqdm(blender_frames, desc="[FaceEnhancement]  proccessing frames", unit="frame"):
              frame_height, frame_width = f_frames.shape[:2]
              bgr_frame = cv2.cvtColor(f_frames, cv2.COLOR_RGB2BGR)
              enchanced_frame_bgr, _, _ = Skin_texture_enchancement.process(bgr_frame)
              RGB_face_enchanced_frame = cv2.cvtColor(enchanced_frame_bgr, cv2.COLOR_BGR2RGB)
              FaceEnhancement_frames.append(RGB_face_enchanced_frame)
              del enchanced_frame_bgr,RGB_face_enchanced_frame
              gc.collect()
         del  Skin_texture_enchancement,bgr_frame
         frame_height, frame_width = FaceEnhancement_frames[0].shape[:2]
         log(f"[FaceEnhancement]  1. frames: {len(FaceEnhancement_frames)} frames.\n Height: {frame_height}, Width: {frame_width}")

         clean_get_gpu_memory(threshold=0.1)
    except Exception as e:
            log(f"[FaceEnhancement] Error: {str(e)}")




#-------------------------------#
# Creating videoclip from frames
#-------------------------------#
    try:
       processed_clip = ImageSequenceClip(cropped_frames, fps=clip.fps)
       clean_get_gpu_memory(threshold=0.1)
       gc.collect()
    except Exception as e:
         log(f"[processed_clip] ERROR: {str(e)}")



#-----------------------------------------------#
#     Adds Logo/overlay for YT_channel
#-----------------------------------------------#
    if Montage_Flag:

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
    else:
        try:
            global Upload_YT_count
            log(f"current YT_count: {Upload_YT_count}")
            with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Youtube_Upload_folder\latest_uploaded.txt","r", encoding="UTF-8") as r:
                Latest_Yt_channel = r.read().strip()
                log(f"Latest_Yt_channel: {Latest_Yt_channel}")
                YT_channel = Latest_Yt_channel

            if Latest_Yt_channel == "MA_Youtube":
                YT_channel = "LR_Youtube"
                overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LR.mp4",has_mask=True)
                log(f"YT_channel: {YT_channel}")
            elif Latest_Yt_channel == "LR_Youtube":
                YT_channel = "LRS_Youtube"
                overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LRS.mp4",has_mask=True)
                log(f"YT_channel: {YT_channel}")
            elif Latest_Yt_channel == "LRS_Youtube":
                YT_channel = "LM_Youtube"
                overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LM.mp4",has_mask=True)
                log(f"YT_channel: {YT_channel}")
            elif Latest_Yt_channel == "LM_Youtube":
                YT_channel = "MR_Youtube"
                overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_MR.mp4",has_mask=True)
                log(f"YT_channel: {YT_channel}")
            elif Latest_Yt_channel == "MR_Youtube":
                    YT_channel = "MA_Youtube"
                    overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_MA.mp4",has_mask=True)
                    log(f"YT_channel: {YT_channel}")

            Global_state.set_current_yt_channel(YT_channel)
            with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Youtube_Upload_folder\latest_uploaded.txt","w", encoding="UTF-8") as w:
                w.write(YT_channel)

            overlay_clip = overlay_clip.subclipped(0, clip.duration)
            logo_mask = overlay_clip.to_mask()
            logo_with_mask = overlay_clip.with_mask(logo_mask)

            if Upload_YT_count == 4:
                Upload_YT_count = 0

        except Exception as e:
            log(f"Error:  {str(e)}")






    try:
        subtitle_clips = create_subtitles_from_triplets(triplets)
    except Exception as e:
            log(f"error during [create_subtitles_from_triplets]: {str(e)}")



#-----------------------------------------------#
# Adds Subtitles + Logo to the video
#-----------------------------------------------#
    final_clip = CompositeVideoClip(
                [processed_clip.with_position('center')]  + subtitle_clips +  [logo_with_mask.with_position('center',0.50)] ,
                size=processed_clip.size
                )

    del logo_with_mask, subtitle_clips, processed_clip
    clean_get_gpu_memory(threshold=0.1)


    fade = CrossFadeIn(1.5)
    final_clip = fade.apply(final_clip)

# -------------------------------#
# Adds Background Music to video
# -------------------------------#
    log("choosing Background Audio\n")
    try:
        already_uploaded_videos = f"Video_clips/Youtube_Upload_folder/{YT_channel}/already_uploaded.json"
        log(f"already_uploaded_videos: {already_uploaded_videos}")
        result = Background_Audio_Decision_Model(audio_file=audio_path,video_path=video_path,already_uploaded_videos=already_uploaded_videos,start_time=start_time,end_time=end_time)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            log(f"Removed temporary audio file: {audio_path}")

        background_audio = result.get("path", "")
        song_name = result.get("song_name", "")
        Background_Audio_Reason = result.get("reason", "")
        log(f"Background Audio : {background_audio}, song_name: {song_name}, Background_Audio_Reason: {Background_Audio_Reason}")
    except Exception as e:
         log(f"error during: [Background_Audio_Decision_Model]: {str(e)}")


    log(f"Background audio path: {background_audio} \n Reason: {Background_Audio_Reason}\n")

    if background_audio:
        background_music_path = background_audio
        final_clip.audio = mix_audio(clip.audio, background_music_path, bg_music_volume=0.35)
    else:
        final_clip.audio = clip.audio
        log(f"keeping original audio")
        Background_Audio_Reason = "original audio only"

    log(f"(AUDIO DURATION): {final_clip.audio.duration}")
    clean_get_gpu_memory(threshold=0.1)





#-----------------------------------------------------#
#    Adds fade in/out to the video and sets the FPS
#------------------------------------------------------#
    final_clip = FadeIn(duration=0.1).apply(final_clip)
    final_clip = FadeOut(duration=0.1).apply(final_clip)
    final_clip.fps = clip.fps



#-----------------------------------------------------#
#    Cleans cpu/gpu memory
#------------------------------------------------------#
    clean_get_gpu_memory(threshold=0.0)


#-----------------------------------------------------#
#    Writes the final Video
#------------------------------------------------------#
    output_dir = f"./Video_clips/Youtube_Upload_folder/{YT_channel}"
    os.makedirs(output_dir, exist_ok=True)

    if Video_output_path:
         out_path = Video_output_path
    else:
        out_path = os.path.join(output_dir, f"{video_name}.mp4")


    _finalclipduration = final_clip.duration
    log(f"FINAL CLIP DURATION: {_finalclipduration}")

    lut_cube_path = "./Video_clips/Utils-Video_creation/LUT/black/RMP_BW709_1.cube"
    vf_filters = []
    apply_lut = True
    if os.path.isfile(lut_cube_path):
        apply_lut = random.choice([True, False])
        if apply_lut:
            vf_filters.append(f"lut3d=file='{lut_cube_path}'")
    else:
        log(f"[ffmpeg] LUT file missing, skipping: {lut_cube_path}")
    vf_filters.append("minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd_threshold=50")
    vf_chain = ",".join(vf_filters)
    log(f"[ffmpeg] LUT applied: {apply_lut} path: {lut_cube_path if apply_lut else 'none'}")

    final_clip.write_videofile(
    out_path,
    logger='bar',
    codec="libx264",
    preset="slow",
    audio_codec="aac",
    threads=6,
    ffmpeg_params=[
        "-crf", "10",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-ar", "48000",
        "-vf", vf_chain,
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-movflags", "+faststart",
    ],
    audio_bitrate="384k",
    remove_temp=True
        )


    gc.collect()
    full_video.close()
    clip.close()
    clean_get_gpu_memory(threshold=0.1)


#-----------------------------------------------------#
#    Boosts x2 FPS on full video
#------------------------------------------------------#
    from utility.RIFE_FPS import run_rife
    try:
      output_video = run_rife(out_path)
      log(f"video is completed: output path : {out_path}, video name: {video_name} video_fps: {clip.fps}, codec: {video_codec}, bitrate: {bitrate}, audio_codec: {audio_codec}, subtitles: {subtitle_text} \n Final video resolution (width x height): {final_clip.size[0]} x {final_clip.size[1]}")
      log(f"Interpolated video saved to: {output_video}")
    except Exception as e:
         log(f"[run_rife] ERROR: {str(e)}")
    finally:
        os.remove(out_path)
        clean_get_gpu_memory(threshold=0.1)

    if Montage_Flag:
        try:
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
    else:
        try:
            global Global_model
            if 'Global_model' not in globals():
                    Global_model = None
            from utility.upload_Socialmedia import upload_video
            log(f"\n\n\n\n\n\n\n\n\n\n-------------------------------------------------------------------\n\n\n\n")
            log("Youtube uploading & agent STARTING....")

            try:
                Current_model_loaded = Global_state.get_current_global_model()
                if Global_model and Current_model_loaded == "gpt-4o":
                    del Global_model
                    Global_model = None
                    clean_get_gpu_memory(threshold=0.3)
            except NameError:
                    pass


            if Global_model is None:
                try:
                    Global_model = Reload_and_change_model(model_name="gpt-5-minimal", message="Reloading model to -> gpt-5-minimal before running [upload_video]")
                except Exception as e:
                    log(f"Error reloading and changing model to gpt-5: {str(e)}")

            if Global_model is None:
                raise ValueError("Failed to initialize Global_model for upload_video")

            YT_channel = Global_state.get_current_yt_channel()

            try:
                social_media = upload_video(model=Global_model,file_path=output_video,subtitle_text=subtitle_text,YT_channel=YT_channel,background_audio_=Background_Audio_Reason,song_name=song_name,video_duration=_finalclipduration)
            except Exception as e:
                log(f"error during upload_video: {str(e)}")
            log(f"Done with uploading to {social_media}")
        except Exception as e:
            log(f"error during uploading: {str(e)}")

        finally:
            clean_get_gpu_memory(threshold=0.8)
