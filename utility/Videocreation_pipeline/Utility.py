import cv2
from moviepy import vfx,AudioFileClip,afx,CompositeAudioClip,afx
from moviepy.audio.fx import MultiplyVolume
from ultralytics.utils.ops import non_max_suppression
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import torch
from utility.clean_memory import clean_get_gpu_memory
from ..log import log
def change_brightness(frame, amount=0.2):
    """
    Darken or brighten the frame.
    amount > 0 : brighten
    amount < 0 : darken
    Optimized to avoid float32 conversion
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # Work with V channel as int16 to avoid overflow, then clip
    v_channel = hsv[..., 2].astype(np.int16)
    v_channel = np.clip(v_channel * (1.0 + amount), 0, 255).astype(np.uint8)
    hsv[..., 2] = v_channel
    changed_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return changed_frame

def change_saturation(frame, mode="Increase", amount=0.2):
    """
    Change saturation without float32 conversion to save memory.
    Uses int16 temporarily to avoid overflow.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    if mode == "Grayscale":
        hsv[..., 1] = 0  # Set saturation to 0 for completely gray frames
    else:
        # Work with S channel as int16 to avoid overflow during multiplication
        s_channel = hsv[..., 1].astype(np.int16)
        if mode == "Increase":
            s_channel = np.clip(s_channel * (1.0 + amount), 0, 255).astype(np.uint8)
        elif mode == "Decrease":
            s_channel = np.clip(s_channel * (1.0 - amount), 0, 255).astype(np.uint8)
        hsv[..., 1] = s_channel

    # Convert back to RGB
    changed_frames = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return changed_frames

def _resize_smart(img, target_w: int, target_h: int):
    """Resize with better interpolation:
    - INTER_AREA when downscaling
    - INTER_CUBIC when upscaling
    """
    h, w = img.shape[:2]
    if target_w < w or target_h < h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    return cv2.resize(img, (target_w, target_h), interpolation=interp)

def mix_audio(original_audio, background_music_path, bg_music_volume=0.25):
        bg_music = AudioFileClip(background_music_path)
        if bg_music.duration < original_audio.duration:
            bg_music = bg_music.with_effects([afx.AudioLoop(duration=original_audio.duration)])
        else:
            bg_music = bg_music.subclipped(0, original_audio.duration)
        bg_music = bg_music.with_effects([MultiplyVolume(bg_music_volume)])
        original_audio = original_audio.with_effects([MultiplyVolume(1.0)])
        mixed_audio = CompositeAudioClip([original_audio, bg_music])
        log(f"[mix_audio] --> (original_audio.duration): {original_audio.duration}, (bg_music.duration): {bg_music.duration}, (mixed_audio.duration): {mixed_audio.duration}")
        return mixed_audio

def parse_editing_notes(notes):
        fade_in_duration = 0
        volume_reduction = 1.0
        import re

        fade_in_match = re.search(r'Fade in (?over|at) (\d+\.?\d*) seconds?', notes, re.IGNORECASE)
        if fade_in_match:
            fade_in_duration = float(fade_in_match.group(1))

        volume_match = re.search(r'lower volume by (\d+\.?+\d)%?', notes, re.IGNORECASE)
        if volume_match:
              percentage = float(volume_match.group(1))
              volume_reduction = 1.0 - (percentage / 100.0)

        return  {"fade_in_duration": fade_in_duration, "volume_reduction": volume_reduction}

def load_cube_lut(path):
    lut_data = []
    size = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
            elif line[0].isdigit() or line[0] == "-":
                values = [float(v) for v in line.split()]
                lut_data.append(values)

    if size is None:
        raise ValueError("Invalid .cube file: missing LUT_3D_SIZE")

    lut = np.array(lut_data).reshape((size, size, size, 3))
    return lut

def mix_audio_with_effects(original_audio, bg_music_path, bg_music_volume=0.42, editing_notes=""):
        bg_music = AudioFileClip(bg_music_path)


        if bg_music.duration < original_audio.duration:
            bg_music = afx.audio_loop(bg_music, duration=original_audio.duration)
        else:
            bg_music = bg_music.subclipped(0, original_audio.duration)


        effects = parse_editing_notes(editing_notes)
        fade_in_duration = effects["fade_in_duration"]
        volume_reduction = effects["volume_reduction"]
        time_specific = effects["time_specific"]


        if fade_in_duration > 0:
            bg_music = bg_music.fx(vfx.FadeIn, duration=fade_in_duration)


        if time_specific:
            print(f"Warning: Time-specific ducking requested ({time_specific}), but no timestamps provided. Applying uniform volume reduction.")
            bg_music = bg_music.volumex(bg_music_volume * volume_reduction)
        else:
            bg_music = bg_music.volumex(bg_music_volume * volume_reduction)

        original_audio = original_audio.volumex(1.0)

        mixed_audio = CompositeAudioClip([original_audio, bg_music])
        return mixed_audio

def detect_and_crop_frames_batch(frames, batch_size=8):
        TARGET_W, TARGET_H = 1080, 1920
        alpha = 0.1
        prev_cx, prev_cy = None, None
        cropped_frames = []
        onnx_path_gpu = r"c:\Users\didri\Desktop\LLM-models\Face-Detection-Models\yolov8x-face-lindevs_cuda.onnx"
        providers = ['CUDAExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 0
        session = ort.InferenceSession(onnx_path_gpu, sess_options, providers=providers)
        input_name = session.get_inputs()[0].name
        print("ONNX input shape:", session.get_inputs()[0].shape)
        progress_bar = tqdm(total=len(frames), desc="[detect_and_crop_frames_batch]Processing frames", unit="frame", dynamic_ncols=True)
        try:
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i+batch_size]
                original_count = len(batch)


                if len(batch) < batch_size:
                    pad_count = batch_size - len(batch)
                    batch += [np.zeros_like(batch[0])] * pad_count


                processed_batch = []
                for frame in batch:

                    img = cv2.resize(frame, (928, 928), interpolation=cv2.INTER_AREA)
                    img = img.astype(np.float32) / 255.0
                    processed_batch.append(img.transpose(2, 0, 1))
                    progress_bar.set_postfix({
                                            "GPU Mem": f"{torch.cuda.memory_allocated()/1e9:.1f}GB",
                                            "Batch Size": f"{len(batch[:original_count])}/{batch_size}"
                                        })
                    progress_bar.update(1)

                if len(processed_batch) < batch_size:
                    processed_batch += [np.zeros((3, 928, 928), dtype=np.float32)] * (batch_size - len(processed_batch))

                input_tensor = np.stack(processed_batch).astype(np.float32)


                outputs = session.run(None, {input_name: input_tensor})[0]


                predictions = torch.tensor(outputs[:original_count])
                detections = non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45)


                for idx, (frame, det) in enumerate(zip(batch[:original_count], detections)):
                    h, w = frame.shape[:2]

                    if det is not None and len(det):

                        areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                        max_idx = torch.argmax(areas)
                        x1, y1, x2, y2 = det[max_idx, :4].cpu().numpy().astype(int)


                        x1 = int(x1 * w / 928)
                        y1 = int(y1 * h / 928)
                        x2 = int(x2 * w / 928)
                        y2 = int(y2 * h / 928)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    else:
                        cx, cy = w // 2, h // 2


                    if prev_cx is None or prev_cy is None:
                        sx, sy = cx, cy
                    else:
                        sx = int(alpha * cx + (1 - alpha) * prev_cx)
                        sy = int(alpha * cy + (1 - alpha) * prev_cy)
                    prev_cx, prev_cy = sx, sy


                    aspect_ratio = TARGET_W / TARGET_H
                    if w / h > aspect_ratio:
                        crop_h = h
                        crop_w = int(h * aspect_ratio)
                    else:
                        crop_w = w
                        crop_h = int(w / aspect_ratio)

                    x0 = max(0, min(sx - crop_w // 2, w - crop_w))
                    y0 = max(0, min(sy - crop_h // 2, h - crop_h))

                    cropped_frame = frame[y0:y0+crop_h, x0:x0+crop_w]
                    if cropped_frame.shape[:2] != (TARGET_H, TARGET_W):
                        cropped_frame = _resize_smart(cropped_frame, TARGET_W, TARGET_H)
                    cropped_frames.append(cropped_frame)
        finally:
            progress_bar.close()
            del  predictions, detections, frame, det, h, w, areas, max_idx,session
            if batch is not None:
                del batch
            session = None
            clean_get_gpu_memory(threshold=0.1)
        return cropped_frames

def compose_montage_clips(input_paths, output_path, YT_channel, audio_path, video_path,start_time,end_time, add_bg_music=True,  bg_music_volume=0.4):
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
    from log import log

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
            already_uploaded_videos = f"Video_clips/Youtube_Upload_folder/{YT_channel}/already_uploaded.json"
            from Agents.utility.Agent_tools import Background_Audio_Decision_Model
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

