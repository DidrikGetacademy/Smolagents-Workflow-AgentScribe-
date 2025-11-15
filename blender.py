import bpy
import os
import numpy as np
from typing import List, Tuple
import tempfile
import shutil
import glob
import sys
import gc
GPEN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'GPEN'))
if GPEN_PATH not in sys.path:
    sys.path.insert(0, GPEN_PATH)
from face_enhancement import FaceEnhancement
from tqdm import tqdm

# Try to use repo logger; fall back to print
try:
    from neon.log import log as _repo_log
except Exception:
    _repo_log = None

def _log(msg: str):
    """Use repo logger if available, otherwise fall back to print."""
    if _repo_log is not None:
        try:
            _repo_log(msg)
            return
        except Exception:
            pass
    print(msg)

# =============================================
# Frame-based API: frames in -> frames out (bpy)
# =============================================

#Try to use repo logger; fall back to print
try:
    from neon.print import print as _repo_log
except Exception:
    _repo_log = None

def _log(msg: str):
    if _repo_log is not None:
        try:
            _repo_log(msg)
            return
        except Exception:
            pass
    print(msg)

def _normalize_frame_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Ensure frame is RGB uint8 HxWx3."""
    if arr is None:
        raise ValueError("Frame is None")
    a = np.asarray(arr)
    if a.ndim == 2:  # gray -> RGB
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] == 4:  # RGBA -> RGB (drop A)
        a = a[:, :, :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got {a.shape}")
    return a

def enhance_frames_bpy(
    frames: List[np.ndarray],
    *,
    detail_amount: float = 0.9, #Enhances fine details, edges, and textures
    blur_radius: int = 5,        # Base blur amount
    use_skin_mask: bool = True,
    skin_key_color: Tuple[float, float, float, float] = (0.9, 0.6, 0.5, 1.0),
    mask_soften: int = 3,
    pore_amount: float = 0.35,   # Micro-contrast intensity
    glow_mix: float = 0.3,      # Cinematic glow amount
    glow_threshold: float = 0.9, # What brightness starts glowing
    vignette_strength: float = 0.2, # Edge darkening amount (reduced from 0.25)
    filmic_look: str = "Medium Contrast",
    batch_size: int = 16,       # Process frames in batches for better memory management

) -> List[np.ndarray]:
    """
    Enhance a list of RGB uint8 frames using Blender's compositor and return enhanced frames (RGB uint8).
    - Input: List[np.ndarray], each HxWx3 RGB uint8
    - Output: List[np.ndarray], same size/type
    Note: Must run inside Blender's Python (bpy available).
    Optimized to reuse compositor setup across all frames with memory batch processing.

    Args:
        batch_size: Number of frames to process in each batch for memory optimization
    """
    if not frames:
        return []

    # Normalize inputs
    norm = [_normalize_frame_rgb_uint8(f) for f in frames]
    h, w = norm[0].shape[:2]

   # _log(f"[bpy] Starting batch enhancement of {len(norm)} frames ({w}x{h})")

    # =============================
    # ONE-TIME SETUP - REUSED FOR ALL FRAMES
    # =============================
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.render.engine = 'CYCLES'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = int(w)
    scene.render.resolution_y = int(h)
    scene.render.resolution_percentage = 100
    # Ensure compositing is actually used and sequencer is disabled
    if hasattr(scene.render, 'use_compositing'):
        scene.render.use_compositing = True
    if hasattr(scene.render, 'use_sequencer'):
        scene.render.use_sequencer = False

    # Color management
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = filmic_look
    scene.view_settings.exposure = 0.0  # Increased from 0.0 for brighter output
    scene.view_settings.gamma = 1.0

    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    nodes.clear()

    # Source image datablock we'll reuse (RGBA float 0..1)
    src_img = bpy.data.images.new("BatchInput", width=w, height=h, alpha=True, float_buffer=False)

    image_node = nodes.new('CompositorNodeImage')
    image_node.image = src_img
    image_node.location = (-1200, 200)

    # Base blur
    blur = nodes.new('CompositorNodeBlur')
    blur.filter_type = 'GAUSS'
    blur.size_x = int(blur_radius)
    blur.size_y = int(blur_radius)
    blur.location = (-1000, 100)
    links.new(image_node.outputs['Image'], blur.inputs['Image'])

    # High-pass (delta = original - blurred)
    delta = nodes.new('CompositorNodeMixRGB')
    delta.blend_type = 'SUBTRACT'
    delta.inputs[0].default_value = 1.0
    delta.location = (-800, 120)
    links.new(image_node.outputs['Image'], delta.inputs[1])
    links.new(blur.outputs['Image'], delta.inputs[2])

    # Scale delta by detail_amount
    scale = nodes.new('CompositorNodeMixRGB')
    scale.blend_type = 'MIX'
    scale.inputs[0].default_value = float(detail_amount)
    scale.inputs[1].default_value = (0.0, 0.0, 0.0, 1.0)
    scale.location = (-600, 120)
    links.new(delta.outputs['Image'], scale.inputs[2])

    # enhanced_base = original + scaled_delta
    add_scaled = nodes.new('CompositorNodeMixRGB')
    add_scaled.blend_type = 'ADD'
    add_scaled.use_clamp = True
    add_scaled.inputs[0].default_value = 1.0
    add_scaled.location = (-400, 150)
    links.new(image_node.outputs['Image'], add_scaled.inputs[1])
    links.new(scale.outputs['Image'], add_scaled.inputs[2])

    # Subtle S-curve
    curves = nodes.new('CompositorNodeCurveRGB')
    curves.location = (-200, 160)
    master = curves.mapping.curves[3]
    master.points.new(0.25, 0.26)  # Shadow point: input 25% → output 26% (lighter shadows)
    master.points.new(0.75, 0.78) # Highlight point: input 75% → output 78% (brighter highlights)
    curves.mapping.update()
    links.new(add_scaled.outputs['Image'], curves.inputs['Image'])

    current_socket = curves.outputs['Image']

    # Optional skin mask to limit enhancement
    if use_skin_mask:
        keying = nodes.new('CompositorNodeKeying')
        keying.location = (-600, -140)
        # Set key color - try different approaches for compatibility
        try:
            if hasattr(keying, 'key_color'):
                keying.key_color = skin_key_color
            elif 'Key Color' in keying.inputs:
                keying.inputs['Key Color'].default_value = skin_key_color[:3]  # RGB only, no alpha
        except Exception:
            # Fallback: key color will use default if setting fails
            pass
        keying.clip_black = 0.2
        keying.clip_white = 0.8
        links.new(image_node.outputs['Image'], keying.inputs['Image'])

        mask_blur = nodes.new('CompositorNodeBlur')
        mask_blur.filter_type = 'GAUSS'
        mask_blur.size_x = int(mask_soften)
        mask_blur.size_y = int(mask_soften)
        mask_blur.location = (-380, -140)
        links.new(keying.outputs['Matte'], mask_blur.inputs['Image'])

        mix_mask = nodes.new('CompositorNodeMixRGB')
        mix_mask.blend_type = 'MIX'
        mix_mask.location = (0, 140)
        links.new(mask_blur.outputs['Image'], mix_mask.inputs[0])
        links.new(image_node.outputs['Image'], mix_mask.inputs[1])
        links.new(curves.outputs['Image'], mix_mask.inputs[2])
        current_socket = mix_mask.outputs['Image']

    # Pore/micro-contrast overlay
    hp_sharpen = nodes.new('CompositorNodeFilter')
    hp_sharpen.filter_type = 'SHARPEN'
    hp_sharpen.location = (-200, -60)
    links.new(delta.outputs['Image'], hp_sharpen.inputs['Image'])

    pores_mix = nodes.new('CompositorNodeMixRGB')
    pores_mix.blend_type = 'OVERLAY'
    pores_mix.inputs[0].default_value = float(pore_amount)
    pores_mix.location = (220, 100)
    links.new(current_socket, pores_mix.inputs[1])
    links.new(hp_sharpen.outputs['Image'], pores_mix.inputs[2])

    current_socket = pores_mix.outputs['Image']

    # Glare (fog glow)
    glare = nodes.new('CompositorNodeGlare')
    glare.glare_type = 'FOG_GLOW'
    glare.quality = 'HIGH'
    glare.mix = float(glow_mix)
    glare.threshold = float(glow_threshold)
    glare.size = 6
    glare.location = (420, -40)
    links.new(current_socket, glare.inputs['Image'])

    current_socket = glare.outputs['Image']

    # Vignette
    ellipse = nodes.new('CompositorNodeEllipseMask')
    ellipse.width = 0.95   # Increased from 0.8 to 0.95 for softer transition
    ellipse.height = 0.95  # Increased from 0.8 to 0.95 for softer transition
    ellipse.x = 0.5
    ellipse.y = 0.5
    ellipse.location = (380, -260)

    mask_blur2 = nodes.new('CompositorNodeBlur')
    mask_blur2.filter_type = 'GAUSS'
    mask_blur2.size_x = 300   # Increased from 200 to 300 for softer vignette
    mask_blur2.size_y = 300   # Increased from 200 to 300 for softer vignette
    mask_blur2.location = (560, -260)
    links.new(ellipse.outputs['Mask'], mask_blur2.inputs['Image'])

    inv = nodes.new('CompositorNodeInvert')
    inv.location = (740, -260)
    links.new(mask_blur2.outputs['Image'], inv.inputs['Color'])

    vignette_mix = nodes.new('CompositorNodeMixRGB')
    vignette_mix.blend_type = 'MULTIPLY'
    vignette_mix.inputs[0].default_value = float(vignette_strength)
    vignette_mix.location = (560, 120)
    links.new(current_socket, vignette_mix.inputs[1])
    links.new(inv.outputs['Color'], vignette_mix.inputs[2])

    # Composite output
    composite = nodes.new('CompositorNodeComposite')
    composite.location = (800, 140)
    links.new(vignette_mix.outputs['Image'], composite.inputs['Image'])

    # Also link to a Viewer node to ensure an accessible buffer in all contexts
    viewer = nodes.new('CompositorNodeViewer')
    viewer.use_alpha = True
    viewer.location = (800, -40)
    links.new(vignette_mix.outputs['Image'], viewer.inputs['Image'])

    # File Output fallback (reliable write to disk)
    tmp_dir = tempfile.mkdtemp(prefix="bpy_batch_")
    file_out = nodes.new('CompositorNodeOutputFile')
    file_out.base_path = tmp_dir
    file_out.location = (1000, 40)
    try:
        file_out.format.file_format = 'PNG'
        file_out.format.color_mode = 'RGBA'
    except Exception:
        pass
    # Use a filename prefix; Blender appends frame digits automatically
    if file_out.file_slots:
        # Ensures files like <tmp>/frame_0001.png without extra subfolder
        file_out.file_slots[0].path = "frame_"
    links.new(vignette_mix.outputs['Image'], file_out.inputs['Image'])

    # =============================
    # REUSABLE VARIABLES FOR FRAME PROCESSING
    # =============================
    enhanced: List[np.ndarray] = []
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    fallback_count = 0

    def _grab_composite_rgb() -> np.ndarray | None:
        for name in ('Render Result', 'Viewer Node'):
            im = bpy.data.images.get(name)
            if not im:
                continue
            # Expect 4 channels, width*height*4 pixels
            expected_len = h * w * 4
            px_len = len(im.pixels)
            if px_len < expected_len:
                continue
            px = np.array(im.pixels[:], dtype=np.float32)
            px = (px * 255.0).clip(0, 255).astype(np.uint8)
            try:
                px = px.reshape(h * w, 4)
            except Exception:
                continue
            return px[:, :3].reshape(h, w, 3)
        return None

    def _find_fallback_png(base_dir: str, frame_idx: int) -> str | None:
        """Return path to compositor-written PNG for this frame, handling Blender subfolder patterns."""
        # Candidate: direct prefix naming
        cand1 = os.path.join(base_dir, f"frame_{frame_idx:04d}.png")
        if os.path.isfile(cand1):
            return cand1
        # Candidate: Blender may create a subfolder named like the path prefix
        cand2 = os.path.join(base_dir, "frame_", f"{frame_idx:04d}.png")
        if os.path.isfile(cand2):
            return cand2
        # Fallback: any PNG with this frame number anywhere under base_dir
        patt = os.path.join(base_dir, "**", f"*{frame_idx:04d}.png")
        matches = glob.glob(patt, recursive=True)
        if matches:
            # Return the most recent
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]
        # As a last resort, grab any PNG just written
        any_png = glob.glob(os.path.join(base_dir, "**", "*.png"), recursive=True)
        if any_png:
            any_png.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return any_png[0]
        return None

    # =============================
    # OPTIMIZED BATCH PROCESSING LOOP
    # =============================
    def _process_frame_batch(batch_frames: List[np.ndarray], start_idx: int, batch_num: int, total_batches: int) -> List[np.ndarray]:
        """Process a batch of frames and return enhanced results."""
        batch_enhanced = []
        batch_fallback_count = 0

        # Create progress bar for frames within this batch
        frame_desc = f"Batch {batch_num}/{total_batches} - Rendering"
        for i, frame in enumerate(tqdm(batch_frames, desc=frame_desc, leave=False, position=2)):
            frame_idx = start_idx + i

            # Resize to match compositor size if needed
            if frame.shape[0] != h or frame.shape[1] != w:
                import cv2
                frame = cv2.resize(frame, (w, h))

            # Update only the image pixels - compositor setup is reused
            rgba[..., :3] = frame.astype(np.float32) / 255.0
            src_img.pixels[:] = rgba.flatten()
            src_img.update()
            scene.frame_current = frame_idx + 1

            out_rgb = None
            try:
                bpy.ops.render.render(write_still=False)
                out_rgb = _grab_composite_rgb()
            except Exception:
                out_rgb = None

            # Fallback to file output if direct composite access fails
            if out_rgb is None:
                png_path = _find_fallback_png(tmp_dir, scene.frame_current)
                if png_path and os.path.isfile(png_path):
                    try:
                        img = bpy.data.images.load(png_path, check_existing=True)
                        exp_len = img.size[0] * img.size[1] * 4
                        if len(img.pixels) >= exp_len:
                            px = np.array(img.pixels[:exp_len], dtype=np.float32)
                            px = (px * 255.0).clip(0, 255).astype(np.uint8)
                            px = px.reshape(img.size[1] * img.size[0], 4)
                            out_rgb = px[:, :3].reshape(img.size[1], img.size[0], 3)
                            if out_rgb.shape[0] != h or out_rgb.shape[1] != w:
                                import cv2
                                out_rgb = cv2.resize(out_rgb, (w, h))
                        try:
                            bpy.data.images.remove(img)
                        except Exception:
                            pass
                    except Exception:
                        out_rgb = None
                    finally:
                        # Immediately delete the written PNG to avoid leaving files behind
                        try:
                            os.remove(png_path)
                        except Exception:
                            pass

            if out_rgb is None:
                batch_enhanced.append(frame.copy())
                batch_fallback_count += 1
            else:
                batch_enhanced.append(out_rgb.copy())

        return batch_enhanced, batch_fallback_count

    try:
        # Process frames in batches for better memory management
        total_batches = (len(norm) + batch_size - 1) // batch_size
        _log(f"[bpy] Processing {len(norm)} frames in {total_batches} batches of size {batch_size}")

        # Create overall progress bar for total frames
        total_pbar = tqdm(total=len(norm), desc="Blender Enhancement", unit="frame", position=0)

        # Create main progress bar for batches
        batch_ranges = list(range(0, len(norm), batch_size))
        for batch_idx in tqdm(batch_ranges, desc="Processing batches", unit="batch", position=1, leave=False):
            # Get current batch
            batch_end = min(batch_idx + batch_size, len(norm))
            current_batch = norm[batch_idx:batch_end]
            batch_num = (batch_idx // batch_size) + 1

            # Process the batch
            batch_enhanced, batch_fallback_count = _process_frame_batch(current_batch, batch_idx, batch_num, total_batches)
            enhanced.extend(batch_enhanced)
            fallback_count += batch_fallback_count

            # Update overall progress
            total_pbar.update(len(current_batch))

            # Clean up batch memory
            del current_batch, batch_enhanced
            gc.collect()

            # Optional: Clear GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Close the overall progress bar
        total_pbar.close()

        if fallback_count:
            _log(f"[bpy] Total compositor fallbacks: {fallback_count}/{len(norm)} frames")
        else:
            _log(f"[bpy] Batch compositing success: {len(norm)} frames processed ({w}x{h})")
    finally:
        # Cleanup temp image and temp dir even on errors
        try:
            bpy.data.images.remove(src_img)
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return enhanced


if __name__ == "__main__":
    """
    Quick test script for enhance_frames_bpy function.
    Usage: Run this script from within Blender's Python console or with Blender Python executable.
    """
    try:
        import cv2

        # Test image paths to try (in order of preference)
        test_paths = [
            r"c:\Users\didri\Downloads\content.png",

        ]

        input_image = None
        used_path = None

        # Find the first available test image
        for test_path in test_paths:
            if os.path.isfile(test_path):
                print(f"Found test image: {test_path}")
                bgr = cv2.imread(test_path)
                if bgr is not None:
                    input_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    used_path = test_path
                    break

        if input_image is None:
            print("No test image found. Please place a test image at one of these locations:")
            for path in test_paths:
                print(f"  - {path}")
            print("\nOr modify the test_paths list in the script.")
            exit(1)

        print(f"Processing image: {used_path}")
        print(f"Image shape: {input_image.shape}")

        # Apply enhancement with different parameter sets for comparison
        test_configs = [
             # detail_amount: float = 0.7, #Enhances fine details, edges, and textures
            # blur_radius: int = 6,        # Base blur amount
            # use_skin_mask: bool = False,
            # skin_key_color: Tuple[float, float, float, float] = (0.85, 0.56, 0.47, 1.0),
            # mask_soften: int = 3,
            # pore_amount: float = 0.35,   # Micro-contrast intensity
            # glow_mix: float = 0.4,      # Cinematic glow amount
            # glow_threshold: float = 0.7, # What brightness starts glowing
            # vignette_strength: float = 0.25, # Edge darkening amount
            {
                "name": "detail",
                "params": {
                    "detail_amount": 0.75,
                    "blur_radius": 5,
                    "pore_amount": 0.35,
                    "glow_mix": 0.2,
                    "vignette_strength": 0.2,
                    "filmic_look": "Medium Contrast",
                    "use_skin_mask": True,
                    "mask_soften": 3,
                    "skin_key_color": (0.9, 0.6, 0.5, 1.0),
                    "glow_threshold": 0.8
                    }
            },
        ]


        def enhance_detail_and_sharpness(frame_bgr, clarity_factor=0.2, sharpen_amount=0.2):
            """
            Kombinerer detail layer clarity + mild sharpen på ett bilde.

            Args:
                frame_bgr: Inngangsbilde (BGR)
                clarity_factor: Hvor sterkt detail layer boostes (0.0–2.0)
                sharpen_amount: Hvor mye mild sharpen (0.0–2.0)
            Returns:
                Forbedret bilde (BGR)
            """

            smooth = cv2.bilateralFilter(frame_bgr, d=9, sigmaColor=75, sigmaSpace=75)


            detail_layer = cv2.subtract(frame_bgr, smooth)


            boosted_detail = cv2.addWeighted(detail_layer, clarity_factor, detail_layer, 0, 0)

            clarity_frame = cv2.add(frame_bgr, boosted_detail)

            blur = cv2.GaussianBlur(clarity_frame, (0, 0), 3)
            sharpened = cv2.addWeighted(clarity_frame, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)

            return sharpened



        def change_saturation(frame ,mode="Increase", amount=0.2):
            if mode == "Increase":
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 1]  *= (1.0 + amount)
                hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
                changed_frames = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            elif mode == "Decrease":
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 1] *= (1.0 - amount)
                hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
                changed_frames = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            return changed_frames


        # Create output directory
        output_dir = './blender_test_output'
        os.makedirs(output_dir, exist_ok=True)

        # Save original for comparison
        original_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, 'original.png'), original_bgr)
        print(f"Saved original: {output_dir}/original.png")


        # Process with different enhancement levels
        for config in test_configs:
            print(f"\nProcessing with {config['name']} settings...")

            try:
                enhanced_frames = enhance_frames_bpy([input_image], **config['params'])

                if enhanced_frames and len(enhanced_frames) > 0:
                    enhanced_rgb = enhanced_frames[0]

                # ----------------------#
                #   FACEENCHANCEMENT
                # ----------------------#
                    class Face_enchance_Args:
                            model = 'GPEN-BFR-2048'
                            task = 'FaceEnhancement'
                            key = None
                            in_size = 2048
                            out_size = 2048
                            channel_multiplier = 2
                            narrow = 1
                            alpha = 0.5
                            use_sr = True
                            use_cuda = True
                            save_face = False
                            aligned = False
                            sr_model = 'realesrnet'
                            sr_scale = 2
                            tile_size = 0
                            ext = '.png'


                    Skin_texture_enchancement = FaceEnhancement(
                        Face_enchance_Args,
                        in_size=Face_enchance_Args.in_size,
                        model=Face_enchance_Args.model,
                        use_sr=Face_enchance_Args.use_sr,
                        device='cuda' if Face_enchance_Args.use_cuda else 'cpu'
                    )
                    try:
                        Color_corrected_frames = change_saturation(enhanced_rgb, mode="Increase", amount=0.1)
                        Color_corrected_frames_bgr = cv2.cvtColor(Color_corrected_frames, cv2.COLOR_RGB2BGR)
                        sharpened_frame_bgr = enhance_detail_and_sharpness(Color_corrected_frames_bgr, clarity_factor=0.05, sharpen_amount=0.00)
                        enchanced_frame, _, _ = Skin_texture_enchancement.process(sharpened_frame_bgr)
                        print("done gpen face enchancement ")
                    except Exception as e:
                        print(f"[FaceEnhancement] Error during Face Enhancement: {e}")

                    output_path = os.path.join(output_dir, f'enhanced_{config["name"]}.png')
                    cv2.imwrite(output_path, enchanced_frame)
                    print(f"✓ Saved {config['name']} enhancement: {output_path}")
                else:
                    print(f"✗ Failed to enhance with {config['name']} settings")

            except Exception as e:
                print(f"✗ Error with {config['name']} settings: {e}")

        print(f"\nAll test images saved to: {output_dir}")
        print("Compare the results to see the enhancement effects!")

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Make sure OpenCV is installed: pip install opencv-python")
    except Exception as e:
        print(f'Test error: {e}')
        import traceback
        traceback.print_exc()


