import bpy
import os
import numpy as np
from typing import List

#https://www.google.com/search?q=bpy+version+5.0.0+blender+python+module&sca_esv=abeb4f522ce11e62&rlz=1C1ASUM_enNO1143NO1143&ei=0iRAadCEOIunwPAPu57M6AU&ved=0ahUKEwjQtY2P8L-RAxWLExAIHTsPE10Q4dUDCBA&uact=5&oq=bpy+version+5.0.0+blender+python+module&gs_lp=Egxnd3Mtd2l6LXNlcnAiJ2JweSB2ZXJzaW9uIDUuMC4wIGJsZW5kZXIgcHl0aG9uIG1vZHVsZTIFEAAY7wUyBRAAGO8FMggQABiABBiiBEiCJVDrC1jiIXADeAGQAQCYAbUBoAHIEKoBBDAuMTa4AQPIAQD4AQGYAgqgAsAHwgIKEAAYRxjWBBiwA8ICChAhGAoYoAEYwwSYAwCIBgGQBgiSBwMzLjegB8o4sgcDMC43uAe4B8IHAzMuN8gHDIAIAQ&sclient=gws-wiz-serp
import sys
import gc
GPEN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'GPEN'))
if GPEN_PATH not in sys.path:
    sys.path.insert(0, GPEN_PATH)
from GPEN.face_enhancement import FaceEnhancement
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
    from neon.log import print as _repo_log
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
    detail_amount: float = 0.7,
    view_transform: str = "Filmic",
    blur_radius: int = 15,
    use_skin_mask: bool = False,
    mask_soften: int = 3,
    pore_amount: float = 0.5,
    glow_mix: float = 0.0,
    glow_threshold: float = 0.9,
   # skin_key_color: tuple = (0.85, 0.56, 0.47, 1.0),
    vignette_strength: float = 0.22,
    filmic_look: str = "Medium High Contrast",
    batch_size: int = 16,  # For memory management (not parallel processing)
) -> List[np.ndarray]:
    """
    Enhance frames using Blender compositor with MAXIMUM performance optimizations.

    CRITICAL: Blender compositor is SINGLE-THREADED and processes one frame at a time.
    batch_size is only for memory management, NOT parallel processing.

    Optimizations applied:
    - NO disk I/O (direct memory access only)
    - Pre-allocated buffers (zero allocation overhead)
    - Memory-mapped pixel access
    - Simplified compositor tree (fewer nodes)
    - Direct Viewer node access (no file fallback)
    """
    if not frames:
        return []

    # Normalize inputs
    norm = [_normalize_frame_rgb_uint8(f) for f in frames]
    h, w = norm[0].shape[:2]

    _log(f"[bpy] OPTIMIZED compositor processing {len(norm)} frames ({w}x{h})")

    # =============================
    # ONE-TIME SETUP - REUSED FOR ALL FRAMES
    # =============================
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.render.engine = 'CYCLES'
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX" # or "CUDA", etc.
    bpy.context.preferences.addons["cycles"].preferences.refresh_devices()
    devices = bpy.context.preferences.addons["cycles"].preferences.devices
    for d in devices:
        d.use = True # Enable all detected devices
        print(f"Enabled device: {d.name} ({d.type})")

    # 4. Ensure the scene is set to use GPU compute
    bpy.context.scene.cycles.device = "GPU"
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = int(w)
    scene.render.resolution_y = int(h)
    scene.render.resolution_percentage = 100

    # CRITICAL: Enable compositor, disable everything else for max speed
    if hasattr(scene.render, 'use_compositing'):
        scene.render.use_compositing = True
    if hasattr(scene.render, 'use_sequencer'):
        scene.render.use_sequencer = False

    # Disable unnecessary render features for speed
    scene.render.use_motion_blur = False
    scene.render.use_border = False
    scene.render.use_crop_to_border = False

    # Color management
    scene.view_settings.view_transform = view_transform
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

    # Glare (fog glow) - conditionally applied to avoid color artifacts in grayscale
    if glow_mix > 0.0:
        glare = nodes.new('CompositorNodeGlare')
        glare.glare_type = 'FOG_GLOW'
        glare.quality = 'HIGH'
        glare.mix = float(glow_mix)
        glare.threshold = float(glow_threshold)
        glare.size = 6
        glare.location = (420, -40)
        links.new(current_socket, glare.inputs['Image'])
        current_socket = glare.outputs['Image']
    # If glow_mix is 0, skip glare entirely to preserve grayscale

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

    # OPTIMIZATION: Use ONLY Viewer node - NO disk I/O for maximum speed
    viewer = nodes.new('CompositorNodeViewer')
    viewer.use_alpha = True
    viewer.location = (800, -40)
    links.new(vignette_mix.outputs['Image'], viewer.inputs['Image'])

    # =============================
    # PRE-ALLOCATED BUFFERS (CRITICAL OPTIMIZATION)
    # =============================
    enhanced: List[np.ndarray] = []

    # Pre-allocate RGBA input buffer (reused for ALL frames)
    rgba_input = np.zeros((h, w, 4), dtype=np.float32)
    rgba_input[..., 3] = 1.0  # Alpha channel always 1.0

    # Pre-allocate output buffer (reused for ALL frames)
    output_buffer = np.zeros((h * w * 4,), dtype=np.float32)

    fallback_count = 0
    expected_px_len = h * w * 4

    def _grab_composite_rgb_optimized() -> np.ndarray | None:
        """OPTIMIZED: Direct memory access from Viewer Node - zero allocation overhead."""
        # Try Viewer Node first (fastest)
        viewer_img = bpy.data.images.get('Viewer Node')
        if viewer_img and len(viewer_img.pixels) >= expected_px_len:
            # CRITICAL: Use foreach_get for MUCH faster pixel access (10-100x faster)
            viewer_img.pixels.foreach_get(output_buffer)
            # Convert float [0,1] to uint8 [0,255] in-place
            np.multiply(output_buffer, 255.0, out=output_buffer)
            np.clip(output_buffer, 0, 255, out=output_buffer)
            # Reshape and extract RGB (no alpha)
            return output_buffer.astype(np.uint8).reshape(h, w, 4)[:, :, :3].copy()

        # Fallback to Render Result
        render_img = bpy.data.images.get('Render Result')
        if render_img and len(render_img.pixels) >= expected_px_len:
            render_img.pixels.foreach_get(output_buffer)
            np.multiply(output_buffer, 255.0, out=output_buffer)
            np.clip(output_buffer, 0, 255, out=output_buffer)
            return output_buffer.astype(np.uint8).reshape(h, w, 4)[:, :, :3].copy()

        return None

    # =============================
    # MAXIMUM SPEED PROCESSING LOOP
    # =============================
    try:
        total_batches = (len(norm) + batch_size - 1) // batch_size
        _log(f"[bpy] Processing {len(norm)} frames ({total_batches} memory batches of {batch_size})")

        # Single progress bar for all frames
        with tqdm(total=len(norm), desc="Blender Enhancement", unit="frame") as pbar:
            for batch_idx in range(0, len(norm), batch_size):
                batch_end = min(batch_idx + batch_size, len(norm))
                current_batch = norm[batch_idx:batch_end]

                # Process each frame in the current batch
                for frame_offset, frame in enumerate(current_batch):
                    frame_idx = batch_idx + frame_offset

                    # Resize if needed (smart interpolation)
                    if frame.shape[0] != h or frame.shape[1] != w:
                        import cv2
                        interp = cv2.INTER_AREA if (w < frame.shape[1] or h < frame.shape[0]) else cv2.INTER_CUBIC
                        frame = cv2.resize(frame, (w, h), interpolation=interp)

                    # CRITICAL: Update pixels in pre-allocated buffer (zero-copy where possible)
                    np.divide(frame, 255.0, out=rgba_input[:, :, :3], casting='unsafe')

                    # CRITICAL: Use foreach_set for 10-100x faster upload (vs pixels[:] = ...)
                    src_img.pixels.foreach_set(rgba_input.flatten())
                    src_img.update()

                    # Render compositor (single-threaded, unavoidable bottleneck)
                    scene.frame_current = frame_idx + 1

                    try:
                        bpy.ops.render.render(write_still=False)
                        out_rgb = _grab_composite_rgb_optimized()
                    except Exception as e:
                        _log(f"[bpy] Render error frame {frame_idx}: {e}")
                        out_rgb = None

                    # Handle fallback (return original frame if compositor fails)
                    if out_rgb is None:
                        enhanced.append(frame.copy())
                        fallback_count += 1
                    else:
                        enhanced.append(out_rgb)

                    pbar.update(1)

                # Memory management: cleanup after each batch
                del current_batch
                gc.collect()

                # Optional GPU memory cleanup
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        # Final statistics
        if fallback_count:
            _log(f"[bpy] Compositor fallbacks: {fallback_count}/{len(norm)} frames")
        else:
            _log(f"[bpy] SUCCESS: {len(norm)} frames enhanced ({w}x{h})")

    finally:
        # Cleanup source image
        try:
            bpy.data.images.remove(src_img)
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
            r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\neon\content (2).png",

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
        print(f"Image dtype: {input_image.dtype}")
        print(f"Image value range: {input_image.min()} - {input_image.max()}")

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
                    "detail_amount": 0.6,
                    "blur_radius": 5,
                    "pore_amount": 0.5,
                    "glow_mix": 0.3,
                    "vignette_strength": 0.2,
                    "filmic_look": "Medium Contrast",
                    "use_skin_mask": True,
                    "mask_soften": 0,
                    "skin_key_color": (0.9, 0.6, 0.5, 1.0),
                    "glow_threshold": 0.5
                    }
            },
        ]




        def change_saturation(frame, mode="Increase", amount=0.15):
            """
            Change saturation of RGB image.
            Args:
                frame: RGB image (numpy array)
                mode: "Increase" or "Decrease"
                amount: saturation change amount (0.0 to 1.0+)
            Returns:
                RGB image with adjusted saturation
            """
            # Ensure frame is uint8 RGB
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            if mode == "Increase":
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 1] *= (1.0 + amount)
                hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
                changed_frames = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            elif mode == "Decrease":
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 1] *= (1.0 - amount)
                hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
                changed_frames = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            else:
                changed_frames = frame.copy()

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
                    print(f"Enhanced RGB shape: {enhanced_rgb.shape}, dtype: {enhanced_rgb.dtype}")
                    print(f"Enhanced RGB range: {enhanced_rgb.min()} - {enhanced_rgb.max()}")

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
                        Color_corrected_frames = change_saturation(enhanced_rgb, mode="Decrease", amount=1.0)  # Make grayscale
                        print(f"After saturation: shape={Color_corrected_frames.shape}, dtype={Color_corrected_frames.dtype}, range={Color_corrected_frames.min()}-{Color_corrected_frames.max()}")

                        enchanced_frame, _, _ = Skin_texture_enchancement.process(Color_corrected_frames)
                        print(f"After GPEN: shape={enchanced_frame.shape}, dtype={enchanced_frame.dtype}, range={enchanced_frame.min()}-{enchanced_frame.max()}")

                        # Convert RGB to BGR for OpenCV saving
                        enchanced_frame_bgr = cv2.cvtColor(enchanced_frame, cv2.COLOR_RGB2BGR)
                        print("done gpen face enchancement - converted to BGR for saving")
                    except Exception as e:
                        print(f"[FaceEnhancement] Error during Face Enhancement: {e}")
                        # Fallback to enhanced RGB if face enhancement fails
                        enchanced_frame_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
                        print("Using fallback - converted enhanced RGB to BGR")

                    output_path = os.path.join(output_dir, f'enhanced_{config["name"]}.png')
                    cv2.imwrite(output_path, enchanced_frame_bgr)
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


