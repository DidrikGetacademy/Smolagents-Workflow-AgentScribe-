#!/usr/bin/env python3
"""
Script to apply a .cube LUT (Look-Up Table) to a PNG image.

Hardcoded paths:
- Input image: C:/Users/didri/Desktop/Full-Agent-Flow_VideoEditing/gc.png
- Output image: C:/Users/didri/Desktop/Full-Agent-Flow_VideoEditing/gc_lut_applied.png
- LUT file: C:/Users/didri/Desktop/Full-Agent-Flow_VideoEditing/Utils-Video_creation/LUT/Black & white cube/blackwhite1.cube
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image


def load_cube_lut(path):
    """
    Load a .cube LUT file and return it as a 3D numpy array.

    Args:
        path (str): Path to the .cube file

    Returns:
        np.ndarray: 3D LUT array with shape (size, size, size, 3)
    """
    lut_data = []
    size = None

    try:
        with open(path, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.upper().startswith("LUT_3D_SIZE"):
                    size = int(line.split()[1])
                elif line[0].isdigit() or line[0] == "-":
                    values = [float(v) for v in line.split()]
                    if len(values) == 3:  # RGB values
                        lut_data.append(values)
    except Exception as e:
        raise ValueError(f"Error reading .cube file: {e}")

    if size is None:
        raise ValueError("Invalid .cube file: missing LUT_3D_SIZE")

    if len(lut_data) != size * size * size:
        raise ValueError(f"Invalid .cube file: expected {size*size*size} data points, got {len(lut_data)}")

    lut = np.array(lut_data).reshape((size, size, size, 3))
    return lut


def apply_lut_to_image(image, lut):
    """
    Apply a 3D LUT to an image.

    Args:
        image (np.ndarray): Input image in RGB format (0-255)
        lut (np.ndarray): 3D LUT array

    Returns:
        np.ndarray: Image with LUT applied
    """
    # Normalize image to 0-1 range
    image_normalized = image.astype(np.float32) / 255.0

    # Get LUT size
    lut_size = lut.shape[0]

    # Scale normalized image values to LUT indices
    # We need to map [0,1] to [0, lut_size-1]
    scaled = image_normalized * (lut_size - 1)

    # Get integer indices and fractional parts for interpolation
    indices = np.floor(scaled).astype(np.int32)
    fractions = scaled - indices

    # Clamp indices to valid range
    indices = np.clip(indices, 0, lut_size - 1)
    indices_next = np.clip(indices + 1, 0, lut_size - 1)

    # Get the 8 corner points of the cube for trilinear interpolation
    def get_lut_value(r_idx, g_idx, b_idx):
        return lut[r_idx, g_idx, b_idx]

    # Extract individual channel indices and fractions
    r_idx, g_idx, b_idx = indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]
    r_idx_next, g_idx_next, b_idx_next = indices_next[:, :, 0], indices_next[:, :, 1], indices_next[:, :, 2]
    r_frac, g_frac, b_frac = fractions[:, :, 0], fractions[:, :, 1], fractions[:, :, 2]

    # Trilinear interpolation
    # Get 8 corner values
    c000 = get_lut_value(r_idx, g_idx, b_idx)
    c001 = get_lut_value(r_idx, g_idx, b_idx_next)
    c010 = get_lut_value(r_idx, g_idx_next, b_idx)
    c011 = get_lut_value(r_idx, g_idx_next, b_idx_next)
    c100 = get_lut_value(r_idx_next, g_idx, b_idx)
    c101 = get_lut_value(r_idx_next, g_idx, b_idx_next)
    c110 = get_lut_value(r_idx_next, g_idx_next, b_idx)
    c111 = get_lut_value(r_idx_next, g_idx_next, b_idx_next)

    # Expand fractions to match the RGB dimension
    r_frac = np.expand_dims(r_frac, axis=2)
    g_frac = np.expand_dims(g_frac, axis=2)
    b_frac = np.expand_dims(b_frac, axis=2)

    # Interpolate along each axis
    c00 = c000 * (1 - b_frac) + c001 * b_frac
    c01 = c010 * (1 - b_frac) + c011 * b_frac
    c10 = c100 * (1 - b_frac) + c101 * b_frac
    c11 = c110 * (1 - b_frac) + c111 * b_frac

    c0 = c00 * (1 - g_frac) + c01 * g_frac
    c1 = c10 * (1 - g_frac) + c11 * g_frac

    result = c0 * (1 - r_frac) + c1 * r_frac

    # Convert back to 0-255 range and ensure proper data type
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    return result


def apply_lut_to_frames(frames, lut_path=None):
    """
    Apply a .cube LUT to a list of RGB frames.

    Args:
        frames (list): List of numpy arrays representing RGB frames (0-255)
        lut_path (str, optional): Path to the .cube LUT file. If None, uses default black & white LUT.

    Returns:
        list: List of frames with LUT applied

    Example:
        # Load some frames (list of numpy arrays)
        processed_frames = apply_lut_to_frames(frame_list)

        # Or with custom LUT
        processed_frames = apply_lut_to_frames(frame_list, "path/to/custom.cube")
    """
    if lut_path is None:
        lut_path = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Utils-Video_creation\LUT\Black & white cube\blackwhite1.cube'

    # Load the LUT once
    try:
        lut = load_cube_lut(lut_path)
    except Exception as e:
        raise ValueError(f"Failed to load LUT from {lut_path}: {e}")

    processed_frames = []

    for i, frame in enumerate(frames):
        try:
            # Ensure frame is the right format (RGB, uint8)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            # Apply LUT to the frame
            processed_frame = apply_lut_to_image(frame, lut)
            processed_frames.append(processed_frame)
            print("applied lut to frame ", i)

        except Exception as e:
            print(f"Warning: Failed to process frame {i}: {e}")
            # Return original frame if processing fails
            processed_frames.append(frame)

    return processed_frames


def apply_lut_to_single_frame(frame, lut_path=None):
    """
    Apply a .cube LUT to a single RGB frame.

    Args:
        frame (np.ndarray): Single numpy array representing RGB frame (0-255)
        lut_path (str, optional): Path to the .cube LUT file. If None, uses default black & white LUT.

    Returns:
        np.ndarray: Frame with LUT applied

    Example:
        # Process a single frame
        processed_frame = apply_lut_to_single_frame(my_frame)
    """
    if lut_path is None:
        lut_path = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Utils-Video_creation\LUT\Black & white cube\blackwhite1.cube'

    # Load the LUT
    try:
        lut = load_cube_lut(lut_path)
    except Exception as e:
        raise ValueError(f"Failed to load LUT from {lut_path}: {e}")

    # Ensure frame is the right format (RGB, uint8)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Apply LUT to the frame
    return apply_lut_to_image(frame, lut)


def main():
    # Hardcoded paths
    input_image = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\gc.png'
    output_image = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Presetpro.png'
    lut_path = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Utils-Video_creation\LUT\Fuji Astia LUT\fuji.cube'

    # Check if input image exists
    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found.")
        sys.exit(1)

    # Check if LUT file exists
    if not os.path.exists(lut_path):
        print(f"Error: LUT file '{lut_path}' not found.")
        sys.exit(1)

    try:
        print(f"Loading image: {input_image}")
        # Load image using PIL and convert to RGB
        with Image.open(input_image) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                print(f"Converting image from {img.mode} to RGB")
                img = img.convert('RGB')

            # Convert to numpy array
            image = np.array(img)

        print(f"Image loaded successfully. Shape: {image.shape}")

        print(f"Loading LUT: {lut_path}")
        lut = load_cube_lut(lut_path)
        print(f"LUT loaded successfully. Size: {lut.shape[0]}³")

        print("Applying LUT to image...")
        result_image = apply_lut_to_image(image, lut)

        print(f"Saving result to: {output_image}")
        # Convert back to PIL Image and save
        result_pil = Image.fromarray(result_image, 'RGB')
        result_pil.save(output_image, 'PNG')

        print("✅ LUT applied successfully!")
        print(f"📁 Output saved as: {output_image}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
