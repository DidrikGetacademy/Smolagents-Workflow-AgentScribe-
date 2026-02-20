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





if __name__ == "__main__":
    """
    Quick test script for enhance_frames_bpy function.
    Usage: Run this script from within Blender's Python console or with Blender Python executable.
    """
    try:
        import cv2

        # Test image paths to try (in order of preference)
        test_paths = [
            r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\short12_rife - frame at 0m0s.jpg",

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



        # Create output directory
        output_dir = './'
        os.makedirs(output_dir, exist_ok=True)

        # Save original for comparison
        original_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, 'original.png'), original_bgr)
        print(f"Saved original: {output_dir}/original.png")


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


        enchanced_frame, _, _ = Skin_texture_enchancement.process(original_bgr)
        print(f"After GPEN: shape={enchanced_frame.shape}, dtype={enchanced_frame.dtype}, range={enchanced_frame.min()}-{enchanced_frame.max()}")


     #   enchanced_frame_bgr = cv2.cvtColor(enchanced_frame, cv2.COLOR_BGR2RGB)


        output_path = os.path.join(output_dir, f'enhanced.png')
        cv2.imwrite(output_path, enchanced_frame)




        print(f"\nAll test images saved to: {output_dir}")
        print("Compare the results to see the enhancement effects!")

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Make sure OpenCV is installed: pip install opencv-python")
    except Exception as e:
        print(f'Test error: {e}')
        import traceback
        traceback.print_exc()
