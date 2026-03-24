import os
import subprocess
import shutil
from dotenv import load_dotenv

# --- Metadata from NFLX_dataset_public.py ---
from NFLX_dataset_public import ref_videos, yuv_fmt, width, height

# --- Configuration ---
TEMP_DIR_ROOT = r"F:\realesrgan_temp" 
OUTPUT_DATASET_DIR = r"C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\data\triplet_dataset\NFLX\ref"

def get_fps_from_path(path):
    """Parses the filename to find the framerate (e.g., '25fps' or '30fps').
       File names in nflx ref end with _{fps}fps.yuv, so we can extract it directly from the name.
       Default to 30 if not found, which is common for many videos."""
    return path[path.rfind('_') + 1:path.rfind('fps')] if 'fps' in path else "30"
   

def process_nflx_batch(video_list):
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    for video in video_list:
        asset_name = os.path.basename(video['path']).replace('.yuv', '')
        fps = get_fps_from_path(video['path'])
        
        target_folder = os.path.join(OUTPUT_DATASET_DIR, asset_name)
        final_upscaled_path = os.path.join(target_folder, "upscaled.mp4")
        final_original_path = os.path.join(target_folder, "original.mp4")

        if os.path.exists(final_upscaled_path):
            continue

        print(f"\n--- Processing: {asset_name} ({fps} fps) ---")
        os.makedirs(target_folder, exist_ok=True)
        
        tmp_extract_dir = os.path.join(TEMP_DIR_ROOT, "extract")
        tmp_upscale_dir = os.path.join(TEMP_DIR_ROOT, "upscale")
        for d in [tmp_extract_dir, tmp_upscale_dir]:
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d)

        try:
            # 1. Convert YUV to Lossless MP4 (for the 'Original' reference)
            # This makes the Blue Team training much faster than reading raw YUV
            print("Creating lossless original MP4...")
            orig_cmd = [
                "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{width}x{height}", "-pix_fmt", yuv_fmt, "-r", fps,
                "-i", video['path'], "-c:v", "libx264", "-crf", "0", final_original_path
            ]
            subprocess.run(orig_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 2. Extract YUV directly to PNG on the SSD
            print("Extracting YUV frames to SSD...")
            extract_cmd = [
                "ffmpeg", "-y", "-f", "rawvideo", "-s", f"{width}x{height}",
                "-pix_fmt", yuv_fmt, "-r", fps, "-i", video['path'],
                "-qscale:v", "1", os.path.join(tmp_extract_dir, "frame_%08d.png")
            ]
            subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 3. Upscale (RTX 2060 Super)
            print("Upscaling...")
            upscale_cmd = [
                os.getenv("REAL_ESRGAN_PATH"), "-i", tmp_extract_dir, "-o", tmp_upscale_dir,
                "-n", "realesrgan-x4plus", "-g", "0", "-f", "png"
            ]
            subprocess.run(upscale_cmd, check=True)

            # 4. Merge to Upscaled MP4
            print("Merging to upscaled MP4...")
            merge_cmd = [
                "ffmpeg", "-y", "-framerate", fps,
                "-i", os.path.join(tmp_upscale_dir, "frame_%08d.png"),
                "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
                final_upscaled_path
            ]
            subprocess.run(merge_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        finally:
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
            shutil.rmtree(tmp_upscale_dir, ignore_errors=True)

if __name__ == "__main__":
    # You can import your ref_videos list here
    load_dotenv()
    print(os.environ.get("REAL_ESRGAN_PATH"))
    process_nflx_batch(ref_videos)
