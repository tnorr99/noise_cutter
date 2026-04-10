import os
import subprocess
import shutil
import time
from dotenv import load_dotenv
from tqdm import tqdm


def run_esrgan(exe, input_dir, output_dir, bar, label):
    """Run realesrgan-ncnn-vulkan and track progress by counting output PNGs.
    Avoids reading stdout/stderr entirely — ESRGAN's self-reported % freezes at ~98%
    while the process flushes GPU buffers and writes remaining frames to disk."""
    total = sum(1 for f in os.listdir(input_dir) if f.endswith('.png'))

    proc = subprocess.Popen(
        [exe, "-i", input_dir, "-o", output_dir,
         "-n", "realesrgan-x4plus", "-g", "0", "-f", "png"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    start = time.time()
    while proc.poll() is None:
        done = sum(1 for f in os.listdir(output_dir) if f.endswith('.png'))
        pct = done / total * 100 if total else 0
        elapsed = int(time.time() - start)
        bar.set_postfix_str(f"{label} | ESRGAN {pct:.0f}% | {elapsed}s")
        time.sleep(0.5)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, exe)

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

    real_esrgan_exe = os.getenv("REAL_ESRGAN_PATH")

    with tqdm(total=len(video_list), desc="Batch     ", unit="video",
              position=0, leave=True, dynamic_ncols=True) as batch_bar:
        for video in video_list:
            asset_name = os.path.basename(video['path']).replace('.yuv', '')
            fps = get_fps_from_path(video['path'])

            target_folder = os.path.join(OUTPUT_DATASET_DIR, asset_name)
            final_upscaled_path = os.path.join(target_folder, "upscaled.mp4")
            final_original_path = os.path.join(target_folder, "original.mp4")

            if os.path.exists(final_upscaled_path):
                batch_bar.update(1)
                continue

            batch_bar.set_postfix_str(f"{asset_name} ({fps} fps)")
            os.makedirs(target_folder, exist_ok=True)

            tmp_extract_dir = os.path.join(TEMP_DIR_ROOT, "extract")
            tmp_upscale_dir = os.path.join(TEMP_DIR_ROOT, "upscale")
            for d in [tmp_extract_dir, tmp_upscale_dir]:
                if os.path.exists(d): shutil.rmtree(d)
                os.makedirs(d)

            try:
                # 1. Convert YUV to Lossless MP4 (for the 'Original' reference)
                batch_bar.set_description("Orig MP4  ")
                orig_cmd = [
                    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}", "-pix_fmt", yuv_fmt, "-r", fps,
                    "-i", video['path'], "-c:v", "libx264", "-crf", "0", final_original_path
                ]
                subprocess.run(orig_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # 2. Extract YUV directly to PNG on the SSD
                batch_bar.set_description("Extracting")
                extract_cmd = [
                    "ffmpeg", "-y", "-f", "rawvideo", "-s", f"{width}x{height}",
                    "-pix_fmt", yuv_fmt, "-r", fps, "-i", video['path'],
                    "-qscale:v", "1", os.path.join(tmp_extract_dir, "frame_%08d.png")
                ]
                subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # 3. Upscale — polls every 0.5s so the bar stays live
                batch_bar.set_description("Upscaling ")
                run_esrgan(real_esrgan_exe, tmp_extract_dir, tmp_upscale_dir,
                           batch_bar, asset_name)

                # 4. Merge to Upscaled MP4
                batch_bar.set_description("Encoding  ")
                merge_cmd = [
                    "ffmpeg", "-y", "-framerate", fps,
                    "-i", os.path.join(tmp_upscale_dir, "frame_%08d.png"),
                    "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
                    final_upscaled_path
                ]
                subprocess.run(merge_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            finally:
                batch_bar.set_description("Batch     ")
                batch_bar.update(1)
                shutil.rmtree(tmp_extract_dir, ignore_errors=True)
                shutil.rmtree(tmp_upscale_dir, ignore_errors=True)

if __name__ == "__main__":
    # You can import your ref_videos list here
    load_dotenv()
    print(os.environ.get("REAL_ESRGAN_PATH"))
    process_nflx_batch(ref_videos)
