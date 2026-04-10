import os
import subprocess
import shutil
import glob
import time
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

# Use the SSD for fast read/writes of thousands of PNG frames
TEMP_DIR_ROOT = r"F:\realesrgan_temp" 

# --- Dataset Configuration ---
# Where your raw Kinetics-400 videos are currently stored
KINETICS_PATH = r"C:\path\to\raw\kinetics400" 

# Where the structured pairs will be saved for training
OUTPUT_DATASET_DIR = r"C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\triplet_dataset\kinetics"

def get_framerate(video_path):
    """Extracts the exact framerate to prevent audio desync during merging."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v",
        "-of", "default=noprint_wrappers=1:nokey=1",
        "-show_entries", "stream=r_frame_rate", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return "30"

def process_batch():
    """Iterates through all videos and processes them safely."""
    # Find all mp4 files in the source directory (including subdirectories)
    search_pattern = os.path.join(os.environ.get("KINETICS_PATH", "**"), "*.mp4")
    video_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(video_files)} videos to process.")
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    real_esrgan_exe = os.environ.get(
        "REAL_ESRGAN_PATH",
        r"C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\red_team\realesrgan\realesrgan-ncnn-vulkan.exe"
    )

    with tqdm(total=len(video_files), desc="Batch     ", unit="video",
              position=0, leave=True, dynamic_ncols=True) as batch_bar:
        for index, input_video in enumerate(video_files):
            # Create a unique, padded folder name for each video (e.g., video_00001)
            video_id = f"video_{index:05d}"
            target_folder = os.path.join(OUTPUT_DATASET_DIR, video_id)

            # Define final file paths
            final_original_path = os.path.join(target_folder, "original.mp4")
            final_upscaled_path = os.path.join(target_folder, "upscaled.mp4")

            # Skip if this video has already been processed (allows for resuming after a crash)
            if os.path.exists(final_upscaled_path):
                tqdm.write(f"Skipping {video_id} - Already processed.")
                batch_bar.update(1)
                continue

            batch_bar.set_postfix_str(os.path.basename(input_video))

            # Create output and temporary directories
            os.makedirs(target_folder, exist_ok=True)
            tmp_extract_dir = os.path.join(TEMP_DIR_ROOT, "extract")
            tmp_upscale_dir = os.path.join(TEMP_DIR_ROOT, "upscale")

            # Ensure temp dirs are empty before starting
            if os.path.exists(tmp_extract_dir): shutil.rmtree(tmp_extract_dir)
            if os.path.exists(tmp_upscale_dir): shutil.rmtree(tmp_upscale_dir)
            os.makedirs(tmp_extract_dir)
            os.makedirs(tmp_upscale_dir)

            try:
                # 1. Copy the original video to the new structured folder
                shutil.copy2(input_video, final_original_path)

                # 2. Extract frames to the SSD
                batch_bar.set_description("Extracting")
                extract_cmd = [
                    "ffmpeg", "-y", "-i", input_video,
                    "-qscale:v", "1", "-qmin", "1", "-qmax", "1",
                    "-vsync", "0", os.path.join(tmp_extract_dir, "frame_%08d.png")
                ]
                subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                # 3. Upscale — polls every 0.5s so the bar stays live
                batch_bar.set_description("Upscaling ")
                run_esrgan(real_esrgan_exe, tmp_extract_dir, tmp_upscale_dir,
                           batch_bar, os.path.basename(input_video))

                # 4. Merge back together
                batch_bar.set_description("Encoding  ")
                framerate = get_framerate(input_video)
                merge_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", framerate,
                    "-i", os.path.join(tmp_upscale_dir, "frame_%08d.png"),
                    "-i", input_video,
                    "-map", "0:v", "-map", "1:a?",
                    "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    final_upscaled_path
                ]
                subprocess.run(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            except subprocess.CalledProcessError:
                tqdm.write(f"Error processing {video_id}. Skipping to next video.")
                shutil.rmtree(target_folder, ignore_errors=True)

            finally:
                batch_bar.set_description("Batch     ")
                batch_bar.update(1)
                # Always clean up the SSD temp files, even if an error occurred
                shutil.rmtree(tmp_extract_dir, ignore_errors=True)
                shutil.rmtree(tmp_upscale_dir, ignore_errors=True)

    tqdm.write("\nBatch processing complete.")

if __name__ == "__main__":
    process_batch()