import os
import subprocess
import shutil
import glob
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

load_dotenv()

# SSD scratch space for PNG frame I/O
TEMP_DIR_ROOT = r"F:\realesrgan_temp"

# E: drive source — only process 1920x1080 variants for resolution consistency with NFLX
WATERLOO_ROOT = r"E:\Datasets\WaterlooIVC4K_HEVC\HEVC"
RESOLUTION_FILTER = "1920x1080"

# Output: one folder per source video with original.mp4 + upscaled.mp4
OUTPUT_DATASET_DIR = r"C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\data\triplet_dataset\waterloo"

REAL_ESRGAN_PATH = os.environ.get(
    "REAL_ESRGAN_PATH",
    r"C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\red_team\realesrgan\realesrgan-ncnn-vulkan.exe"
)


def get_framerate(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v",
        "-of", "default=noprint_wrappers=1:nokey=1",
        "-show_entries", "stream=r_frame_rate", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return "30"


def process_waterloo_batch():
    # Collect all 1920x1080 MP4s across every scene folder
    pattern = os.path.join(WATERLOO_ROOT, "*", f"{RESOLUTION_FILTER}_*.mp4")
    video_files = sorted(glob.glob(pattern))
    print(f"Found {len(video_files)} {RESOLUTION_FILTER} videos to process.")

    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    with tqdm(total=len(video_files), desc="Batch", unit="video",
              position=0, leave=True, dynamic_ncols=True) as batch_bar:
        for input_video in video_files:
            # Build a stable ID from folder number + filename, e.g. "01_1920x1080_1"
            scene_id = os.path.basename(os.path.dirname(input_video))
            base_name = os.path.splitext(os.path.basename(input_video))[0]
            video_id = f"{scene_id}_{base_name}"

            target_folder = os.path.join(OUTPUT_DATASET_DIR, video_id)
            final_original_path = os.path.join(target_folder, "original.mp4")
            final_upscaled_path = os.path.join(target_folder, "upscaled.mp4")

            if os.path.exists(final_upscaled_path):
                tqdm.write(f"Skipping {video_id} — already processed.")
                batch_bar.update(1)
                continue

            batch_bar.set_postfix_str(video_id)
            os.makedirs(target_folder, exist_ok=True)

            tmp_extract_dir = os.path.join(TEMP_DIR_ROOT, "extract")
            tmp_upscale_dir = os.path.join(TEMP_DIR_ROOT, "upscale")
            for d in [tmp_extract_dir, tmp_upscale_dir]:
                if os.path.exists(d):
                    shutil.rmtree(d)
                os.makedirs(d)

            try:
                # 1. Copy original to structured output
                shutil.copy2(input_video, final_original_path)

                # 2. Decode to PNG frames on SSD
                batch_bar.set_description("Extracting")
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_video,
                    "-qscale:v", "1", "-qmin", "1", "-qmax", "1", "-vsync", "0",
                    os.path.join(tmp_extract_dir, "frame_%08d.png")
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # 3. AI Upscale — polls every 0.5s so the bar stays live
                batch_bar.set_description("Upscaling ")
                run_esrgan(REAL_ESRGAN_PATH, tmp_extract_dir, tmp_upscale_dir,
                           batch_bar, video_id)

                # 4. Encode upscaled frames back to MP4
                batch_bar.set_description("Encoding  ")
                framerate = get_framerate(input_video)
                subprocess.run([
                    "ffmpeg", "-y",
                    "-framerate", framerate,
                    "-i", os.path.join(tmp_upscale_dir, "frame_%08d.png"),
                    "-i", input_video,
                    "-map", "0:v", "-map", "1:a?",
                    "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    final_upscaled_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            except subprocess.CalledProcessError:
                tqdm.write(f"Error processing {video_id}. Cleaning up and skipping.")
                shutil.rmtree(target_folder, ignore_errors=True)

            finally:
                batch_bar.set_description("Batch     ")
                batch_bar.update(1)
                shutil.rmtree(tmp_extract_dir, ignore_errors=True)
                shutil.rmtree(tmp_upscale_dir, ignore_errors=True)

    tqdm.write("\nWaterloo batch processing complete.")


if __name__ == "__main__":
    process_waterloo_batch()
