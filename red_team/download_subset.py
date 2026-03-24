import os
import subprocess
import csv
import sys
# --- Configuration ---
# Path to the downloaded Kinetics-400 CSV file
CSV_PATH = os.environ.get("KINETICS_CSV_PATH", r"C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\kinetics400_train.csv")

# Where to save the raw downloaded mp4s (Matches the SOURCE_DATASET_DIR from the previous script)
OUTPUT_DIR = os.environ.get("KINETICS_PATH", r"C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\kinetics")

# Number of successful videos to download for the test batch
TARGET_COUNT = 50 

def download_kinetics_subset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    successful_downloads = 0

    print(f"Starting download of {TARGET_COUNT} videos...")

    with open(CSV_PATH, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if successful_downloads >= TARGET_COUNT:
                break

            video_id = row['youtube_id']
            # Kinetics provides start/end in seconds. We calculate duration for ffmpeg.
            start_time = int(row['time_start'])
            duration = int(row['time_end']) - start_time 
            
            url = f"https://www.youtube.com/watch?v={video_id}"
            output_filename = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

            # Skip if we already downloaded this one
            if os.path.exists(output_filename):
                continue

            print(f"Attempting to fetch video {video_id}...")

            # yt-dlp command mapping
            # --download-sections allows us to only download the 10-second clip, saving massive bandwidth
            cmd = ["yt-dlp",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4", # Force mp4 format
                "--download-sections", f"*{start_time}-{start_time + duration}",
                "--force-keyframes-at-cuts", # Ensures the cut is clean
                "--quiet", "--no-warnings",  # Keep the console clean
                "-o", output_filename,
                url
            ]

            try:
                # Run the command. YouTube videos frequently get deleted/privated over time,
                # so we expect many of these to fail. The script just moves to the next one.
                result = subprocess.run(cmd)
                
                if result.returncode == 0:
                    successful_downloads += 1
                    print(f"Success! ({successful_downloads}/{TARGET_COUNT} downloaded)\n")
                else:
                    print("Video unavailable or restricted. Skipping to next...\n")
                    
            except Exception as e:
                print(f"System error running yt-dlp: {e}. Skipping...\n")

    print(f"\nFinished. Downloaded {successful_downloads} clips to {OUTPUT_DIR}")

if __name__ == "__main__":
    download_kinetics_subset()