import cv2
import imagehash
from PIL import Image
import numpy as np

def get_video_hashes(video_path, frame_skip=30):
    """
    Extracts frames from a video and generates a perceptual hash for each.
    frame_skip: Number of frames to skip between hashes (e.g., 30 = ~1 hash per second at 30fps)
    """
    cap = cv2.VideoCapture(video_path)
    hashes = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            # Convert OpenCV BGR format to RGB, then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Compute perceptual hash
            h = imagehash.phash(pil_image)
            hashes.append(h)
            
        frame_count += 1

    cap.release()
    return hashes

def compare_videos(source_hashes, target_hashes, threshold=5):
    """
    Compares two lists of hashes and returns a similarity score.
    threshold: Maximum Hamming distance to consider two frames a "match".
    """
    matches = 0
    # Simple frame-by-frame comparison (in a real system, you'd use dynamic time warping 
    # or sequence alignment to account for cropped/shifted timelines)
    for h1 in source_hashes:
        for h2 in target_hashes:
            # Calculate Hamming distance
            if h1 - h2 <= threshold:
                matches += 1
                break # Move to next source frame once a match is found
                
    if len(source_hashes) == 0:
        return 0
        
    similarity_percentage = (matches / len(source_hashes)) * 100
    return similarity_percentage

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Index your "Dataset" (Original Video)
    print("Hashing original video...")
    original_hashes = get_video_hashes("original_copyright_video.mp4")

    # 2. Analyze the "Suspect" Video (Altered/Upscaled)
    print("Hashing suspect video...")
    suspect_hashes = get_video_hashes("suspect_upscaled_video.mp4")

    # 3. Compare
    similarity = compare_videos(suspect_hashes, original_hashes)
    print(f"Similarity Score: {similarity:.2f}%")
    
    if similarity > 70:
        print("Result: Likely Copyright Match.")
    else:
        print("Result: No strong match detected.")