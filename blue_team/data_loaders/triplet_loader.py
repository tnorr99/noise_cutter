import os
import random
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

class VideoTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_skip=30):
        """
        Args:
            root_dir (str): Path to the dataset (e.g., .../triplet_dataset/nflx)
            transform (callable, optional): PyTorch transforms to apply to the frames.
            frame_skip (int): Randomly sample a frame from the first X frames to ensure variety.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip
        
        # Find all valid video asset folders
        self.video_folders = [
            os.path.join(root_dir, d) for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ]

    def __len__(self):
        return len(self.video_folders)

    def _extract_frame(self, video_path, frame_idx=0):
        """Extracts a specific frame from an MP4 file."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Fallback in case of a corrupted frame
            return Image.new('RGB', (224, 224), color='black')
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def __getitem__(self, idx):
        # 1. Anchor/Positive from the same timestamp
        target_folder = self.video_folders[idx]
        anchor_path = os.path.join(target_folder, "original.mp4")
        positive_path = os.path.join(target_folder, "upscaled.mp4")
        
        target_frame = random.randint(0, self.frame_skip)
        
        # 2. Hard Negative: Same video, but a drastically different frame
        # Jump ahead by 100 frames (roughly 4 seconds at 25fps) to ensure the scene changed slightly
        negative_frame = target_frame + 100 
        negative_path = anchor_path # Use the original video for the negative

        # 3. Extract Images
        anchor_img = self._extract_frame(anchor_path, target_frame)
        positive_img = self._extract_frame(positive_path, target_frame)
        negative_img = self._extract_frame(negative_path, negative_frame)
        # 5. Apply PyTorch Transforms (Resizing, Normalization)
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img