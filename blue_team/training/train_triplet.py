import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

# Assuming your loader and model are saved as discussed in the project structure
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from blue_team.data_loaders.triplet_loader import VideoTripletDataset
from blue_team.models.DML import VideoFeatureExtractor

# --- Configuration ---
# All dataset roots to train on — add new upscaled datasets here as they become available
DATASET_DIRS = [
    r"C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\data\triplet_dataset\NFLX\ref",
    r"C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\data\triplet_dataset\waterloo",
]
CHECKPOINT_DIR = r"C:\Users\tnorr\OneDrive\Documents\AI\noise_cutter\blue_team\training\checkpoints"
RESUME_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "model_best.pth")  # set to "" to train from scratch

BATCH_SIZE = 16  # Adjust based on your RTX 2060 Super's VRAM limits
EPOCHS = 40      # Total epochs including any resumed run
LEARNING_RATE = 0.0001

def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """Saves the model's state dictionary to the disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "model_best.pth")
        torch.save(state, best_path)

def train_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing training on: {device}")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoTripletDataset(root_dirs=DATASET_DIRS, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 3. Initialize Model, Loss, and Optimizer
    model = VideoFeatureExtractor(pretrained=False).to(device)
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.2, p=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    best_loss = float('inf')

    # Resume from checkpoint if available
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"Resuming from checkpoint: {RESUME_CHECKPOINT}")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"  Resumed at epoch {start_epoch}, best loss so far: {best_loss:.4f}")
    else:
        print("Starting fresh training run.")

    model.train()

    # 4. The Training Loop
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        epoch_loss = 0.0

        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            # Move data to the GPU
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Extract embeddings
            embed_anchor = model.features(anchor).view(anchor.size(0), -1)
            embed_positive = model.features(positive).view(positive.size(0), -1)
            embed_negative = model.features(negative).view(negative.size(0), -1)

            # L2 Normalize the vectors ---
            embed_anchor = F.normalize(embed_anchor, p=2, dim=1)
            embed_positive = F.normalize(embed_positive, p=2, dim=1)
            embed_negative = F.normalize(embed_negative, p=2, dim=1)

            # Calculate Loss
            loss = triplet_loss_fn(embed_anchor, embed_positive, embed_negative)

            # Backward pass: Calculate gradients and update weights
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print status every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)} | Current Loss: {loss.item():.4f}")

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_epoch_loss:.4f}")

        # 5. Save the Checkpoint
        is_best = avg_epoch_loss < best_loss
        if is_best:
            best_loss = avg_epoch_loss
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(), # Saving optimizer state allows resuming exactly where you left off
        }, is_best, filename=f"checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_model()