Project: Noise Cutter
AI-Enhanced Video Similarity & Content ID Bypass Detection
Project Goal: To evaluate and implement a deep learning system capable of identifying copyrighted video content (TV/Film) that has been altered using AI Upscaling (e.g., Real-ESRGAN) to bypass traditional Content ID fingerprinting.

📂 Project Structure
Plaintext
noise_cutter/
├── data/
│   ├── raw/                 # Ground Truth (Original) YUV/MP4 files
│   └── triplet_dataset/     # Processed pairs for training
│       ├── kinetics/        # YouTube-sourced UGC (Baseline)
│       └── nflx/            # Cinematic YUV-sourced (Primary)
├── red_team/                # Data Generation & Augmentation
│   ├── batch_upscale.py     # YUV to Lossless MP4 + AI Upscaling
│   └── realesrgan-ncnn/     # Vulkan-accelerated upscaling engine
└── blue_team/               # Detection & Embedding Research
    ├── models/              # ResNet18 Feature Extractor
    ├── data_loaders/        # VideoTripletDataset (Patch-based)
    └── training/            # Triplet Loss training scripts
🔴 Red Team: Data Generation
The Red Team creates the "fakes." To simulate high-end copyright bypass, we use professional-grade datasets and upscaling tools.

1. Data Sources
NFLX Public Dataset: High-quality 1080p YUV sequences.

SJTU 8K 360: Immersive, high-resolution textures for extreme upscaling tests.

Kinetics-400: General action clips used for initial pipeline validation.

2. Processing Pipeline
The batch_upscale.py script implements a Tiered Storage Strategy:

Read: From HDD (Mass storage of YUV files).

Process: On SSD (F:\realesrgan_temp) to handle high-frequency PNG frame I/O.

Upscale: Powered by Real-ESRGAN-ncnn-vulkan utilizing the RTX 2060 Super.

🔵 Blue Team: Detection Engine
The Blue Team builds the "Fingerprinter." Unlike standard models, this is trained to be sensitive to pixel-level AI hallucinations.

1. Model Architecture
Backbone: ResNet18 (trained from scratch/non-pretrained).

Embedding Size: 512-dimensional vector.

Normalization: L2 Normalization is applied to all outputs to map features to a unit hypersphere, ensuring compatibility with FAISS (Cosine Similarity).

2. Training Strategy: Patch-Based Triplet Loss
To focus the model on "AI Noise" rather than "Scene Geometry," we utilize:

Random Subsegment Cropping: Instead of resizing, the model looks at random 224x224 patches at native resolution.

Hard Temporal Negatives: The model compares a frame to its AI-upscaled version (Positive) and a different timestamp from the same video (Negative).

Loss Function: TripletMarginLoss with a shrunk margin (0.2) to accommodate the normalized vector space.

🚀 How to Run
Environment Setup
Requires Conda for managing binary dependencies (FFmpeg, CUDA).

Bash
conda activate noise_cutter
# Ensure FFmpeg is in your PATH
1. Generate Training Data
Place .yuv files in data/raw/nflx/ and run the upscale tool:

Bash
python red_team/batch_upscale.py
2. Train the Model
The model will learn to distinguish the original "Anchor" from the "Upscaled Positive."

Bash
python blue_team/training/train_triplet.py
📈 Current Status & Observations
Loss Stability: Implementation of L2 normalization and a smaller margin has stabilized the Triplet Loss, preventing "Exploding Gradients."

Data Quality: Moving from Kinetics (UGC) to NFLX/SJTU (Cinematic/8K) has improved the model's ability to scrutinize professional film textures.

Bottleneck: Currently limited by small batch counts; expansion to Waterloo 4K or Vimeo-90K datasets is planned to improve generalization.

🛠 Prerequisites
Hardware: NVIDIA RTX 2060 Super (or better).

Tools: Real-ESRGAN (NCNN Vulkan version), FFmpeg.

Libraries: PyTorch, TorchVision, OpenCV, NumPy.