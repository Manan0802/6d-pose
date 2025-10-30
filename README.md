# 6D Pose Estimation Project

This is an implementation of a 6D Pose Estimation model, designed to predict the 3D rotation (quaternion) and 3D translation (vector) of objects from RGB-D data.

This repository provides a streamlined pipeline for training, evaluating, and running inference with the model.

## Project Structure
6D-Pose-Quick/
│
├── 📄 README.md                # Project overview, setup guide, and usage instructions
│
├── ⚙️ configs/
│   └── config.yaml             # Configuration file (paths, hyperparameters, training settings)
│
├── 📦 data/
│   └── LINEMOD/                # Example dataset (organized by class or object)
│
├── 🧩 dataset/
│   └── linemod_dataset.py      # Dataset loader and preprocessing for LINEMOD
│
├── 🧠 models/
│   ├── mpt.py                  # Multi-Patch Transformer model definition
│   ├── bca.py                  # Bidirectional Cross-Attention module
│   └── pose_net.py             # Pose estimation network (core architecture)
│
├── 🧰 utils/
│   ├── metrics.py              # Evaluation metrics (ADD, ADD-S, reprojection error, etc.)
│   └── visualize.py            # Visualization tools for predictions and pose overlays
│
├── 💾 checkpoints/
│   └── (saved models)          # Folder for saving trained model checkpoints
│
├── 🚀 train.py                 # Script to train the pose estimation model
├── 🧪 evaluate.py              # Script to evaluate trained models on validation/test data
├── 🔍 inference.py             # Run inference on new images or videos for pose estimation
│
└── 📘 requirements.txt         # (optional) List of required dependencies (PyTorch, numpy, etc.)
## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourname/6D-Pose-Quick.git](https://github.com/yourname/6D-Pose-Quick.git)
    cd 6D-Pose-Quick
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # on Mac/Linux
    # venv\Scripts\activate   # on Windows
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Evaluation

To evaluate a pre-trained model, run the `evaluate.py` script with the path to a model checkpoint:

```bash
python evaluate.py checkpoints/model_epoch_400.pth

python inference.py checkpoints/model_epoch_400.pth \
                    data/LINEMOD/ape/rgb/0001.png \
                    data/LINEMOD/ape/depth/0001.png \
                    data/LINEMOD/ape/mask/0001.png

