import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import random

# Ensure we can import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from CompVision_Model.vision_model import UNet

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "../CV_Dataset")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def visualize_random_samples(model, num_samples=5):
    print(f"Loading file list...")

    image_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "images", "*.npy")))
    mask_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "masks", "*.npy")))

    if len(image_paths) == 0:
        print("No images found.")
        return

    #  RANDOM SELECTION
    # We pick 'num_samples' random indices from the total list
    total_files = len(image_paths)
    indices = random.sample(range(total_files), min(num_samples, total_files))

    print(f"Showing {len(indices)} completely random walls (Positive or Negative)...")

    for idx in indices:
        # Load raw data
        img_raw = np.load(image_paths[idx])  # (3, 256, 256)
        mask_raw = np.load(mask_paths[idx])  # Could be (256, 256) OR (1, 256, 256)

        # Prepare for Model (Add Batch Dimension)
        # Tensor Shape: (1, 3, 256, 256)
        input_tensor = torch.from_numpy(img_raw).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]  # Extract result: (256, 256)

        # FIX: HANDLE MASK SHAPE
        # If mask is 3D (1, 256, 256), take the first slice [0].
        # If mask is 2D (256, 256), use it directly.
        if mask_raw.ndim == 3:
            display_mask = mask_raw[0]
        else:
            display_mask = mask_raw

        # Calculate Status
        has_socket = display_mask.max() > 0
        pred_max = pred.max()
        status = "SOCKET" if has_socket else "BLANK"

        # PLOT
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 1. Input (U-Channel)
        axes[0].imshow(img_raw[0], cmap='viridis')
        axes[0].set_title("Input (U-Coord)")
        axes[0].axis('off')

        # 2. Input (Geometry Mask)
        axes[1].imshow(img_raw[2], cmap='gray')
        axes[1].set_title("Input (Wall Shape)")
        axes[1].axis('off')

        # 3. Ground Truth
        axes[2].imshow(display_mask, cmap='gray')
        axes[2].set_title(f"Truth: {status}")
        axes[2].axis('off')

        # 4. Prediction
        im = axes[3].imshow(pred, cmap='jet', vmin=0, vmax=1)
        axes[3].set_title(f"Pred (Max: {pred_max:.2f})")
        axes[3].axis('off')

        plt.colorbar(im, ax=axes[3])
        plt.suptitle(f"Sample #{idx}", fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
    else:
        # Load Model
        model = UNet(n_channels=3, n_classes=1).to(DEVICE)

        # Load weights safely
        try:
            # Try new safe loading (PyTorch 2.4+)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        except TypeError:
            # Fallback for older PyTorch
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

        model.eval()

        # Show 10 random samples
        visualize_random_samples(model, num_samples=10)