import os

# ALLOW DUPLICATE OPENMP LIBS
# This prevents the "OMP" error, common on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vision_model import UNet

# CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "CV_Dataset")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

BATCH_SIZE = 16
LR = 0.001
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WallDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.npy")))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .npy files found in {os.path.join(root_dir, 'images')}")

        # Load mask paths
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, "masks", "*.npy")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.image_paths[idx]
        image = np.load(img_path).astype(np.float32)

        # Load Mask
        mask_path = self.mask_paths[idx]
        mask = np.load(mask_path).astype(np.float32)

        # If mask is (256, 256), make it (1, 256, 256), so PyTorch doesn't crash
        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]

        return torch.from_numpy(image), torch.from_numpy(mask)


def save_debug_snapshot(model, loader, device, epoch, batch_idx):
    """Saves a visual snapshot of the model repeatedly as the model learns the walls. Looks for "interesting" samples with sockets in the batch to show."""
    model.eval()
    try:
        # Get one batch from the loader
        # get the *current* batch if possible, or just a fresh one
        data_iter = iter(loader)
        images, masks = next(data_iter)
        images, masks = images.to(device), masks.to(device)

        with torch.no_grad():
            output = model(images)
            pred = torch.sigmoid(output)

        # Look for positive sample...
        # Default to index 0 (if all are empty)
        target_idx = 0

        # Check if any mask in the batch has a socket (max value > 0)
        for i in range(images.shape[0]):
            if masks[i].max() > 0:
                target_idx = i
                break

        # Take the selected item
        img_np = images[target_idx].cpu().numpy()
        mask_np = masks[target_idx].cpu().numpy()
        pred_np = pred[target_idx].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        # Input (U channel)
        axes[0].imshow(img_np[0], cmap='viridis')
        axes[0].set_title(f"Input E{epoch}_B{batch_idx} (Idx {target_idx})")
        axes[0].axis('off')

        # Truth
        # Handle shape for plotting (squeeze the channel dimension)
        display_mask = mask_np[0] if mask_np.ndim == 3 else mask_np
        axes[1].imshow(display_mask, cmap='gray')
        axes[1].set_title("Truth (Should be white)")
        axes[1].axis('off')

        # Prediction
        im = axes[2].imshow(pred_np[0], cmap='jet', vmin=0, vmax=1)
        axes[2].set_title(f"Pred (Max: {pred_np[0].max():.2f})")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])

        save_path = os.path.join(RESULTS_DIR, "debug_current.png")
        plt.savefig(save_path)
        plt.close(fig)

    except Exception as e:
        print(f"Snapshot failed: {e}")
    finally:
        model.train()


def train():
    print(f"--- Starting Training on {DEVICE} ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Data
    full_dataset = WallDataset(DATASET_DIR)

    if len(full_dataset) == 0:
        print("CRITICAL: Dataset is empty.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Increase num_workers if you can
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    # 2. Model (3 Channels Input)
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)

    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (images, masks) in enumerate(loop):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            #SNAPSHOT EVERY 50 BATCHES
            if batch_idx % 50 == 0:
                save_debug_snapshot(model, train_loader, DEVICE, epoch + 1, batch_idx)

        # Validation
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Clear line to print clean stats
        loop.clear()
        print(f"Epoch {epoch + 1}: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best.pth"))
            print("  -> Saved Best Model")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"epoch_{epoch + 1}.pth"))

    print("Training Complete.")


if __name__ == "__main__":
    train()