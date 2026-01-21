import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "../CV_Dataset")

# ENTER THE ID OF THE UNIT YOU WANT TO INSPECT
TARGET_UNIT_ID = "gebouwE_unit_0001"

def inspect():
    print(f"Inspecting dataset at: {DATASET_DIR}")

    image_dir = os.path.join(DATASET_DIR, "images")
    mask_dir = os.path.join(DATASET_DIR, "masks")

    if not os.path.exists(image_dir):
        print(f"Error: Folder {image_dir} does not exist.")
        return

    # 1. GLOBAL STATISTICS
    mask_files = glob.glob(os.path.join(mask_dir, "*.npy"))
    total_files = len(mask_files)

    if total_files == 0:
        print("No samples found.")
        return

    print(f"Global Scan: Found {total_files} total samples. calculating stats...")

    non_empty_count = 0

    for mask_path in mask_files:
        mask = np.load(mask_path)
        if mask.max() > 0:
            non_empty_count += 1

    print(f"--------------------------------------------------")
    print(f"GLOBAL DATASET STATS")
    print(f"Total Walls: {total_files}")
    print(f"Walls with Sockets: {non_empty_count}")
    print(f"Percentage Positive: {(non_empty_count / total_files) * 100:.2f}%")
    print(f"--------------------------------------------------")

    if non_empty_count == 0:
        print("CRITICAL ERROR: Your dataset has ZERO targets.")
        return

    # 2. TARGETED INSPECTION
    print(f"\n--- INSPECTING UNIT: {TARGET_UNIT_ID} ---")

    search_pattern = os.path.join(image_dir, f"*{TARGET_UNIT_ID}*.npy")
    target_files = sorted(glob.glob(search_pattern))

    if not target_files:
        print(f"No walls found for unit '{TARGET_UNIT_ID}'. Check your spelling.")
        return

    print(f"Found {len(target_files)} walls for this unit. Visualizing sequentially...")

    for i, img_path in enumerate(target_files):
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"Warning: Missing mask for {filename}")
            continue

        # Load Data
        image = np.load(img_path)  # (3, 256, 256)
        label = np.load(mask_path)  # Could be (1, 256, 256) OR (256, 256)

        if label.ndim == 3:
            display_label = label[0]  # Take first channel
        else:
            display_label = label  # Already 2D

        # Check status
        has_socket = display_label.max() > 0
        status = "CONTAINS SOCKET" if has_socket else "Empty Wall"

        # PLOT
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 1. U-Channel
        axes[0].imshow(image[0], cmap='viridis')
        axes[0].set_title(f"Wall #{i}: U-Coord")
        axes[0].axis('off')

        # 2. V-Channel
        axes[1].imshow(image[1], cmap='viridis')
        axes[1].set_title("Input: V-Coord")
        axes[1].axis('off')

        # 3. Geometry Mask
        axes[2].imshow(image[2], cmap='gray')
        axes[2].set_title("Input: Wall Shape")
        axes[2].axis('off')

        # 4. TARGET (The Socket)
        # Overlay on geometry
        axes[3].imshow(image[2], cmap='gray', alpha=0.5)
        axes[3].imshow(display_label, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        axes[3].set_title(f"Truth: {status}")
        axes[3].axis('off')

        plt.suptitle(f"File: {filename}", fontsize=12)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    inspect()