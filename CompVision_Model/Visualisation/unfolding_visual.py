import os
import matplotlib.pyplot as plt
from CompVision_Model.wall_unfolding import UnfoldingEngine

# --- CONFIG ---
TEST_UNIT_PATH = "main/Data/DataOriginal/paviljoen/unit_0004"  # select any unit from Data folder


def check_channels():
    if not os.path.exists(TEST_UNIT_PATH):
        print(f"Error: {TEST_UNIT_PATH} not found.")
        return

    print(f"Processing {TEST_UNIT_PATH}...")
    engine = UnfoldingEngine(img_size=256)
    walls = engine.process_unit(TEST_UNIT_PATH, [])

    print(f"\nFound {len(walls)} walls (Coplanar Groups).")

    for i, wall in enumerate(walls):
        # Calculate dimensions in Meters
        w_m = (wall.u_range[1] - wall.u_range[0]) / 1000.0
        h_m = (wall.v_range[1] - wall.v_range[0]) / 1000.0

        print(f"Wall {i} | Size: {w_m:.2f}m x {h_m:.2f}m | Normal: {wall.normal}")

        # Visualize
        tensor = wall.image_tensor
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(tensor[0], cmap='viridis')
        axes[0].set_title(f"Wall {i} - U")

        axes[1].imshow(tensor[1], cmap='viridis')
        axes[1].set_title(f"Wall {i} - V")

        axes[2].imshow(tensor[2], cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f"Wall {i} - Mask\n(Black=Hole)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    check_channels()