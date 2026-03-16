"""
Multiple ways to visualize and inspect the border_mask matrix.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def inspect_border_mask(image_path, threshold=30):
    """
    Show border_mask in multiple ways for inspection.
    """
    from border_detection import detect_borders_optimized
    
    # Get the border mask
    border_mask, gradient_magnitude = detect_borders_optimized(image_path, threshold)
    
    print("=" * 60)
    print("BORDER_MASK MATRIX INSPECTION")
    print("=" * 60)
    
    # 1. Basic info
    print(f"\n1. MATRIX PROPERTIES:")
    print(f"   Shape: {border_mask.shape}")
    print(f"   Data type: {border_mask.dtype}")
    print(f"   Size: {border_mask.size} pixels ({border_mask.size / 1e6:.2f} MP)")
    
    # 2. Unique values
    unique_values = np.unique(border_mask)
    print(f"\n2. UNIQUE VALUES: {unique_values}")
    print(f"   (0 = background, 255 = border)")
    
    # 3. Statistics
    border_pixels = np.sum(border_mask == 255)
    background_pixels = np.sum(border_mask == 0)
    total = border_mask.size
    
    print(f"\n3. PIXEL COUNTS:")
    print(f"   Border pixels (255): {border_pixels:,} ({100*border_pixels/total:.1f}%)")
    print(f"   Background pixels (0): {background_pixels:,} ({100*background_pixels/total:.1f}%)")
    
    # 4. Print a small sample section (e.g., top-left 10x10)
    sample_size = min(20, border_mask.shape[0], border_mask.shape[1])
    print(f"\n4. SAMPLE SECTION (top-left {sample_size}x{sample_size}):")
    sample = border_mask[:sample_size, :sample_size]
    print(sample)
    
    # 5. Find border pixel coordinates
    border_coords = np.argwhere(border_mask == 255)
    print(f"\n5. BORDER COORDINATES:")
    print(f"   Total border pixels: {len(border_coords)}")
    if len(border_coords) > 0:
        print(f"   First 5 border pixel locations (row, col):")
        for i, coord in enumerate(border_coords[:5]):
            print(f"     {i+1}. ({coord[0]}, {coord[1]})")
    
    # 6. Visualizations
    print(f"\n6. GENERATING VISUALIZATIONS...")
    
    # Load original for overlay
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Border mask as grayscale
    axes[0, 1].imshow(border_mask, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Border Mask (0 or 255)')
    axes[0, 1].axis('off')
    
    # Border mask as red overlay on original
    overlay = img.copy()
    overlay[border_mask == 255] = [255, 0, 0]  # Red
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Borders in RED on Original')
    axes[0, 2].axis('off')
    
    # Border pixels only (black background)
    border_only = np.zeros_like(img)
    border_only[border_mask == 255] = [255, 255, 255]  # White borders
    axes[1, 0].imshow(border_only)
    axes[1, 0].set_title('Border Pixels Only')
    axes[1, 0].axis('off')
    
    # Gradient magnitude (raw values)
    axes[1, 1].imshow(gradient_magnitude, cmap='hot')
    axes[1, 1].set_title('Gradient Magnitude (raw values)')
    axes[1, 1].axis('off')
    
    # Histogram of gradient values
    axes[1, 2].hist(gradient_magnitude.flatten(), bins=50, color='blue', alpha=0.7)
    axes[1, 2].axvline(threshold, color='red', linestyle='--', label=f'threshold={threshold}')
    axes[1, 2].set_title('Gradient Value Distribution')
    axes[1, 2].set_xlabel('Gradient Magnitude')
    axes[1, 2].set_ylabel('Pixel Count')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    output_file = f'border_inspection_{Path(image_path).stem}.png'
    plt.savefig(output_file, dpi=150)
    plt.show()
    
    print(f"   Saved visualization to: {output_file}")
    print("=" * 60)
    
    return border_mask


def print_matrix_section(border_mask, row_start=0, col_start=0, size=10):
    """
    Print a specific section of the border_mask matrix.
    """
    section = border_mask[row_start:row_start+size, col_start:col_start+size]
    
    print(f"\nMatrix section [{row_start}:{row_start+size}, {col_start}:{col_start+size}]:")
    print("     ", end="")
    for c in range(size):
        print(f"{col_start+c:4}", end="")
    print()
    print("    " + "-" * (size * 4 + 1))
    
    for i, row in enumerate(section):
        print(f"{row_start+i:3} |", end="")
        for val in row:
            marker = "###" if val == 255 else "   ."
            print(f"{marker}", end="")
        print(" |")
    
    print("    " + "-" * (size * 4 + 1))
    print("    ### = border pixel (255),  .  = background (0)")


def save_border_coordinates(border_mask, output_file='border_coords.txt'):
    """
    Save all border pixel coordinates to a text file for inspection.
    """
    border_coords = np.argwhere(border_mask == 255)
    
    with open(output_file, 'w') as f:
        f.write(f"Border Pixel Coordinates\n")
        f.write(f"Total: {len(border_coords)} pixels\n")
        f.write(f"Format: (row, column)\n")
        f.write("=" * 30 + "\n\n")
        
        for i, (row, col) in enumerate(border_coords):
            f.write(f"{i+1:6d}: ({row:4d}, {col:4d})\n")
    
    print(f"Saved {len(border_coords)} border coordinates to: {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        
        # Main inspection
        mask = inspect_border_mask(image_path, threshold)
        
        # Print a specific section
        print("\n")
        print_matrix_section(mask, row_start=100, col_start=100, size=15)
        
        # Save coordinates (optional - can be large file!)
        # save_border_coordinates(mask, f'border_coords_{Path(image_path).stem}.txt')
        
    else:
        print("Usage: python visualize_border_mask.py <image_path> [threshold]")
        print("\nThis script shows you:")
        print("  - Matrix shape and data type")
        print("  - How many 0s and 255s")
        print("  - A printed sample of the matrix")
        print("  - Visual plots of the border mask")
        print("  - Border pixel locations")
