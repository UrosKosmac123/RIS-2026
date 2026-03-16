"""
Crop petri dish images to remove left/right flat edges.
Keeps only the circular inner part.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def crop_petri_dish(image_path, crop_percent_left=0.15, crop_percent_right=0.15, padding=10, visualize=True):
    """
    Crop petri dish image to remove flat edges, keeping only circular part.
    
    Args:
        image_path: Path to image
        crop_percent_left: Percentage to crop from left side (0.15 = 15%)
        crop_percent_right: Percentage to crop from right side (0.15 = 15%)
        padding: Extra pixels to include around the crop
        visualize: Show before/after
    
    Returns:
        cropped_image, crop_coords
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Simple crop: remove percentage from left/right only (can be different)
    x1 = int(w * crop_percent_left) - padding
    x2 = int(w * (1 - crop_percent_right)) + padding
    y1 = 0  # Keep full height
    y2 = h  # Keep full height
    
    # Ensure coordinates are within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Perform crop
    cropped = img_rgb[y1:y2, x1:x2]
    
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Original with crop rectangle
        axes[0].imshow(img_rgb)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             linewidth=3, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title(f'Original ({w}x{h})')
        axes[0].axis('off')
        
        # Cropped
        axes[1].imshow(cropped)
        axes[1].set_title(f'Cropped ({cropped.shape[1]}x{cropped.shape[0]})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    crop_coords = (x1, y1, x2, y2)
    return cropped, crop_coords


if __name__ == "__main__":
    import sys
    
    # Default: process all PNGs from Podatki directory
    input_dir = "Podatki"
    output_dir = "Podatki_cropped"
    crop_left = 0.15
    crop_right = 0.15
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        crop_left = float(sys.argv[3])
    if len(sys.argv) > 4:
        crop_right = float(sys.argv[4])
    
    # Batch process all PNGs in input_dir
    print(f"Batch processing PNGs from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Crop: {crop_left*100}% from left, {crop_right*100}% from right")
    print("-" * 50)
    
    # Find all PNG files (including subdirectories like ris2026-krog1-ucni)
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' not found!")
        print(f"\nUsage:")
        print(f"  python crop_petri_dish.py")
        print(f"    (process all PNGs from 'Podatki' -> 'Podatki_cropped')")
        print(f"  python crop_petri_dish.py <input_dir> <output_dir> [crop_left] [crop_right]")
        sys.exit(1)
    
    image_paths = list(input_path.rglob('*.png'))  # Recursive search
    
    if not image_paths:
        print(f"No PNG files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} PNG files")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Process each image
    success_count = 0
    for img_path in image_paths:
        # Preserve subdirectory structure in output
        relative_path = img_path.relative_to(input_path)
        output_subdir = output_path / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {relative_path}...", end=' ')
        
        try:
            cropped, coords = crop_petri_dish(img_path, crop_percent_left=crop_left, crop_percent_right=crop_right, visualize=False)
            
            # Save to output directory
            output_file = output_subdir / img_path.name
            cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), cropped_bgr)
            
            print(f"✓ {cropped.shape[1]}x{cropped.shape[0]}")
            success_count += 1
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("-" * 50)
    print(f"Done! {success_count}/{len(image_paths)} images cropped.")
    print(f"Saved to: {output_dir}/")
