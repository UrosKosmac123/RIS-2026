"""
Extract individual bacteria colony shapes from petri dish images.
Uses thresholding + contour detection to isolate each colony.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def extract_colonies(image_path, min_area=100, max_area=50000, visualize=False):
    """
    Extract individual bacteria colony shapes from a petri dish image.
    
    Args:
        image_path: Path to the PNG image
        min_area: Ignore contours smaller than this (noise)
        max_area: Ignore contours larger than this (petri dish edge)
        visualize: Show extraction process
    
    Returns:
        colonies: List of dicts with 'mask', 'bbox', 'centroid', 'area', 'image'
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Threshold to separate colonies from agar background
    # Colonies are usually brighter than the agar
    # Use adaptive thresholding for better results
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try Otsu's thresholding first (automatic)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 3: Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Close holes
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove small noise
    
    # Step 4: Find contours (individual colonies)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 5: Filter and extract each colony
    colonies = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter by area (remove noise and petri dish edge)
        if area < min_area or area > max_area:
            continue
        
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Create mask for this colony
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill contour
        
        # Get centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + bw//2, y + bh//2
        
        # Extract colony image (with mask applied)
        colony_img = img_rgb.copy()
        colony_img[mask == 0] = [0, 0, 0]  # Black background
        
        # Crop to bounding box
        colony_cropped = colony_img[y:y+bh, x:x+bw]
        mask_cropped = mask[y:y+bh, x:x+bw]
        
        colony_info = {
            'id': i,
            'mask': mask_cropped,
            'full_mask': mask,
            'bbox': (x, y, bw, bh),
            'centroid': (cx, cy),
            'area': area,
            'image': colony_cropped,
            'contour': contour
        }
        colonies.append(colony_info)
    
    # Sort by area (largest first)
    colonies.sort(key=lambda x: x['area'], reverse=True)
    
    if visualize:
        visualize_extraction(img_rgb, binary, colonies)
    
    return colonies


def visualize_extraction(original, binary, colonies):
    """Visualize the extraction process."""
    n_colonies = len(colonies)
    
    # Create figure with extraction steps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Binary mask
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binary Mask (Otsu)')
    axes[0, 1].axis('off')
    
    # Colonies highlighted
    highlighted = original.copy()
    for col in colonies:
        x, y, bw, bh = col['bbox']
        cv2.rectangle(highlighted, (x, y), (x+bw, y+bh), (255, 0, 0), 2)
        cv2.circle(highlighted, col['centroid'], 5, (0, 255, 0), -1)
    
    axes[0, 2].imshow(highlighted)
    axes[0, 2].set_title(f'Detected Colonies: {n_colonies}')
    axes[0, 2].axis('off')
    
    # Show a few example colonies
    for idx in range(min(3, n_colonies)):
        axes[1, idx].imshow(colonies[idx]['image'])
        axes[1, idx].set_title(f'Colony {idx+1}\nArea: {int(colonies[idx]["area"])}px')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('colony_extraction.png', dpi=150)
    plt.show()
    
    print(f"Found {n_colonies} colonies")
    print(f"Saved visualization to colony_extraction.png")


def save_colonies(colonies, output_dir='extracted_colonies'):
    """
    Save each extracted colony as a separate PNG.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for colony in colonies:
        # Save colony image
        filename = f"colony_{colony['id']:03d}_area{int(colony['area'])}.png"
        filepath = output_path / filename
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(colony['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), img_bgr)
        
        # Also save mask
        mask_filename = f"colony_{colony['id']:03d}_mask.png"
        mask_filepath = output_path / mask_filename
        cv2.imwrite(str(mask_filepath), colony['mask'])
    
    print(f"Saved {len(colonies)} colonies to {output_dir}/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Method 1: Contour-based extraction
        print("Extracting colonies using contour detection...")
        colonies = extract_colonies(image_path, min_area=200, max_area=50000)
        print(np.array(colonies[2]["image"]).shape)

        if colonies:
            #save_colonies(colonies, 'extracted_colonies')
            
            # Print info
            print("\nColony Information:")
            for col in colonies[:5]:  # Show first 5
                print(f"  Colony {col['id']}: Area={int(col['area'])}px, "
                      f"Center={col['centroid']}, BBox={col['bbox']}")
        else:
            print("No colonies found. Try adjusting min_area/max_area thresholds.")
    else:
        print("Not enough arguments, do:")
        print("Usage: python extract_colonies.py <image_path>")
