"""
Bacteria Colony Border Detection using Gradient-Based Approach

Your idea implemented with vectorized NumPy for speed:
1. Convert to grayscale (sum color channels)
2. Compute differences between consecutive pixels (gradients)
3. Threshold to find borders
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def detect_borders(image_path, threshold=30, sigma=1.0, visualize=True):
    """
    Detect bacteria colony borders using gradient magnitude.
    
    Args:
        image_path: Path to the image
        threshold: Gradient magnitude threshold for border detection
        sigma: Gaussian blur sigma (0 to skip, helps reduce noise)
        visualize: Show the results
    
    Returns:
        border_mask: Binary mask of detected borders
        gradient_magnitude: Full gradient magnitude map
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 1: Convert to "grayscale" by summing all color levels
    # This is exactly what you suggested - sum all color channels
    gray = np.sum(img, axis=2).astype(np.float32)
    
    # Optional: Gaussian blur to reduce noise (helps a lot on noisy images)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Step 2: Compute gradients (differences between consecutive entries)
    # Vectorized and FAST using NumPy's diff
    
    # Horizontal gradient: differences between columns (left-right)
    grad_x = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    
    # Vertical gradient: differences between rows (up-down)  
    grad_y = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    
    # Combined gradient magnitude (like Sobel operator)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Step 3: Threshold to find borders
    # Large difference = border, small difference = not border
    border_mask = gradient_magnitude > threshold
    
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title(f'Grayscale (Sum of Channels)')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(gradient_magnitude, cmap='hot')
        axes[1, 0].set_title(f'Gradient Magnitude')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(border_mask, cmap='gray')
        axes[1, 1].set_title(f'Borders (threshold={threshold})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'border_result_{Path(image_path).stem}.png', dpi=150)
        plt.show()
        print(f"Saved visualization to border_result_{Path(image_path).stem}.png")
    
    return border_mask, gradient_magnitude


def detect_borders_optimized(image_path, threshold=30):
    """
    Even faster version using OpenCV's built-in Sobel operator.
    Same mathematical result, highly optimized in C++.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale (OpenCV uses weighted sum by default for better perception)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel operator = your approach but with [-1, 0, 1] kernels
    # Computes gradient in x and y directions
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # dx
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # dy
    
    # Gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255 for display
    grad_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold for borders
    _, border_mask = cv2.threshold(grad_norm, threshold, 255, cv2.THRESH_BINARY)
    
    return border_mask, gradient_magnitude


def batch_process(image_dir, output_dir='borders_output'):
    """
    Process all images in a directory.
    """
    import glob
    
    image_paths = list(Path(image_dir).glob('*.jpg')) + \
                  list(Path(image_dir).glob('*.jpeg')) + \
                  list(Path(image_dir).glob('*.png'))
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Found {len(image_paths)} images")
    
    for img_path in image_paths:
        print(f"Processing {img_path.name}...", end=' ')
        
        try:
            border_mask, _ = detect_borders(img_path, threshold=30, visualize=False)
            
            # Save border mask
            output_file = output_path / f"{img_path.stem}_border.png"
            cv2.imwrite(str(output_file), border_mask.astype(np.uint8) * 255)
            
            print(f"✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        detect_borders(image_path)
    else:
        print("Usage: python border_detection.py <image_path>")
        print("\nOr to process a directory:")
        print("  from border_detection import batch_process")
        print("  batch_process('path/to/images')")
        print("\nYour approach is essentially a Sobel edge detector!")
        print("It's FAST because NumPy operations are vectorized in C.")
