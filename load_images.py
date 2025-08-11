#!/usr/bin/env python3
"""
Napari Experiment Helper: Load Your Own Images
This script helps you load and visualize your own image files
"""

import napari
import numpy as np
from skimage import io, data
import os
from pathlib import Path

def load_image_file(file_path):
    """Load an image file"""
    try:
        if file_path.lower().endswith(('.tif', '.tiff')):
            # Handle TIFF files (potentially multi-dimensional)
            image = io.imread(file_path)
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Handle standard image formats
            image = io.imread(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return None
        
        print(f"Loaded image: {file_path}")
        print(f"Shape: {image.shape}, Type: {image.dtype}")
        return image
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_sample_images():
    """Load sample images from scikit-image"""
    samples = {
        'coins': data.coins(),
        'camera': data.camera(),
        'astronaut': data.astronaut(),
        'binary_blobs': data.binary_blobs(length=256, n_dim=2),
        'checkerboard': data.checkerboard(),
        'cell': data.cell(),
    }
    return samples

def visualize_images(image_dict):
    """Create napari viewer with multiple images"""
    viewer = napari.Viewer(title="Image Viewer")
    
    for name, image in image_dict.items():
        if image is not None:
            if len(image.shape) == 2:  # Grayscale
                viewer.add_image(image, name=name, colormap='gray')
            elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                viewer.add_image(image, name=name)
            elif len(image.shape) == 3:  # 3D volume
                viewer.add_image(image, name=f"{name}_3D")
            else:
                print(f"Unsupported image shape for {name}: {image.shape}")
    
    napari.run()

def main():
    """Main function"""
    print("Napari Image Loader")
    print("=" * 30)
    
    choice = input("Load (1) sample images or (2) your own image file? [1/2]: ")
    
    if choice == "1":
        print("Loading sample images...")
        images = load_sample_images()
        visualize_images(images)
    
    elif choice == "2":
        file_path = input("Enter the path to your image file: ").strip()
        if os.path.exists(file_path):
            image = load_image_file(file_path)
            if image is not None:
                name = Path(file_path).stem
                visualize_images({name: image})
        else:
            print("File not found!")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
