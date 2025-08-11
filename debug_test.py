#!/usr/bin/env python3
"""
Debug script to test image loading and basic functionality
"""

import numpy as np
import os
from pathlib import Path

def test_basic_imports():
    """Test if all imports work"""
    try:
        print("Testing imports...")
        import napari
        print(f"  ✓ Napari version: {napari.__version__}")
        
        from skimage import data, filters, segmentation, measure, io, color
        print("  ✓ scikit-image imported successfully")
        
        from scipy import ndimage
        print("  ✓ scipy imported successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_image_loading():
    """Test loading your images"""
    print("\nTesting image loading...")
    
    workspace_path = "/home/ashu_17/Documents/Napari"
    image_files = [
        "Spoon.jpeg",
        "Umbrella.jpeg", 
        "Book.jpeg",
        "Key.jpeg",
        "download.jpeg"
    ]
    
    loaded_count = 0
    
    for filename in image_files:
        filepath = os.path.join(workspace_path, filename)
        print(f"  Checking {filename}...")
        
        if os.path.exists(filepath):
            try:
                from skimage import io
                image = io.imread(filepath)
                print(f"    ✓ Loaded successfully: {image.shape}, dtype: {image.dtype}")
                loaded_count += 1
            except Exception as e:
                print(f"    ✗ Error loading: {e}")
        else:
            print(f"    ✗ File not found: {filepath}")
    
    print(f"\nSuccessfully loaded {loaded_count}/{len(image_files)} images")
    return loaded_count > 0

def test_basic_processing():
    """Test basic image processing"""
    print("\nTesting basic image processing...")
    
    try:
        from skimage import data, filters
        import numpy as np
        
        # Use a sample image
        test_image = data.coins()
        print(f"  ✓ Sample image created: {test_image.shape}")
        
        # Test basic filters
        gaussian = filters.gaussian(test_image, sigma=1)
        print(f"  ✓ Gaussian filter applied: {gaussian.shape}")
        
        sobel = filters.sobel(test_image)
        print(f"  ✓ Sobel filter applied: {sobel.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Processing error: {e}")
        return False

def test_napari_headless():
    """Test if Napari can create a viewer in headless mode"""
    print("\nTesting Napari viewer creation...")
    
    try:
        import napari
        from skimage import data
        
        # Try to create a viewer without showing it
        print("  Creating viewer (headless)...")
        viewer = napari.Viewer(show=False)
        
        # Add a simple image
        test_image = data.coins()
        viewer.add_image(test_image, name="test")
        print(f"  ✓ Viewer created with {len(viewer.layers)} layer(s)")
        
        # Close the viewer
        viewer.close()
        print("  ✓ Viewer closed successfully")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Napari error: {e}")
        return False

def main():
    """Run all tests"""
    print("Napari Installation and Image Loading Debug")
    print("=" * 50)
    
    # Test 1: Imports
    imports_ok = test_basic_imports()
    
    # Test 2: Image loading
    images_ok = test_image_loading()
    
    # Test 3: Basic processing
    processing_ok = test_basic_processing()
    
    # Test 4: Napari headless
    napari_ok = test_napari_headless()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Imports: {'✓' if imports_ok else '✗'}")
    print(f"  Image Loading: {'✓' if images_ok else '✗'}")
    print(f"  Image Processing: {'✓' if processing_ok else '✗'}")
    print(f"  Napari Viewer: {'✓' if napari_ok else '✗'}")
    
    if all([imports_ok, images_ok, processing_ok, napari_ok]):
        print("\n🎉 All tests passed! Your setup should work.")
        print("\nTry running the experiment again:")
        print("python experiment-1.py")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        
        if not images_ok:
            print("\nImage loading issues detected. Please check:")
            print("- Are your image files in the correct location?")
            print("- Are the file names spelled correctly?")
            print("- Are the files readable?")

if __name__ == "__main__":
    main()
