#!/usr/bin/env python3
"""
Napari Experiment - Console Output Version
This version shows processing results in the console without GUI
"""

import numpy as np
import os
from pathlib import Path
from skimage import io, color, filters, segmentation, measure
from scipy import ndimage

def load_and_analyze_images():
    """Load and analyze your images"""
    print("Napari Image Processing Experiment - Console Version")
    print("=" * 60)
    
    workspace_path = "/home/ashu_17/Documents/Napari"
    image_files = [
        "Spoon.jpeg",
        "Umbrella.jpeg", 
        "Book.jpeg",
        "Key.jpeg",
        "download.jpeg"
    ]
    
    print("Step 1: Loading your images...")
    
    results = {}
    
    for filename in image_files:
        filepath = os.path.join(workspace_path, filename)
        
        if not os.path.exists(filepath):
            print(f"  ⚠️ File not found: {filename}")
            continue
            
        try:
            # Load the image
            image = io.imread(filepath)
            name = Path(filename).stem
            
            print(f"  ✓ Loaded {filename}")
            print(f"    Shape: {image.shape}")
            print(f"    Data type: {image.dtype}")
            print(f"    Value range: {image.min()} to {image.max()}")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_image = color.rgb2gray(image)
                print(f"    Converted to grayscale: {gray_image.shape}")
            else:
                gray_image = image
            
            # Process the image
            print(f"  Processing {name}...")
            
            # Apply filters
            gaussian = filters.gaussian(gray_image, sigma=2)
            sobel = filters.sobel(gray_image)
            median = filters.median(gray_image, np.ones((5, 5)))
            
            # Thresholding
            threshold = filters.threshold_otsu(gray_image)
            binary = gray_image > threshold
            
            # Count objects (simple connected components)
            labels = measure.label(binary)
            num_objects = len(np.unique(labels)) - 1  # -1 for background
            
            # Calculate statistics
            edge_strength = np.mean(sobel)
            noise_level = np.std(gray_image - gaussian)
            
            results[name] = {
                'original_shape': image.shape,
                'gray_shape': gray_image.shape,
                'data_type': str(image.dtype),
                'value_range': f"{image.min():.2f} to {image.max():.2f}",
                'gray_range': f"{gray_image.min():.3f} to {gray_image.max():.3f}",
                'threshold_value': f"{threshold:.3f}",
                'detected_objects': num_objects,
                'edge_strength': f"{edge_strength:.3f}",
                'noise_level': f"{noise_level:.3f}",
                'binary_coverage': f"{np.mean(binary)*100:.1f}%"
            }
            
            print(f"    ✓ Processing complete")
            print(f"    Threshold: {threshold:.3f}")
            print(f"    Detected objects: {num_objects}")
            print(f"    Edge strength: {edge_strength:.3f}")
            print(f"    Binary coverage: {np.mean(binary)*100:.1f}%")
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
            print()
    
    return results

def display_summary(results):
    """Display summary of all results"""
    print("=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    
    if not results:
        print("No images were successfully processed.")
        return
    
    print(f"Successfully processed {len(results)} images:\n")
    
    # Create a comparison table
    print(f"{'Image':<12} {'Objects':<8} {'Edge Str':<9} {'Coverage':<9} {'Threshold':<10}")
    print("-" * 60)
    
    for name, data in results.items():
        print(f"{name:<12} {data['detected_objects']:<8} {data['edge_strength']:<9} "
              f"{data['binary_coverage']:<9} {data['threshold_value']:<10}")
    
    print("\nDetailed Analysis:")
    print("-" * 40)
    
    for name, data in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Original size: {data['original_shape']}")
        print(f"  Data type: {data['data_type']}")
        print(f"  Value range: {data['value_range']}")
        print(f"  Grayscale range: {data['gray_range']}")
        print(f"  Otsu threshold: {data['threshold_value']}")
        print(f"  Detected objects: {data['detected_objects']}")
        print(f"  Average edge strength: {data['edge_strength']}")
        print(f"  Noise level: {data['noise_level']}")
        print(f"  Binary coverage: {data['binary_coverage']}")
    
    # Analysis insights
    print("\n" + "=" * 60)
    print("ANALYSIS INSIGHTS")
    print("=" * 60)
    
    object_counts = [int(data['detected_objects']) for data in results.values()]
    edge_strengths = [float(data['edge_strength']) for data in results.values()]
    
    print(f"Object detection range: {min(object_counts)} to {max(object_counts)} objects")
    print(f"Edge strength range: {min(edge_strengths):.3f} to {max(edge_strengths):.3f}")
    
    # Find most/least complex images
    max_objects_img = max(results.keys(), key=lambda x: int(results[x]['detected_objects']))
    max_edges_img = max(results.keys(), key=lambda x: float(results[x]['edge_strength']))
    
    print(f"\nMost complex (objects): {max_objects_img} ({results[max_objects_img]['detected_objects']} objects)")
    print(f"Highest edge activity: {max_edges_img} (strength: {results[max_edges_img]['edge_strength']})")
    
    print("\nNext steps:")
    print("- To see visual results, run: napari (in a GUI environment)")
    print("- Try adjusting filter parameters for different effects")
    print("- Compare how different objects respond to the same processing")

def main():
    """Main function"""
    try:
        results = load_and_analyze_images()
        display_summary(results)
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE!")
        print("=" * 60)
        print("This analysis shows what Napari would visualize interactively.")
        print("In a GUI environment, you would see:")
        print("- Original images with multiple viewing options")
        print("- Processed versions with different colormaps")
        print("- Interactive layer controls")
        print("- Zoom, pan, and measurement tools")
        print("- Side-by-side comparisons")
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
