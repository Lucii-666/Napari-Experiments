#!/usr/bin/env python3
"""
Napari Experiment 1: Image Processing and Visualization
This experiment demonstrates:
1. Loading and displaying images
2. Image processing operations
3. Interactive visualization
4. Layer management
5. Adding annotations
"""

import numpy as np
import napari
from skimage import data, filters, segmentation, measure, io, color
from scipy import ndimage
import os
from pathlib import Path

def load_user_images():
    """Load user's images from the workspace"""
    print("Step 1: Loading your images...")
    
    # Define the workspace path
    workspace_path = "/home/ashu_17/Documents/Napari"
    
    # List of your image files
    image_files = [
        "Spoon.jpeg",
        "Umbrella.jpeg", 
        "Book.jpeg",
        "Key.jpeg",
        "download.jpeg"
    ]
    
    loaded_images = {}
    
    for filename in image_files:
        filepath = os.path.join(workspace_path, filename)
        try:
            # Load the image
            image = io.imread(filepath)
            
            # Convert to grayscale for processing if it's RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_image = color.rgb2gray(image)
                loaded_images[filename] = {
                    'original': image,
                    'gray': gray_image,
                    'name': Path(filename).stem
                }
            else:
                loaded_images[filename] = {
                    'original': image,
                    'gray': image,
                    'name': Path(filename).stem
                }
            
            print(f"  Loaded {filename}: {image.shape}")
            
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
    
    return loaded_images

def create_sample_data():
    """Create sample data for the experiment (fallback)"""
    print("Step 1: Creating sample data...")
    
    # Create synthetic data
    coins = data.coins()  # Sample image from scikit-image
    
    # Create a 3D volume (stack of images with noise)
    volume = np.zeros((10, 256, 256))
    for i in range(10):
        volume[i] = coins + np.random.normal(0, 10, coins.shape)
    
    # Create binary mask
    binary = coins > filters.threshold_otsu(coins)
    
    return coins, volume, binary

def process_multiple_images(images_dict):
    """Apply image processing operations to multiple images"""
    print("Step 2: Processing your images...")
    
    processed_results = {}
    
    for filename, image_data in images_dict.items():
        print(f"  Processing {image_data['name']}...")
        gray_image = image_data['gray']
        
        # Apply different filters
        gaussian = filters.gaussian(gray_image, sigma=2)
        sobel = filters.sobel(gray_image)
        median = filters.median(gray_image, np.ones((5, 5)))
        
        # Apply edge detection
        edges = filters.sobel(gray_image)
        
        # Thresholding for segmentation
        threshold = filters.threshold_otsu(gray_image)
        binary = gray_image > threshold
        
        # Segment the image using watershed
        try:
            distance = ndimage.distance_transform_edt(binary)
            local_maxima = filters.rank.maximum(distance, np.ones((20, 20))) == distance
            markers = measure.label(local_maxima)
            labels = segmentation.watershed(-distance, markers, mask=binary)
        except:
            # Fallback segmentation if watershed fails
            labels = measure.label(binary)
        
        processed_results[filename] = {
            'original': image_data['original'],
            'gray': gray_image,
            'gaussian': gaussian,
            'sobel': sobel,
            'median': median,
            'edges': edges,
            'binary': binary,
            'labels': labels,
            'name': image_data['name']
        }
    
    return processed_results

def create_annotations(image_shape):
    """Create sample annotations"""
    print("Step 3: Creating annotations...")
    
    # Create points (centroids of detected objects)
    y, x = np.mgrid[50:200:50, 50:200:50]
    points = np.column_stack([y.ravel(), x.ravel()])
    
    # Create shapes (rectangles and circles)
    shapes_data = [
        {'type': 'rectangle', 'data': [[50, 50], [100, 100]], 'face_color': 'red', 'edge_color': 'red'},
        {'type': 'ellipse', 'data': [[150, 150], [200, 200]], 'face_color': 'blue', 'edge_color': 'blue'},
    ]
    
    return points, shapes_data

def run_experiment():
    """Main experiment function"""
    print("Starting Napari Image Processing Experiment with Your Images...")
    print("=" * 60)
    
    # Step 1: Load your images
    user_images = load_user_images()
    
    if not user_images:
        print("No images found! Using sample data instead...")
        original_image, volume_data, binary_mask = create_sample_data()
        # Continue with original experiment logic...
        return
    
    # Step 2: Process all your images
    processed_images = process_multiple_images(user_images)
    
    # Step 4: Create Napari viewer
    print("Step 3: Creating Napari viewer...")
    viewer = napari.Viewer(title="Your Images - Processing Experiment")
    
    # Step 5: Add layers to the viewer
    print("Step 4: Adding layers to viewer...")
    
    # Add all images and their processed versions
    for filename, results in processed_images.items():
        name = results['name']
        
        # Add original image (RGB if available)
        if len(results['original'].shape) == 3:
            viewer.add_image(results['original'], name=f"{name} - Original (RGB)")
        
        # Add grayscale version
        viewer.add_image(results['gray'], name=f"{name} - Grayscale", colormap="gray", visible=False)
        
        # Add processed versions
        viewer.add_image(results['gaussian'], name=f"{name} - Gaussian", colormap="viridis", visible=False)
        viewer.add_image(results['sobel'], name=f"{name} - Sobel Edges", colormap="hot", visible=False)
        viewer.add_image(results['median'], name=f"{name} - Median Filter", colormap="plasma", visible=False)
        viewer.add_image(results['edges'], name=f"{name} - Edge Detection", colormap="magma", visible=False)
        
        # Add binary mask
        viewer.add_image(results['binary'].astype(int), name=f"{name} - Binary Mask", colormap="green", visible=False)
        
        # Add segmentation labels
        viewer.add_labels(results['labels'], name=f"{name} - Segmented", visible=False)
    
    # Create annotations for the first image
    first_image = list(processed_images.values())[0]
    points_data, shapes_data = create_annotations(first_image['gray'].shape)
    
    # Add annotations
    viewer.add_points(points_data, name="Sample Points", size=10, face_color="yellow", visible=False)
    
    # Add shapes
    shapes_layer = viewer.add_shapes(name="Manual Annotations", visible=False)
    for shape in shapes_data:
        if shape['type'] == 'rectangle':
            shapes_layer.add_rectangles(shape['data'], face_color=shape['face_color'], edge_color=shape['edge_color'])
        elif shape['type'] == 'ellipse':
            shapes_layer.add_ellipses(shape['data'], face_color=shape['face_color'], edge_color=shape['edge_color'])
    
    print("\nExperiment Setup Complete!")
    print(f"\nLoaded and processed {len(processed_images)} images:")
    for filename, results in processed_images.items():
        print(f"  - {results['name']}: {results['original'].shape}")
    
    print("\nWhat you can do now:")
    print("- Toggle layer visibility to compare original vs processed images")
    print("- Switch between different images and processing results") 
    print("- Adjust contrast and brightness for each layer")
    print("- Switch between different colormaps")
    print("- Compare edge detection results across different images")
    print("- Examine segmentation results")
    print("- Add manual annotations to interesting features")
    print("- Measure distances and areas on your images")
    
    # Display statistics for all images
    print(f"\nImage Analysis Results:")
    for filename, results in processed_images.items():
        name = results['name']
        gray = results['gray']
        labels = results['labels']
        print(f"\n{name}:")
        print(f"  - Shape: {gray.shape}")
        print(f"  - Data type: {gray.dtype}")
        print(f"  - Value range: {gray.min():.3f} to {gray.max():.3f}")
        print(f"  - Detected objects: {len(np.unique(labels)) - 1}")  # -1 for background
        
        # Calculate some basic statistics
        edge_strength = np.mean(results['edges'])
        print(f"  - Average edge strength: {edge_strength:.3f}")
    
    # Start the napari event loop
    napari.run()

if __name__ == "__main__":
    run_experiment()