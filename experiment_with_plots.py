#!/usr/bin/env python3
"""
Napari Experiment with Plotting - Visual Output Version
This version processes images and creates detailed plots showing all results
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from skimage import io, color, filters, segmentation, measure
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use a non-interactive backend
plt.switch_backend('Agg')

def load_and_process_images():
    """Load and process all images"""
    print("Napari Image Processing Experiment with Visual Plots")
    print("=" * 60)
    
    workspace_path = "/home/ashu_17/Documents/Napari"
    image_files = [
        "Spoon.jpeg",
        "Umbrella.jpeg", 
        "Book.jpeg",
        "Key.jpeg",
        "download.jpeg"
    ]
    
    print("Loading and processing your images...")
    
    all_results = {}
    
    for filename in image_files:
        filepath = os.path.join(workspace_path, filename)
        
        if not os.path.exists(filepath):
            print(f"  ⚠️ File not found: {filename}")
            continue
            
        try:
            # Load the image
            image = io.imread(filepath)
            name = Path(filename).stem
            
            print(f"  ✓ Processing {name}...")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_image = color.rgb2gray(image)
            else:
                gray_image = image.astype(float) / 255.0 if image.dtype == np.uint8 else image
            
            # Apply various filters
            gaussian = filters.gaussian(gray_image, sigma=2)
            sobel = filters.sobel(gray_image)
            median = filters.median(gray_image, np.ones((3, 3)))
            
            # Edge detection
            edges = filters.sobel(gray_image)
            
            # Thresholding
            threshold_val = filters.threshold_otsu(gray_image)
            binary = gray_image > threshold_val
            
            # Segmentation
            try:
                distance = ndimage.distance_transform_edt(binary)
                # Use smaller kernel for local maxima to avoid issues
                local_maxima = filters.rank.maximum(distance, np.ones((5, 5))) == distance
                markers = measure.label(local_maxima)
                labels = segmentation.watershed(-distance, markers, mask=binary)
            except:
                # Simple fallback segmentation
                labels = measure.label(binary)
            
            # Store results
            all_results[name] = {
                'original': image,
                'gray': gray_image,
                'gaussian': gaussian,
                'sobel': sobel,
                'median': median,
                'edges': edges,
                'binary': binary,
                'labels': labels,
                'distance': distance if 'distance' in locals() else np.zeros_like(gray_image),
                'threshold_val': threshold_val,
                'num_objects': len(np.unique(labels)) - 1
            }
            
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
    
    return all_results

def create_comprehensive_plots(results):
    """Create comprehensive plots for each image"""
    print("\nCreating visual plots...")
    
    output_dir = "/home/ashu_17/Documents/Napari/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in results.items():
        print(f"  Creating plots for {name}...")
        
        # Create a comprehensive figure
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Image Processing Analysis: {name.upper()}', fontsize=16, fontweight='bold')
        
        # Row 1: Original, Grayscale, Gaussian, Median
        if len(data['original'].shape) == 3:
            axes[0, 0].imshow(data['original'])
            axes[0, 0].set_title('Original (RGB)')
        else:
            axes[0, 0].imshow(data['original'], cmap='gray')
            axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(data['gray'], cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(data['gaussian'], cmap='viridis')
        axes[0, 2].set_title('Gaussian Filter')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(data['median'], cmap='plasma')
        axes[0, 3].set_title('Median Filter')
        axes[0, 3].axis('off')
        
        # Row 2: Edge detection, Binary, Distance transform, Segmentation
        axes[1, 0].imshow(data['edges'], cmap='hot')
        axes[1, 0].set_title('Edge Detection (Sobel)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(data['binary'], cmap='RdYlBu')
        axes[1, 1].set_title(f'Binary (threshold: {data["threshold_val"]:.3f})')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(data['distance'], cmap='jet')
        axes[1, 2].set_title('Distance Transform')
        axes[1, 2].axis('off')
        
        # Segmentation with colorful labels
        axes[1, 3].imshow(data['labels'], cmap='nipy_spectral')
        axes[1, 3].set_title(f'Segmentation ({data["num_objects"]} objects)')
        axes[1, 3].axis('off')
        
        # Row 3: Histograms and analysis
        # Histogram of grayscale values
        axes[2, 0].hist(data['gray'].flatten(), bins=50, alpha=0.7, color='blue')
        axes[2, 0].axvline(data['threshold_val'], color='red', linestyle='--', 
                          label=f'Threshold: {data["threshold_val"]:.3f}')
        axes[2, 0].set_title('Grayscale Histogram')
        axes[2, 0].set_xlabel('Intensity')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].legend()
        
        # Edge strength histogram
        axes[2, 1].hist(data['edges'].flatten(), bins=50, alpha=0.7, color='orange')
        axes[2, 1].set_title('Edge Strength Distribution')
        axes[2, 1].set_xlabel('Edge Strength')
        axes[2, 1].set_ylabel('Frequency')
        
        # Object size analysis
        if data['num_objects'] > 0:
            object_sizes = []
            for obj_id in range(1, data['num_objects'] + 1):
                size = np.sum(data['labels'] == obj_id)
                object_sizes.append(size)
            
            axes[2, 2].bar(range(1, len(object_sizes) + 1), object_sizes, color='green', alpha=0.7)
            axes[2, 2].set_title('Object Sizes')
            axes[2, 2].set_xlabel('Object ID')
            axes[2, 2].set_ylabel('Size (pixels)')
        else:
            axes[2, 2].text(0.5, 0.5, 'No objects detected', ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('Object Sizes')
        
        # Statistics text
        stats_text = f"""Image Statistics:
Shape: {data['gray'].shape}
Min/Max: {data['gray'].min():.3f} / {data['gray'].max():.3f}
Mean: {data['gray'].mean():.3f}
Std: {data['gray'].std():.3f}
Edge Strength: {data['edges'].mean():.3f}
Objects Detected: {data['num_objects']}
Binary Coverage: {data['binary'].mean()*100:.1f}%"""
        
        axes[2, 3].text(0.05, 0.95, stats_text, transform=axes[2, 3].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[2, 3].set_xlim(0, 1)
        axes[2, 3].set_ylim(0, 1)
        axes[2, 3].axis('off')
        axes[2, 3].set_title('Statistics')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f'{name}_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {output_path}")

def create_comparison_plot(results):
    """Create a comparison plot of all images"""
    print("  Creating comparison plot...")
    
    num_images = len(results)
    fig, axes = plt.subplots(num_images, 6, figsize=(24, 4*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Comparative Image Processing Analysis', fontsize=20, fontweight='bold')
    
    for idx, (name, data) in enumerate(results.items()):
        # Original
        if len(data['original'].shape) == 3:
            axes[idx, 0].imshow(data['original'])
        else:
            axes[idx, 0].imshow(data['original'], cmap='gray')
        axes[idx, 0].set_title(f'{name}\nOriginal')
        axes[idx, 0].axis('off')
        
        # Grayscale
        axes[idx, 1].imshow(data['gray'], cmap='gray')
        axes[idx, 1].set_title('Grayscale')
        axes[idx, 1].axis('off')
        
        # Gaussian filtered
        axes[idx, 2].imshow(data['gaussian'], cmap='viridis')
        axes[idx, 2].set_title('Gaussian Filter')
        axes[idx, 2].axis('off')
        
        # Edge detection
        axes[idx, 3].imshow(data['edges'], cmap='hot')
        axes[idx, 3].set_title('Edge Detection')
        axes[idx, 3].axis('off')
        
        # Binary
        axes[idx, 4].imshow(data['binary'], cmap='RdYlBu')
        axes[idx, 4].set_title('Binary Mask')
        axes[idx, 4].axis('off')
        
        # Segmentation
        axes[idx, 5].imshow(data['labels'], cmap='nipy_spectral')
        axes[idx, 5].set_title(f'Segmentation\n({data["num_objects"]} objects)')
        axes[idx, 5].axis('off')
    
    plt.tight_layout()
    
    output_dir = "/home/ashu_17/Documents/Napari/plots"
    comparison_path = os.path.join(output_dir, 'comparison_all_images.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {comparison_path}")

def create_summary_statistics_plot(results):
    """Create summary statistics visualization"""
    print("  Creating summary statistics plot...")
    
    # Extract data for plotting
    names = list(results.keys())
    object_counts = [results[name]['num_objects'] for name in names]
    edge_strengths = [results[name]['edges'].mean() for name in names]
    binary_coverage = [results[name]['binary'].mean() * 100 for name in names]
    thresholds = [results[name]['threshold_val'] for name in names]
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Summary Statistics Across All Images', fontsize=16, fontweight='bold')
    
    # Object counts
    bars1 = axes[0, 0].bar(names, object_counts, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Detected Objects Count')
    axes[0, 0].set_ylabel('Number of Objects')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for bar, count in zip(bars1, object_counts):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       str(count), ha='center', va='bottom')
    
    # Edge strengths
    bars2 = axes[0, 1].bar(names, edge_strengths, color='orange', alpha=0.7)
    axes[0, 1].set_title('Average Edge Strength')
    axes[0, 1].set_ylabel('Edge Strength')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for bar, strength in zip(bars2, edge_strengths):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                       f'{strength:.3f}', ha='center', va='bottom')
    
    # Binary coverage
    bars3 = axes[1, 0].bar(names, binary_coverage, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Binary Coverage')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for bar, coverage in zip(bars3, binary_coverage):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{coverage:.1f}%', ha='center', va='bottom')
    
    # Thresholds
    bars4 = axes[1, 1].bar(names, thresholds, color='lightcoral', alpha=0.7)
    axes[1, 1].set_title('Otsu Threshold Values')
    axes[1, 1].set_ylabel('Threshold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    for bar, threshold in zip(bars4, thresholds):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{threshold:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_dir = "/home/ashu_17/Documents/Napari/plots"
    summary_path = os.path.join(output_dir, 'summary_statistics.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {summary_path}")

def main():
    """Main function"""
    try:
        # Process all images
        results = load_and_process_images()
        
        if not results:
            print("No images were successfully processed.")
            return
        
        # Create individual plots for each image
        create_comprehensive_plots(results)
        
        # Create comparison plot
        create_comparison_plot(results)
        
        # Create summary statistics
        create_summary_statistics_plot(results)
        
        print("\n" + "=" * 60)
        print("PLOTTING COMPLETE!")
        print("=" * 60)
        print(f"Generated plots for {len(results)} images")
        print("\nPlots saved in: /home/ashu_17/Documents/Napari/plots/")
        print("\nGenerated files:")
        
        output_dir = "/home/ashu_17/Documents/Napari/plots"
        for name in results.keys():
            print(f"  - {name}_analysis.png (detailed analysis)")
        print(f"  - comparison_all_images.png (side-by-side comparison)")
        print(f"  - summary_statistics.png (statistical summary)")
        
        print("\nYou can now:")
        print("1. View the individual analysis plots for detailed insights")
        print("2. Compare all images side-by-side in the comparison plot")
        print("3. Review statistical trends in the summary plot")
        print("4. Use these insights to understand your image processing results")
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        for name, data in results.items():
            print(f"{name.upper()}:")
            print(f"  Objects detected: {data['num_objects']}")
            print(f"  Edge strength: {data['edges'].mean():.3f}")
            print(f"  Binary coverage: {data['binary'].mean()*100:.1f}%")
            print(f"  Threshold: {data['threshold_val']:.3f}")
            print()
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
