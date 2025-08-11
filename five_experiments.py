#!/usr/bin/env python3
"""
Five Different Napari Experiments with Visual Plots
Each experiment focuses on different image processing techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from skimage import io, color, filters, segmentation, measure, morphology, feature, restoration
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use a non-interactive backend
plt.switch_backend('Agg')

def load_images():
    """Load all images"""
    workspace_path = "/home/ashu_17/Documents/Napari"
    image_files = [
        "Spoon.jpeg",
        "Umbrella.jpeg", 
        "Book.jpeg",
        "Key.jpeg",
        "download.jpeg"
    ]
    
    images = {}
    for filename in image_files:
        filepath = os.path.join(workspace_path, filename)
        if os.path.exists(filepath):
            image = io.imread(filepath)
            name = Path(filename).stem
            
            # Convert to grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_image = color.rgb2gray(image)
            else:
                gray_image = image.astype(float) / 255.0 if image.dtype == np.uint8 else image
            
            images[name] = {
                'original': image,
                'gray': gray_image
            }
    
    return images

def experiment_1_edge_detection(images):
    """Experiment 1: Comprehensive Edge Detection Analysis"""
    print("Experiment 1: Edge Detection Analysis")
    print("-" * 40)
    
    results = {}
    
    for name, data in images.items():
        print(f"  Processing {name}...")
        gray = data['gray']
        
        # Different edge detection methods
        sobel = filters.sobel(gray)
        prewitt = filters.prewitt(gray)
        roberts = filters.roberts(gray)
        canny = feature.canny(gray, sigma=1.0)
        laplacian = filters.laplace(gray)
        
        results[name] = {
            'original': data['original'],
            'gray': gray,
            'sobel': sobel,
            'prewitt': prewitt,
            'roberts': roberts,
            'canny': canny,
            'laplacian': np.abs(laplacian)
        }
    
    # Create plots
    output_dir = "/home/ashu_17/Documents/Napari/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Individual plots for each image
    for name, data in results.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Experiment 1: Edge Detection Analysis - {name.upper()}', fontsize=16, fontweight='bold')
        
        # Row 1
        if len(data['original'].shape) == 3:
            axes[0, 0].imshow(data['original'])
        else:
            axes[0, 0].imshow(data['original'], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(data['gray'], cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(data['sobel'], cmap='hot')
        axes[0, 2].set_title('Sobel')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(data['prewitt'], cmap='hot')
        axes[0, 3].set_title('Prewitt')
        axes[0, 3].axis('off')
        
        # Row 2
        axes[1, 0].imshow(data['roberts'], cmap='hot')
        axes[1, 0].set_title('Roberts')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(data['canny'], cmap='hot')
        axes[1, 1].set_title('Canny')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(data['laplacian'], cmap='hot')
        axes[1, 2].set_title('Laplacian')
        axes[1, 2].axis('off')
        
        # Edge strength comparison
        methods = ['Sobel', 'Prewitt', 'Roberts', 'Laplacian']
        strengths = [
            np.mean(data['sobel']),
            np.mean(data['prewitt']),
            np.mean(data['roberts']),
            np.mean(data['laplacian'])
        ]
        
        axes[1, 3].bar(methods, strengths, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
        axes[1, 3].set_title('Edge Strength Comparison')
        axes[1, 3].set_ylabel('Mean Edge Strength')
        axes[1, 3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'experiment1_edge_detection_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"    ✓ Saved edge detection plots for {len(results)} images")
    return results

def experiment_2_noise_filtering(images):
    """Experiment 2: Noise Reduction and Filtering"""
    print("\nExperiment 2: Noise Reduction and Filtering")
    print("-" * 40)
    
    results = {}
    
    for name, data in images.items():
        print(f"  Processing {name}...")
        gray = data['gray']
        
        # Add artificial noise for demonstration
        noisy = gray + np.random.normal(0, 0.1, gray.shape)
        noisy = np.clip(noisy, 0, 1)
        
        # Different filtering methods
        gaussian = filters.gaussian(noisy, sigma=1)
        median = filters.median(noisy, np.ones((5, 5)))
        bilateral = restoration.denoise_bilateral(noisy, sigma_color=0.1, sigma_spatial=10)
        tv_denoised = restoration.denoise_tv_chambolle(noisy, weight=0.1)
        
        results[name] = {
            'original': data['original'],
            'clean': gray,
            'noisy': noisy,
            'gaussian': gaussian,
            'median': median,
            'bilateral': bilateral,
            'tv_denoised': tv_denoised
        }
    
    # Create plots
    output_dir = "/home/ashu_17/Documents/Napari/plots"
    
    for name, data in results.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Experiment 2: Noise Reduction - {name.upper()}', fontsize=16, fontweight='bold')
        
        # Row 1
        if len(data['original'].shape) == 3:
            axes[0, 0].imshow(data['original'])
        else:
            axes[0, 0].imshow(data['original'], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(data['clean'], cmap='gray')
        axes[0, 1].set_title('Clean Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(data['noisy'], cmap='gray')
        axes[0, 2].set_title('Noisy (Added Gaussian)')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(data['gaussian'], cmap='gray')
        axes[0, 3].set_title('Gaussian Filter')
        axes[0, 3].axis('off')
        
        # Row 2
        axes[1, 0].imshow(data['median'], cmap='gray')
        axes[1, 0].set_title('Median Filter')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(data['bilateral'], cmap='gray')
        axes[1, 1].set_title('Bilateral Filter')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(data['tv_denoised'], cmap='gray')
        axes[1, 2].set_title('Total Variation Denoising')
        axes[1, 2].axis('off')
        
        # Quality metrics
        methods = ['Gaussian', 'Median', 'Bilateral', 'TV Denoise']
        mse_values = [
            np.mean((data['clean'] - data['gaussian'])**2),
            np.mean((data['clean'] - data['median'])**2),
            np.mean((data['clean'] - data['bilateral'])**2),
            np.mean((data['clean'] - data['tv_denoised'])**2)
        ]
        
        axes[1, 3].bar(methods, mse_values, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
        axes[1, 3].set_title('Mean Squared Error\n(Lower is Better)')
        axes[1, 3].set_ylabel('MSE')
        axes[1, 3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'experiment2_noise_filtering_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"    ✓ Saved noise filtering plots for {len(results)} images")
    return results

def experiment_3_morphological_operations(images):
    """Experiment 3: Morphological Operations"""
    print("\nExperiment 3: Morphological Operations")
    print("-" * 40)
    
    results = {}
    
    for name, data in images.items():
        print(f"  Processing {name}...")
        gray = data['gray']
        
        # Create binary image
        threshold = filters.threshold_otsu(gray)
        binary = gray > threshold
        
        # Morphological operations
        eroded = morphology.erosion(binary, morphology.disk(3))
        dilated = morphology.dilation(binary, morphology.disk(3))
        opened = morphology.opening(binary, morphology.disk(3))
        closed = morphology.closing(binary, morphology.disk(3))
        skeleton = morphology.skeletonize(binary)
        
        results[name] = {
            'original': data['original'],
            'gray': gray,
            'binary': binary,
            'eroded': eroded,
            'dilated': dilated,
            'opened': opened,
            'closed': closed,
            'skeleton': skeleton
        }
    
    # Create plots
    output_dir = "/home/ashu_17/Documents/Napari/plots"
    
    for name, data in results.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Experiment 3: Morphological Operations - {name.upper()}', fontsize=16, fontweight='bold')
        
        # Row 1
        if len(data['original'].shape) == 3:
            axes[0, 0].imshow(data['original'])
        else:
            axes[0, 0].imshow(data['original'], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(data['gray'], cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(data['binary'], cmap='RdYlBu')
        axes[0, 2].set_title('Binary')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(data['eroded'], cmap='RdYlBu')
        axes[0, 3].set_title('Erosion')
        axes[0, 3].axis('off')
        
        # Row 2
        axes[1, 0].imshow(data['dilated'], cmap='RdYlBu')
        axes[1, 0].set_title('Dilation')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(data['opened'], cmap='RdYlBu')
        axes[1, 1].set_title('Opening')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(data['closed'], cmap='RdYlBu')
        axes[1, 2].set_title('Closing')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(data['skeleton'], cmap='hot')
        axes[1, 3].set_title('Skeleton')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'experiment3_morphological_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"    ✓ Saved morphological operations plots for {len(results)} images")
    return results

def experiment_4_segmentation_comparison(images):
    """Experiment 4: Different Segmentation Methods"""
    print("\nExperiment 4: Segmentation Methods Comparison")
    print("-" * 40)
    
    results = {}
    
    for name, data in images.items():
        print(f"  Processing {name}...")
        gray = data['gray']
        
        # Different segmentation methods
        # Otsu thresholding
        otsu_thresh = filters.threshold_otsu(gray)
        otsu_binary = gray > otsu_thresh
        
        # Multi-level thresholding
        multi_thresh = filters.threshold_multiotsu(gray, classes=3)
        multi_seg = np.digitize(gray, bins=multi_thresh)
        
        # Watershed segmentation
        edges = filters.sobel(gray)
        # Use peak_local_max instead
        from skimage.feature import peak_local_max
        coords = peak_local_max(filters.gaussian(gray, sigma=1), min_distance=20)
        markers = np.zeros_like(gray, dtype=bool)
        markers[tuple(coords.T)] = True
        markers = measure.label(markers)
        watershed_seg = segmentation.watershed(edges, markers)
        
        # Region growing
        try:
            coords2 = peak_local_max(filters.gaussian(gray, sigma=2), min_distance=30)
            seed_markers = np.zeros_like(gray, dtype=bool)
            seed_markers[tuple(coords2.T)] = True
            seed_labels = measure.label(seed_markers)
            region_seg = segmentation.watershed(-gray, seed_labels)
        except:
            region_seg = watershed_seg
        
        results[name] = {
            'original': data['original'],
            'gray': gray,
            'otsu_binary': otsu_binary,
            'multi_seg': multi_seg,
            'watershed_seg': watershed_seg,
            'region_seg': region_seg,
            'otsu_count': len(np.unique(otsu_binary)) - 1,
            'multi_count': len(np.unique(multi_seg)) - 1,
            'watershed_count': len(np.unique(watershed_seg)) - 1,
            'region_count': len(np.unique(region_seg)) - 1
        }
    
    # Create plots
    output_dir = "/home/ashu_17/Documents/Napari/plots"
    
    for name, data in results.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Experiment 4: Segmentation Comparison - {name.upper()}', fontsize=16, fontweight='bold')
        
        # Row 1
        if len(data['original'].shape) == 3:
            axes[0, 0].imshow(data['original'])
        else:
            axes[0, 0].imshow(data['original'], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(data['gray'], cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(data['otsu_binary'], cmap='RdYlBu')
        axes[0, 2].set_title(f'Otsu Binary\n({data["otsu_count"]} regions)')
        axes[0, 2].axis('off')
        
        # Row 2
        axes[1, 0].imshow(data['multi_seg'], cmap='nipy_spectral')
        axes[1, 0].set_title(f'Multi-Otsu\n({data["multi_count"]} regions)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(data['watershed_seg'], cmap='nipy_spectral')
        axes[1, 1].set_title(f'Watershed\n({data["watershed_count"]} regions)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(data['region_seg'], cmap='nipy_spectral')
        axes[1, 2].set_title(f'Region Growing\n({data["region_count"]} regions)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'experiment4_segmentation_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"    ✓ Saved segmentation comparison plots for {len(results)} images")
    return results

def experiment_5_texture_analysis(images):
    """Experiment 5: Texture and Feature Analysis"""
    print("\nExperiment 5: Texture and Feature Analysis")
    print("-" * 40)
    
    results = {}
    
    for name, data in images.items():
        print(f"  Processing {name}...")
        gray = data['gray']
        
        # Convert to uint8 for some texture functions
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        # Local Binary Pattern
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # Gabor filters
        gabor_real, gabor_imag = filters.gabor(gray, frequency=0.6)
        gabor_mag = np.sqrt(gabor_real**2 + gabor_imag**2)
        
        # Structure tensor
        Axx, Axy, Ayy = feature.structure_tensor(gray, sigma=1)
        eigenvals = feature.structure_tensor_eigenvalues([Axx, Axy, Ayy])
        coherence = eigenvals[0]
        
        # Entropy filter (local entropy)
        from skimage.filters.rank import entropy
        entropy_img = entropy(gray_uint8, morphology.disk(5))
        
        results[name] = {
            'original': data['original'],
            'gray': gray,
            'lbp': lbp,
            'gabor_real': gabor_real,
            'gabor_mag': gabor_mag,
            'coherence': coherence,
            'entropy': entropy_img
        }
    
    # Create plots
    output_dir = "/home/ashu_17/Documents/Napari/plots"
    
    for name, data in results.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Experiment 5: Texture Analysis - {name.upper()}', fontsize=16, fontweight='bold')
        
        # Row 1
        if len(data['original'].shape) == 3:
            axes[0, 0].imshow(data['original'])
        else:
            axes[0, 0].imshow(data['original'], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(data['gray'], cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(data['lbp'], cmap='hot')
        axes[0, 2].set_title('Local Binary Pattern')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(data['gabor_real'], cmap='RdBu')
        axes[0, 3].set_title('Gabor Filter (Real)')
        axes[0, 3].axis('off')
        
        # Row 2
        axes[1, 0].imshow(data['gabor_mag'], cmap='hot')
        axes[1, 0].set_title('Gabor Magnitude')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(data['coherence'], cmap='viridis')
        axes[1, 1].set_title('Structure Coherence')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(data['entropy'], cmap='plasma')
        axes[1, 2].set_title('Local Entropy')
        axes[1, 2].axis('off')
        
        # Texture statistics
        stats_text = f"""Texture Statistics:
LBP Mean: {np.mean(data['lbp']):.2f}
LBP Std: {np.std(data['lbp']):.2f}
Gabor Mean: {np.mean(data['gabor_mag']):.3f}
Coherence Mean: {np.mean(data['coherence']):.3f}
Entropy Mean: {np.mean(data['entropy']):.2f}"""
        
        axes[1, 3].text(0.05, 0.95, stats_text, transform=axes[1, 3].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace')
        axes[1, 3].set_xlim(0, 1)
        axes[1, 3].set_ylim(0, 1)
        axes[1, 3].axis('off')
        axes[1, 3].set_title('Texture Statistics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'experiment5_texture_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"    ✓ Saved texture analysis plots for {len(results)} images")
    return results

def create_experiments_summary(exp1, exp2, exp3, exp4, exp5):
    """Create a summary comparison of all experiments"""
    print("\nCreating experiments summary...")
    
    output_dir = "/home/ashu_17/Documents/Napari/plots"
    
    # Get image names
    image_names = list(exp1.keys())
    
    # Create summary for each image
    for name in image_names:
        fig, axes = plt.subplots(5, 6, figsize=(24, 20))
        fig.suptitle(f'All Experiments Summary - {name.upper()}', fontsize=20, fontweight='bold')
        
        # Experiment 1: Edge Detection
        if len(exp1[name]['original'].shape) == 3:
            axes[0, 0].imshow(exp1[name]['original'])
        else:
            axes[0, 0].imshow(exp1[name]['original'], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(exp1[name]['sobel'], cmap='hot')
        axes[0, 1].set_title('Sobel Edge')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(exp1[name]['canny'], cmap='hot')
        axes[0, 2].set_title('Canny Edge')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(exp1[name]['prewitt'], cmap='hot')
        axes[0, 3].set_title('Prewitt Edge')
        axes[0, 3].axis('off')
        
        axes[0, 4].imshow(exp1[name]['roberts'], cmap='hot')
        axes[0, 4].set_title('Roberts Edge')
        axes[0, 4].axis('off')
        
        axes[0, 5].text(0.1, 0.5, 'Experiment 1:\nEdge Detection\nMethods', 
                       transform=axes[0, 5].transAxes, fontsize=12, fontweight='bold')
        axes[0, 5].axis('off')
        
        # Experiment 2: Noise Filtering
        axes[1, 0].imshow(exp2[name]['noisy'], cmap='gray')
        axes[1, 0].set_title('Noisy Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(exp2[name]['gaussian'], cmap='gray')
        axes[1, 1].set_title('Gaussian Filter')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(exp2[name]['median'], cmap='gray')
        axes[1, 2].set_title('Median Filter')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(exp2[name]['bilateral'], cmap='gray')
        axes[1, 3].set_title('Bilateral Filter')
        axes[1, 3].axis('off')
        
        axes[1, 4].imshow(exp2[name]['tv_denoised'], cmap='gray')
        axes[1, 4].set_title('TV Denoising')
        axes[1, 4].axis('off')
        
        axes[1, 5].text(0.1, 0.5, 'Experiment 2:\nNoise Reduction\nand Filtering', 
                       transform=axes[1, 5].transAxes, fontsize=12, fontweight='bold')
        axes[1, 5].axis('off')
        
        # Experiment 3: Morphological Operations
        axes[2, 0].imshow(exp3[name]['binary'], cmap='RdYlBu')
        axes[2, 0].set_title('Binary')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(exp3[name]['eroded'], cmap='RdYlBu')
        axes[2, 1].set_title('Erosion')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(exp3[name]['dilated'], cmap='RdYlBu')
        axes[2, 2].set_title('Dilation')
        axes[2, 2].axis('off')
        
        axes[2, 3].imshow(exp3[name]['opened'], cmap='RdYlBu')
        axes[2, 3].set_title('Opening')
        axes[2, 3].axis('off')
        
        axes[2, 4].imshow(exp3[name]['skeleton'], cmap='hot')
        axes[2, 4].set_title('Skeleton')
        axes[2, 4].axis('off')
        
        axes[2, 5].text(0.1, 0.5, 'Experiment 3:\nMorphological\nOperations', 
                       transform=axes[2, 5].transAxes, fontsize=12, fontweight='bold')
        axes[2, 5].axis('off')
        
        # Experiment 4: Segmentation
        axes[3, 0].imshow(exp4[name]['otsu_binary'], cmap='RdYlBu')
        axes[3, 0].set_title('Otsu Binary')
        axes[3, 0].axis('off')
        
        axes[3, 1].imshow(exp4[name]['multi_seg'], cmap='nipy_spectral')
        axes[3, 1].set_title('Multi-Otsu')
        axes[3, 1].axis('off')
        
        axes[3, 2].imshow(exp4[name]['watershed_seg'], cmap='nipy_spectral')
        axes[3, 2].set_title('Watershed')
        axes[3, 2].axis('off')
        
        axes[3, 3].imshow(exp4[name]['region_seg'], cmap='nipy_spectral')
        axes[3, 3].set_title('Region Growing')
        axes[3, 3].axis('off')
        
        # Segmentation statistics
        seg_stats = f"""Segmentation Results:
Otsu: {exp4[name]['otsu_count']} regions
Multi-Otsu: {exp4[name]['multi_count']} regions
Watershed: {exp4[name]['watershed_count']} regions
Region: {exp4[name]['region_count']} regions"""
        
        axes[3, 4].text(0.05, 0.95, seg_stats, transform=axes[3, 4].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[3, 4].axis('off')
        
        axes[3, 5].text(0.1, 0.5, 'Experiment 4:\nSegmentation\nMethods', 
                       transform=axes[3, 5].transAxes, fontsize=12, fontweight='bold')
        axes[3, 5].axis('off')
        
        # Experiment 5: Texture Analysis
        axes[4, 0].imshow(exp5[name]['lbp'], cmap='hot')
        axes[4, 0].set_title('Local Binary Pattern')
        axes[4, 0].axis('off')
        
        axes[4, 1].imshow(exp5[name]['gabor_mag'], cmap='hot')
        axes[4, 1].set_title('Gabor Magnitude')
        axes[4, 1].axis('off')
        
        axes[4, 2].imshow(exp5[name]['coherence'], cmap='viridis')
        axes[4, 2].set_title('Structure Coherence')
        axes[4, 2].axis('off')
        
        axes[4, 3].imshow(exp5[name]['entropy'], cmap='plasma')
        axes[4, 3].set_title('Local Entropy')
        axes[4, 3].axis('off')
        
        # Texture statistics
        texture_stats = f"""Texture Statistics:
LBP Mean: {np.mean(exp5[name]['lbp']):.2f}
Gabor Mean: {np.mean(exp5[name]['gabor_mag']):.3f}
Coherence: {np.mean(exp5[name]['coherence']):.3f}
Entropy: {np.mean(exp5[name]['entropy']):.2f}"""
        
        axes[4, 4].text(0.05, 0.95, texture_stats, transform=axes[4, 4].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[4, 4].axis('off')
        
        axes[4, 5].text(0.1, 0.5, 'Experiment 5:\nTexture and\nFeature Analysis', 
                       transform=axes[4, 5].transAxes, fontsize=12, fontweight='bold')
        axes[4, 5].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'all_experiments_summary_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"    ✓ Saved comprehensive summary for {len(image_names)} images")

def main():
    """Main function to run all experiments"""
    print("Five Different Napari Experiments")
    print("=" * 60)
    
    # Load images
    print("Loading images...")
    images = load_images()
    
    if not images:
        print("No images found!")
        return
    
    print(f"Loaded {len(images)} images: {list(images.keys())}")
    
    # Run all experiments
    exp1_results = experiment_1_edge_detection(images)
    exp2_results = experiment_2_noise_filtering(images)
    exp3_results = experiment_3_morphological_operations(images)
    exp4_results = experiment_4_segmentation_comparison(images)
    exp5_results = experiment_5_texture_analysis(images)
    
    # Create comprehensive summary
    create_experiments_summary(exp1_results, exp2_results, exp3_results, exp4_results, exp5_results)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 60)
    print(f"Generated plots for {len(images)} images across 5 experiments")
    print("\nExperiment files generated:")
    print("1. experiment1_edge_detection_[image].png - Edge detection methods")
    print("2. experiment2_noise_filtering_[image].png - Noise reduction techniques")
    print("3. experiment3_morphological_[image].png - Morphological operations")
    print("4. experiment4_segmentation_[image].png - Segmentation methods")
    print("5. experiment5_texture_[image].png - Texture analysis")
    print("6. all_experiments_summary_[image].png - Complete overview")
    print(f"\nTotal plots generated: {len(images) * 6}")
    print("\nPlots saved in: /home/ashu_17/Documents/Napari/plots/")
if __name__ == "__main__":
    main()