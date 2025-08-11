# Napari Image Processing Experiments

A comprehensive collection of image processing experiments using Napari and scikit-image, demonstrating various computer vision techniques on real-world images.

## 🎯 Overview

This repository contains a complete image processing laboratory with 5 different experiments that showcase fundamental computer vision and image analysis techniques. Each experiment processes multiple sample images and generates detailed visualizations for educational and research purposes.

## 📁 Repository Structure

```
Napari/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── EXPERIMENT_GUIDE.md               # Detailed experiment guide
├── experiment-1.py                   # Original interactive experiment
├── experiment_console.py             # Console-only version
├── experiment_with_plots.py          # Single comprehensive experiment
├── five_experiments.py               # Complete 5-experiment suite
├── load_images.py                    # Image loading utility
├── test_napari_simple.py            # Installation verification
├── debug_test.py                     # Debug utilities
├── plots/                            # Generated visualizations
│   ├── experiment1_edge_detection_*.png
│   ├── experiment2_noise_filtering_*.png
│   ├── experiment3_morphological_*.png
│   ├── experiment4_segmentation_*.png
│   ├── experiment5_texture_*.png
│   └── all_experiments_summary_*.png
└── sample_images/                    # Your input images
    ├── Spoon.jpeg
    ├── Umbrella.jpeg
    ├── Book.jpeg
    ├── Key.jpeg
    └── download.jpeg
```

## 🔬 Experiments

### Experiment 1: Edge Detection Analysis
**File:** `experiment1_edge_detection_[image].png`

Compares multiple edge detection algorithms:
- **Sobel Operator** - Gradient-based edge detection
- **Prewitt Operator** - Alternative gradient method
- **Roberts Cross-Gradient** - Simple edge detection
- **Canny Edge Detector** - Multi-stage optimal edge detection
- **Laplacian** - Second-derivative edge enhancement

**Key Learning:** Understanding different approaches to edge detection and their sensitivity to noise and texture.

### Experiment 2: Noise Reduction and Filtering
**File:** `experiment2_noise_filtering_[image].png`

Demonstrates various denoising techniques:
- **Gaussian Filtering** - Linear smoothing filter
- **Median Filtering** - Non-linear noise reduction
- **Bilateral Filtering** - Edge-preserving smoothing
- **Total Variation Denoising** - Advanced regularization method

**Key Learning:** Comparing edge-preserving vs. smoothing filters and their effectiveness metrics.

### Experiment 3: Morphological Operations
**File:** `experiment3_morphological_[image].png`

Explores binary image processing operations:
- **Erosion & Dilation** - Basic morphological operations
- **Opening & Closing** - Composite operations for noise removal
- **Skeletonization** - Shape representation and analysis

**Key Learning:** Understanding structure analysis and shape modification techniques.

### Experiment 4: Segmentation Methods
**File:** `experiment4_segmentation_[image].png`

Compares different segmentation approaches:
- **Otsu Thresholding** - Automatic binary segmentation
- **Multi-level Otsu** - Multi-class segmentation
- **Watershed Segmentation** - Region-based segmentation
- **Region Growing** - Seed-based segmentation

**Key Learning:** Analyzing automatic segmentation strategies and their effectiveness on different image types.

### Experiment 5: Texture and Feature Analysis
**File:** `experiment5_texture_[image].png`

Advanced texture characterization methods:
- **Local Binary Patterns (LBP)** - Texture description
- **Gabor Filters** - Frequency and orientation analysis
- **Structure Tensor** - Local structure analysis
- **Local Entropy** - Information content mapping

**Key Learning:** Understanding texture characterization and feature extraction techniques.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Display environment for GUI (optional)

### Installation

1. **Clone or download the repository:**
   ```bash
   cd ~/Documents/Napari
   ```

2. **Set up Python virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install napari[all] matplotlib scipy scikit-image
   ```

4. **Verify installation:**
   ```bash
   python test_napari_simple.py
   ```

### Running Experiments

#### Option 1: Run All 5 Experiments (Recommended)
```bash
python five_experiments.py
```
This generates 30 individual plots + 5 summary plots = **37 total visualizations**

#### Option 2: Interactive Napari GUI
```bash
python experiment-1.py
```
Requires display environment for interactive visualization.

#### Option 3: Console Analysis Only
```bash
python experiment_console.py
```
Text-based analysis without plots.

#### Option 4: Single Comprehensive Analysis
```bash
python experiment_with_plots.py
```
Generates detailed analysis plots for all images.

## 📊 Generated Results

### Plot Types Generated

1. **Individual Experiment Plots** (25 files)
   - `experiment1_edge_detection_[ImageName].png`
   - `experiment2_noise_filtering_[ImageName].png`
   - `experiment3_morphological_[ImageName].png`
   - `experiment4_segmentation_[ImageName].png`
   - `experiment5_texture_[ImageName].png`

2. **Comprehensive Summaries** (5 files)
   - `all_experiments_summary_[ImageName].png`

3. **Additional Analysis** (7 files)
   - `comparison_all_images.png`
   - `summary_statistics.png`
   - `[ImageName]_analysis.png` (detailed individual analysis)

### Sample Results

Based on the included sample images:

| Image | Detected Objects | Edge Strength | Complexity |
|-------|-----------------|---------------|------------|
| **Umbrella** | 56 objects | 0.031 | High (fabric texture) |
| **Book** | 56 objects | 0.028 | High (text/binding) |
| **Spoon** | 31 objects | 0.021 | Medium (smooth metal) |
| **Download** | 14 objects | 0.019 | Medium |
| **Key** | 6 objects | 0.022 | Low (simple shape) |

## 🔧 Technical Details

### Key Libraries Used

- **Napari** (0.6.3) - Interactive image visualization
- **scikit-image** (0.25.2) - Image processing algorithms
- **NumPy** (2.2.6) - Numerical computing
- **SciPy** (1.16.1) - Scientific computing
- **Matplotlib** (latest) - Plotting and visualization

### Image Processing Techniques Implemented

1. **Edge Detection:**
   - Gradient-based methods (Sobel, Prewitt, Roberts)
   - Optimal edge detection (Canny)
   - Laplacian enhancement

2. **Filtering:**
   - Linear filters (Gaussian)
   - Non-linear filters (Median, Bilateral)
   - Advanced denoising (Total Variation)

3. **Morphological Analysis:**
   - Binary operations (Erosion, Dilation)
   - Composite operations (Opening, Closing)
   - Shape analysis (Skeletonization)

4. **Segmentation:**
   - Threshold-based (Otsu, Multi-Otsu)
   - Region-based (Watershed, Region Growing)
   - Marker-controlled segmentation

5. **Texture Analysis:**
   - Local descriptors (LBP)
   - Frequency analysis (Gabor filters)
   - Structural analysis (Structure tensor)
   - Information theory (Local entropy)

## 📈 Analysis Insights

### What You Can Learn

1. **Image Complexity Assessment:**
   - Compare object detection across different image types
   - Analyze texture complexity and edge strength
   - Understand noise characteristics

2. **Algorithm Performance:**
   - Compare effectiveness of different edge detectors
   - Evaluate segmentation method suitability
   - Assess noise reduction quality

3. **Parameter Effects:**
   - Observe how filter parameters affect results
   - Understand trade-offs between noise reduction and edge preservation
   - Learn about algorithm sensitivity

### Educational Value

- **Computer Vision Fundamentals:** Core image processing concepts
- **Algorithm Comparison:** Side-by-side method evaluation
- **Real-world Application:** Processing actual photographs vs. synthetic data
- **Visual Learning:** Immediate visual feedback for all operations
- **Quantitative Analysis:** Statistical metrics and comparisons

## 🎨 Customization

### Adding Your Own Images

1. Place image files in the main directory
2. Update the `image_files` list in any experiment script:
   ```python
   image_files = [
       "YourImage1.jpg",
       "YourImage2.png",
       # Add more images...
   ]
   ```

### Modifying Experiments

Each experiment is modular and can be customized:

- **Adjust parameters:** Change filter sizes, thresholds, etc.
- **Add new methods:** Implement additional algorithms
- **Modify visualizations:** Change colormaps, layouts, statistics
- **Export results:** Save processed images or data

### Creating New Experiments

Use the existing experiments as templates:

```python
def experiment_6_your_method(images):
    """Your custom experiment"""
    results = {}
    
    for name, data in images.items():
        # Your processing code here
        processed = your_algorithm(data['gray'])
        
        results[name] = {
            'original': data['original'],
            'processed': processed
        }
    
    # Create plots
    # ... plotting code ...
    
    return results
```

## 🐛 Troubleshooting

### Common Issues

1. **GUI not showing:**
   ```bash
   # Use console version instead
   python experiment_console.py
   ```

2. **Memory errors:**
   ```bash
   # Reduce image sizes or close other applications
   ```

3. **Import errors:**
   ```bash
   # Reinstall dependencies
   pip install --upgrade napari[all] matplotlib scipy scikit-image
   ```

4. **Permission errors:**
   ```bash
   # Check file permissions
   chmod +x *.py
   ```

### Getting Help

- Check the `EXPERIMENT_GUIDE.md` for detailed instructions
- Run `python debug_test.py` for diagnostic information
- Verify installation with `python test_napari_simple.py`

## 📚 References and Further Reading

### Scientific Background

- **Edge Detection:** Canny, J. (1986). "A Computational Approach to Edge Detection"
- **Morphological Operations:** Serra, J. (1983). "Image Analysis and Mathematical Morphology"
- **Texture Analysis:** Ojala, T. et al. (2002). "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns"
- **Segmentation:** Vincent, L. and Soille, P. (1991). "Watersheds in Digital Spaces"

### Software Documentation

- [Napari Documentation](https://napari.org/)
- [scikit-image Documentation](https://scikit-image.org/)
- [Image Processing with Python](https://github.com/scikit-image/scikit-image)

### Educational Resources

- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)
- [Digital Image Processing (Gonzalez & Woods)](https://www.imageprocessingplace.com/)
- [scikit-image Tutorials](https://scikit-image.org/docs/stable/auto_examples/)

## 🤝 Contributing

Feel free to:

- Add new experiments
- Improve existing algorithms
- Enhance visualizations
- Add more sample images
- Improve documentation
- Report bugs or suggest features

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## 🏆 Acknowledgments

- **Napari Team** - For the excellent visualization framework
- **scikit-image Contributors** - For comprehensive image processing tools
- **Python Scientific Community** - For the robust ecosystem

---

**Happy Image Processing! 🔬📸**

*This repository demonstrates the power of combining interactive visualization (Napari) with robust image processing (scikit-image) for educational and research purposes.*
