#!/usr/bin/env python3
"""
Test script to verify Napari installation
"""

import sys
import numpy as np

def test_napari_import():
    """Test if napari can be imported successfully"""
    try:
        import napari
        print(f"[PASS] Napari imported successfully! Version: {napari.__version__}")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import napari: {e}")
        return False

def test_napari_dependencies():
    """Test if key dependencies are available"""
    dependencies = {
        'numpy': None,
        'scipy': None,
        'scikit-image': 'skimage',
        'vispy': None,
        'qtpy': None
    }
    
    print("\nTesting key dependencies:")
    all_good = True
    
    for dep_name, import_name in dependencies.items():
        try:
            module_name = import_name if import_name else dep_name
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"[PASS] {dep_name}: {version}")
        except ImportError as e:
            print(f"[FAIL] {dep_name}: Failed to import")
            all_good = False
    
    return all_good

def test_napari_viewer_creation():
    """Test if we can create a napari viewer (headless mode)"""
    try:
        import napari
        # Create viewer in headless mode (won't show GUI)
        viewer = napari.Viewer(show=False)
        
        # Create some test data
        test_image = np.random.random((100, 100))
        viewer.add_image(test_image, name="test_image")
        
        print("[PASS] Napari viewer created successfully!")
        print(f"[PASS] Test image added to viewer: {len(viewer.layers)} layer(s)")
        
        # Clean up
        viewer.close()
        return True
        
    except Exception as e:
        print(f"[FAIL] Failed to create napari viewer: {e}")
        return False

def main():
    """Run all tests"""
    print("Verifying Napari installation...\n")
    
    # Test 1: Import
    import_ok = test_napari_import()
    
    # Test 2: Dependencies
    deps_ok = test_napari_dependencies()
    
    # Test 3: Viewer creation
    viewer_ok = test_napari_viewer_creation()
    
    print("\n" + "="*50)
    if import_ok and deps_ok and viewer_ok:
        print("[SUCCESS] All tests passed! Napari is properly installed and working.")
        print("\nYou can now:")
        print("1. Launch Napari GUI: napari")
        print("2. Use Napari in Python scripts")
        print("3. Load and visualize your images")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
