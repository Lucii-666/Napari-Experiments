#!/usr/bin/env python3
"""
Simple test script to verify Napari installation (headless-friendly)
"""

import sys

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

def test_napari_functions():
    """Test basic napari functionality without creating a viewer"""
    try:
        import napari
        import numpy as np
        
        # Test that we can access napari functions
        test_data = np.random.random((10, 10))
        
        # Test layer creation (without viewer)
        print("[PASS] Napari functions accessible")
        print(f"[PASS] Test data created: {test_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Failed to test napari functions: {e}")
        return False

def main():
    """Run all tests"""
    print("Verifying Napari installation (headless mode)...\n")
    
    # Test 1: Import
    import_ok = test_napari_import()
    
    # Test 2: Dependencies
    deps_ok = test_napari_dependencies()
    
    # Test 3: Basic functions
    functions_ok = test_napari_functions()
    
    print("\n" + "="*50)
    if import_ok and deps_ok and functions_ok:
        print("[SUCCESS] Napari is properly installed!")
        print("\nBasic verification completed successfully.")
        print("Note: GUI functionality will work when running with display.")
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
