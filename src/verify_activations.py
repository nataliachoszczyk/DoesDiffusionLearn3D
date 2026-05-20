import os
import glob
import numpy as np
import argparse

# --- Configuration ---
ACTIVATIONS_DIR = "/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/data/activations"
EXPECTED_SHAPES = ["Sphere", "SmoothCylinder", "SmoothCube_v2"] 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", default=["layer_04", "layer_10", "layer_18"], help="Target layers to verify")
    parser.add_argument("--timesteps", nargs="+", default=["t_800", "t_500", "t_200"], help="Target timesteps to verify")
    args = parser.parse_args()

    print("=== STARTING DATA VALIDATION ===\n")
    print(f"Targeting layers   : {args.layers}")
    print(f"Targeting timesteps: {args.timesteps}\n")
    
    if not os.path.exists(ACTIVATIONS_DIR):
        print(f"[ERROR] Directory not found: {ACTIVATIONS_DIR}")
        return

    # 1. Gather ONLY targeted files
    all_npy_files = []
    for layer in args.layers:
        for t in args.timesteps:
            for shape in EXPECTED_SHAPES:
                search_path = os.path.join(ACTIVATIONS_DIR, layer, t, shape, "*.npy")
                all_npy_files.extend(glob.glob(search_path))
                
    total_files = len(all_npy_files)
    print(f"Total .npy files found for validation: {total_files}")
    
    if total_files == 0:
        print("[WARNING] No files found for the specified targets.")
        return

    corrupted_files = []
    nan_files = []
    shape_mismatches = []
    reference_shape = None
    
    print("\nScanning files for integrity, NaNs, and shape consistency...")
    
    # 2. Scan specific files
    for idx, filepath in enumerate(all_npy_files):
        try:
            tensor = np.load(filepath)
            
            if np.isnan(tensor).any() or np.isinf(tensor).any():
                nan_files.append(filepath)
                
            current_shape = tensor.shape
            if reference_shape is None:
                reference_shape = current_shape
                print(f"[INFO] Reference tensor shape established as: {reference_shape}")
            elif current_shape != reference_shape:
                shape_mismatches.append((filepath, current_shape))
                
        except Exception as e:
            corrupted_files.append(filepath)

        if (idx + 1) % 500 == 0 or (idx + 1) == total_files:
            print(f"Processed {idx + 1}/{total_files} files...")

    # --- REPORTING ---
    print("\n" + "="*30)
    print("=== VALIDATION REPORT ===")
    print("="*30)
    
    print(f"Total Files Scanned : {total_files}")
    print(f"Tensor Dimensions   : {reference_shape}")
    
    if corrupted_files:
        print(f"\n[❌] CORRUPTED FILES: {len(corrupted_files)}")
    else:
        print("\n[✅] ALL FILES LOADED SUCCESSFULLY")

    if nan_files:
        print(f"\n[❌] FILES WITH NaNs/Infs: {len(nan_files)}")
    else:
        print("\n[✅] NO NaNs OR INFINITIES FOUND")

    if shape_mismatches:
        print(f"\n[❌] SHAPE MISMATCHES: {len(shape_mismatches)}")
    else:
        print("\n[✅] ALL TENSORS HAVE IDENTICAL SHAPES")
        
    print("\n=== BREAKDOWN BY CATEGORY ===")
    for layer in args.layers:
        for t in args.timesteps:
            for shape in EXPECTED_SHAPES:
                path = os.path.join(ACTIVATIONS_DIR, layer, t, shape)
                count = len(glob.glob(os.path.join(path, "*.npy")))
                print(f"{layer} | {t} | {shape.ljust(15)} : {count} files")

if __name__ == "__main__":
    main()