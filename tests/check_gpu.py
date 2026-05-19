import torch
import diffusers
from diffusers import SanaPipeline

def verify_environment():
    print("--- Environment Verification ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Diffusers version: {diffusers.__version__}")

    # Check if CUDA (GPU support) is available on the current node
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get details about the A100 GPU provided by Athena
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Device: {device_name}")
        print(f"Total VRAM: {total_mem:.2f} GB")

        # Check if we have enough VRAM for SANA-1.6B (approx. 16GB required)
        if total_mem >= 16:
            print("Status: Ready for SANA-1.6B inference.")
        else:
            print("Warning: VRAM might be insufficient for the full model.")
    else:
        print("Error: GPU not detected! Ensure you are running this via Slurm on a GPU node.")

if __name__ == "__main__":
    verify_environment()
