import os
import glob
import torch
import numpy as np
import argparse
from PIL import Image
from diffusers import SanaPipeline
from torchvision import transforms

# --- Configuration ---
INPUT_ROOT = "/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/data/final/images"
OUTPUT_ROOT = "/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/data/activations"
CHUNK_SIZE = 300 

class SpatialActivationHook:
    def __init__(self):
        self.activations = {}
        
    def hook_fn(self, layer_name):
        def forward_hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.activations[layer_name] = hidden_states.detach().cpu().clone()
        return forward_hook

def main():
    # Parse dynamic arguments for targeted extraction
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_shape", type=int, default=1000, help="Max images per shape.")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite of existing files.")
    parser.add_argument("--layers", nargs="+", default=["transformer_blocks.4", "transformer_blocks.10", "transformer_blocks.18"], help="Target layers.")
    parser.add_argument("--timesteps", nargs="+", type=int, default=[800, 500, 200], help="Target timesteps.")
    args = parser.parse_args()

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    print(f"=== STARTING TASK ID: {task_id} ===")

    # 1. Gather images and group by shape
    search_path = os.path.join(INPUT_ROOT, "**/*.png")
    all_images_raw = sorted(glob.glob(search_path, recursive=True))
    
    images_by_shape = {}
    for img_path in all_images_raw:
        shape_name = os.path.basename(os.path.dirname(img_path))
        if shape_name not in images_by_shape:
            images_by_shape[shape_name] = []
            
        # Optional: Skip shapes not in our list if you have other junk folders
        if shape_name in ["Sphere", "SmoothCylinder", "SmoothCube_v2"]:
            images_by_shape[shape_name].append(img_path)

    # Apply limits
    all_images = []
    for shape, paths in images_by_shape.items():
        selected_paths = paths[:args.max_per_shape]
        all_images.extend(selected_paths)
        print(f"Shape '{shape}': selected {len(selected_paths)} images.")

    all_images.sort()
    total_images = len(all_images)
    print(f"Total images to process across all jobs: {total_images}")
    
    # Target chunk for this array task
    start_idx = task_id * CHUNK_SIZE
    end_idx = min(start_idx + CHUNK_SIZE, total_images)
    
    if start_idx >= total_images:
        print("No data for this Task ID (out of bounds). Exiting.")
        return

    my_images = all_images[start_idx:end_idx]

    # 2. Setup Model (CRITICAL FIX: using bfloat16 to prevent NaNs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        torch_dtype=torch.bfloat16
    ).to(device)
    torch.set_grad_enabled(False)

    # 3. Register hooks dynamically
    hook_obj = SpatialActivationHook()
    handles = []
    
    for name, module in pipeline.transformer.named_modules():
        if name in args.layers:
            handle = module.register_forward_hook(hook_obj.hook_fn(name))
            handles.append(handle)

    if len(handles) != len(args.layers):
        raise ValueError(f"Target layers not found! Requested: {args.layers}")

    preprocess = transforms.Compose([
        transforms.Resize((1024, 1024)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    prompt_embeds = pipeline.encode_prompt(prompt="", num_images_per_prompt=1, do_classifier_free_guidance=False)[0]

    # 4. Process images
    for img_path in my_images:
        relative_path = os.path.relpath(img_path, INPUT_ROOT)
        subfolder = os.path.dirname(relative_path) 
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        try:
            image = Image.open(img_path).convert("RGB")
            # Convert input to bfloat16 to match model weights
            img_tensor = preprocess(image).unsqueeze(0).to(device, dtype=torch.bfloat16)
            
            latents = pipeline.vae.encode(img_tensor)[0]
            latents = latents * pipeline.vae.config.scaling_factor
            noise = torch.randn_like(latents)

            for t in args.timesteps:
                timesteps = torch.tensor([t], device=device)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                _ = pipeline.transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    return_dict=False
                )

                for layer_name in args.layers:
                    formatted_layer = layer_name.replace("transformer_blocks.", "layer_")
                    if len(formatted_layer.split("_")[-1]) == 1:
                         formatted_layer = formatted_layer.replace("layer_", "layer_0")
                         
                    target_dir = os.path.join(OUTPUT_ROOT, formatted_layer, f"t_{t}", subfolder)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    layer_num = layer_name.split(".")[-1].zfill(2)
                    file_suffix = f"_l{layer_num}_t{t}"
                    save_path = os.path.join(target_dir, f"{filename}{file_suffix}.npy")
                    
                    # Save if overwrite is True, OR if file doesn't exist
                    if args.overwrite or not os.path.exists(save_path):
                         # Convert bfloat16 back to float32 before saving because numpy handles float32 much better
                         safe_tensor = hook_obj.activations[layer_name].to(torch.float32).numpy()
                         np.save(save_path, safe_tensor)

        except Exception as e:
            print(f"[CRASH] Error processing {filename}: {str(e)}")
            
        hook_obj.activations.clear()

    for handle in handles:
        handle.remove()
        
    print(f"=== TASK ID {task_id} COMPLETED ===")

if __name__ == "__main__":
    main()