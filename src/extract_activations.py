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
TARGET_LAYERS = ["transformer_blocks.4", "transformer_blocks.10", "transformer_blocks.18"]
TIMESTEPS = [800, 500, 200]
CHUNK_SIZE = 50 

class SpatialActivationHook:
    def __init__(self):
        self.activations = {}
        
    def hook_fn(self, layer_name):
        def forward_hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.activations[layer_name] = hidden_states.detach().cpu().clone()
        return forward_hook

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_shape", type=int, default=None, 
                        help="Maximum number of images to process PER SHAPE (e.g. 100).")
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
        images_by_shape[shape_name].append(img_path)

    # 2. Apply the per-shape limit
    all_images = []
    for shape, paths in images_by_shape.items():
        if args.max_per_shape is not None:
            selected_paths = paths[:args.max_per_shape]
        else:
            selected_paths = paths
            
        all_images.extend(selected_paths)
        print(f"Shape '{shape}': selected {len(selected_paths)} images.")

    # Sort the final combined list to ensure deterministic order across all jobs
    all_images.sort()

    total_images = len(all_images)
    print(f"Total images to process across all jobs: {total_images}")
    
    # Calculate indices for this specific SLURM job
    start_idx = task_id * CHUNK_SIZE
    end_idx = min(start_idx + CHUNK_SIZE, total_images)
    
    if start_idx >= total_images:
        print("No data for this Task ID (out of bounds). Exiting.")
        return

    my_images = all_images[start_idx:end_idx]
    print(f"This job will process {len(my_images)} images (indices {start_idx} to {end_idx-1}).")

    # 3. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        torch_dtype=torch.float16
    ).to(device)
    torch.set_grad_enabled(False)

    # 4. Register hooks
    hook_obj = SpatialActivationHook()
    handles = []
    
    for name, module in pipeline.transformer.named_modules():
        if name in TARGET_LAYERS:
            handle = module.register_forward_hook(hook_obj.hook_fn(name))
            handles.append(handle)

    if len(handles) != len(TARGET_LAYERS):
        raise ValueError("Not all target layers were found in the model.")

    preprocess = transforms.Compose([
        transforms.Resize((1024, 1024)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    prompt_embeds = pipeline.encode_prompt(prompt="", num_images_per_prompt=1, do_classifier_free_guidance=False)[0]

    # 5. Process images
    for img_path in my_images:
        relative_path = os.path.relpath(img_path, INPUT_ROOT)
        subfolder = os.path.dirname(relative_path) 
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(image).unsqueeze(0).to(device, dtype=torch.float16)
            
            latents = pipeline.vae.encode(img_tensor)[0]
            latents = latents * pipeline.vae.config.scaling_factor
            noise = torch.randn_like(latents)

            for t in TIMESTEPS:
                timesteps = torch.tensor([t], device=device)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                _ = pipeline.transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    return_dict=False
                )

                for layer_name in TARGET_LAYERS:
                    formatted_layer = layer_name.replace("transformer_blocks.", "layer_")
                    if len(formatted_layer.split("_")[-1]) == 1:
                         formatted_layer = formatted_layer.replace("layer_", "layer_0")
                         
                    target_dir = os.path.join(OUTPUT_ROOT, formatted_layer, f"t_{t}", subfolder)
                    os.makedirs(target_dir, exist_ok=True)
                    layer_num = layer_name.split(".")[-1].zfill(2)
                    file_suffix = f"_l{layer_num}_t{t}"
                    save_path = os.path.join(target_dir, f"{filename}{file_suffix}.npy")
                    
                    if not os.path.exists(save_path):
                         np.save(save_path, hook_obj.activations[layer_name].numpy())

        except Exception as e:
            print(f"[CRASH] Error processing {filename}: {str(e)}")
            
        hook_obj.activations.clear()

    for handle in handles:
        handle.remove()
        
    print(f"=== TASK ID {task_id} COMPLETED ===")

if __name__ == "__main__":
    main()