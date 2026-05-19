import os
import torch
import numpy as np
from PIL import Image
from diffusers import SanaPipeline
from torchvision import transforms

IMAGE_PATH = "/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/data/final/images/Sphere/Sphere_new_000000.png"
OUTPUT_DIR = "./tests/test_output"
TARGET_LAYER = "transformer_blocks.14"
TIMESTEP = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

class SpatialActivationHook:
    def __init__(self):
        self.activation = None

    def __call__(self, module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        self.activation = hidden_states.detach().cpu().clone()

def main():
    print("[1] Checking device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("[2] Loading SANA-1.6B model...")
    pipeline = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        torch_dtype=torch.float16
    ).to(device)
    
    torch.set_grad_enabled(False)

    print(f"[3] Registering hook on layer: {TARGET_LAYER}...")
    hook_obj = SpatialActivationHook()
    
    handle = None

    print("Available pipeline components:", pipeline.components.keys())

    for name, module in pipeline.transformer.named_modules():
        if name == TARGET_LAYER:
            handle = module.register_forward_hook(hook_obj)
            break
            
    if handle is None:
        print(f"\n[ERROR] Layer '{TARGET_LAYER}' not found! Dostępne warstwy w transformerze zawierające '14':")
        for name, _ in pipeline.transformer.named_modules():
            if "14" in name:
                print(f" - {name}")
        raise ValueError(f"Layer {TARGET_LAYER} not found!")

    print(f"[4] Processing image through VAE ({IMAGE_PATH})...")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Nie znaleziono pliku obrazka pod ścieżką: {IMAGE_PATH}")

    image = Image.open(IMAGE_PATH).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((1024, 1024)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_tensor = preprocess(image).unsqueeze(0).to(device, dtype=torch.float16)
    
    latents = pipeline.vae.encode(img_tensor)[0]
    latents = latents * pipeline.vae.config.scaling_factor

    print("[5] Adding noise and running forward pass...")
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([TIMESTEP], device=device)
    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

    prompt_embeds = pipeline.encode_prompt(
        prompt="", 
        num_images_per_prompt=1, 
        do_classifier_free_guidance=False
    )[0]

    _ = pipeline.transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        return_dict=False
    )

    print("[6] Saving spatial activations...")
    if hook_obj.activation is not None:
        print(f"Captured tensor shape: {hook_obj.activation.shape}")
        save_path = os.path.join(OUTPUT_DIR, "test_activation_layer14.npy")
        np.save(save_path, hook_obj.activation.numpy())
        print(f"Saved to: {save_path}")
    else:
        print("Error: Hook did not capture any data.")

    handle.remove()

if __name__ == "__main__":
    main()