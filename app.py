from flask import Flask, jsonify, request
import os
import subprocess
import random
import torch
import numpy as np
from PIL import Image
import shutil
import requests
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Get configuration from environment variables
app.config['DEBUG'] = True

# Create necessary directories
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / "unet").mkdir(exist_ok=True)
(MODELS_DIR / "vae").mkdir(exist_ok=True)
(MODELS_DIR / "clip").mkdir(exist_ok=True)

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
    return dest_path.exists()

def download_models():
    models = {
        "unet/flux1-dev-fp8.safetensors": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors",
        "vae/ae.sft": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft",
        "clip/clip_l.safetensors": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors",
        "clip/t5xxl_fp8_e4m3fn.safetensors": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
    }
    
    for model_path, url in models.items():
        dest_path = MODELS_DIR / model_path
        if not dest_path.exists():
            print(f"Downloading {model_path}...")
            download_file(url, dest_path)

# Download models on startup
download_models()

# Import ComfyUI related modules after model download
import nodes
from nodes import NODE_CLASS_MAPPINGS
from totoro_extras import nodes_custom_sampler
from totoro import model_management

# Initialize model components
DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

# Load models
with torch.inference_mode():
    clip = DualCLIPLoader.load_clip(
        str(MODELS_DIR / "clip/t5xxl_fp8_e4m3fn.safetensors"),
        str(MODELS_DIR / "clip/clip_l.safetensors"),
        "flux"
    )[0]
    unet = UNETLoader.load_unet(
        str(MODELS_DIR / "unet/flux1-dev-fp8.safetensors"),
        "fp8_e4m3fn"
    )[0]
    vae = VAELoader.load_vae(str(MODELS_DIR / "vae/ae.sft"))[0]

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to Flask API",
        "status": "success"
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy"
    })

@app.route('/generate-image', methods=['POST'])
def generate_image():
    # Get parameters from request, set defaults if not provided
    data = request.get_json()
    
    width = data.get('width', 1024)
    height = data.get('height', 1024)
    seed = data.get('seed', 0)
    steps = data.get('steps', 20)
    sampler_name = data.get('sampler_name', 'euler')
    scheduler = data.get('scheduler', 'simple')
    positive_prompt = data.get('positive_prompt', '')

    if seed == 0:
        seed = random.randint(0, 18446744073709551615)

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"generated_{seed}.png"

    with torch.inference_mode():
        cond, pooled = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
        cond = [[cond, {"pooled_output": pooled}]]
        noise = RandomNoise.get_noise(seed)[0]
        guider = BasicGuider.get_guider(unet, cond)[0]
        sampler = KSamplerSelect.get_sampler(sampler_name)[0]
        sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, 1.0)[0]
        latent_image = EmptyLatentImage.generate(closestNumber(width, 16), closestNumber(height, 16))[0]
        sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
        model_management.soft_empty_cache()
        decoded = VAEDecode.decode(vae, sample)[0].detach()
        
        # Save the generated image
        Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(str(output_path))

    return jsonify({
        "status": "success",
        "data": {
            "image_path": str(output_path),
            "parameters": {
                "width": width,
                "height": height,
                "seed": seed,
                "steps": steps,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive_prompt": positive_prompt
            }
        }
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 