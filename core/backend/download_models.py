# core/backend/download_models.py

from pathlib import Path
from huggingface_hub import hf_hub_download

# 1) List all your model filenames
MODEL_FILES = [
    "best_gan_model.pth",
    "set_200generator.pth",
    "cgan_models_t2f_seg_250.pth",
    "ixi_multiclass_model.pt",
    "ixi_multiclass_model.pth",
    "pix2pix_weights.pth",
]

# 2) Your Hugging Face repo path
REPO_ID = "dlee0091/MDS16-models"

def download_models():
    # Base folder is this script's directory
    base_dir = Path(__file__).parent
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    for fname in MODEL_FILES:
        print(f"→ Downloading {fname}…")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            token=True
        )
        dest_path = model_dir / fname
        # Move the downloaded file into models/ if it's not already there
        if Path(local_path) != dest_path:
            Path(local_path).replace(dest_path)
        print(f"✔ Saved to {dest_path}")

if __name__ == "__main__":
    download_models()
