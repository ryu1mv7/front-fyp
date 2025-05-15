# core/backend/download_models.py

import os
from huggingface_hub import hf_hub_download

# 1) List all your model filenames
MODEL_FILES = [
    "best_gan_model.pth",
    "set_200generator.pth",
]

# 2) Your Hugging Face repo path
REPO_ID = "dlee0091/MDS16-models"

def download_models(dest_folder="models"):
    os.makedirs(dest_folder, exist_ok=True)
    for fname in MODEL_FILES:
        print(f"→ Downloading {fname}…")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            token=True  # uses your saved token
        )
        # Move into your chosen folder
        dest_path = os.path.join(dest_folder, fname)
        if local_path != dest_path:
            os.replace(local_path, dest_path)
        print(f"✔ Saved to {dest_path}")

if __name__ == "__main__":
    download_models()
