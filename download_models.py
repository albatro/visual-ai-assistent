"""
Download all required model weights for MuseTalk v1.5.
Run: python download_models.py
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "MuseTalk" / "models"


def download_hf_model(repo_id: str, local_dir: Path, allow_patterns=None):
    """Download a model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    print(f"\n>>> Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    print(f"    Done: {repo_id}")


def download_gdown(file_id: str, output: Path):
    """Download a file from Google Drive via gdown."""
    import gdown
    if output.exists():
        print(f"    Already exists: {output}")
        return
    print(f"\n>>> Downloading Google Drive {file_id} -> {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(output), quiet=False)


def download_url(url: str, output: Path):
    """Download a file from a direct URL."""
    import requests
    if output.exists():
        print(f"    Already exists: {output}")
        return
    print(f"\n>>> Downloading {url}")
    output.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(output, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)", end="")
    print()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. MuseTalk v1.5 UNet
    musetalk_v15_dir = MODELS_DIR / "musetalkV15"
    if not (musetalk_v15_dir / "unet.pth").exists():
        download_hf_model(
            "TMElyralab/MuseTalk",
            MODELS_DIR,
            allow_patterns=["musetalkV15/*"],
        )
    else:
        print(f"MuseTalk v1.5 UNet already downloaded")

    # 2. SD VAE
    vae_dir = MODELS_DIR / "sd-vae-ft-mse"
    if not (vae_dir / "diffusion_pytorch_model.bin").exists() and not (vae_dir / "diffusion_pytorch_model.safetensors").exists():
        download_hf_model(
            "stabilityai/sd-vae-ft-mse",
            vae_dir,
        )
    else:
        print(f"SD VAE already downloaded")

    # 3. Whisper (tiny)
    whisper_dir = MODELS_DIR / "whisper"
    if not (whisper_dir / "pytorch_model.bin").exists() and not (whisper_dir / "model.safetensors").exists():
        download_hf_model(
            "openai/whisper-tiny",
            whisper_dir,
        )
    else:
        print(f"Whisper already downloaded")

    # 4. DWPose
    dwpose_dir = MODELS_DIR / "dwpose"
    dwpose_file = dwpose_dir / "dw-ll_ucoco_384.pth"
    if not dwpose_file.exists():
        download_hf_model(
            "TMElyralab/MuseTalk",
            MODELS_DIR,
            allow_patterns=["dwpose/*"],
        )
    else:
        print(f"DWPose already downloaded")

    # 5. Face parsing (BiSeNet)
    fp_dir = MODELS_DIR / "face-parse-bisent"
    fp_file = fp_dir / "79999_iter.pth"
    if not fp_file.exists():
        download_hf_model(
            "TMElyralab/MuseTalk",
            MODELS_DIR,
            allow_patterns=["face-parse-bisent/*"],
        )
    else:
        print(f"Face parsing already downloaded")

    # 6. ResNet18 for face parsing (from PyTorch hub)
    resnet_file = fp_dir / "resnet18-5c106cde.pth"
    if not resnet_file.exists():
        download_url(
            "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            resnet_file,
        )
    else:
        print(f"ResNet18 already downloaded")

    print("\n=== All models downloaded successfully! ===")

    # Verify
    print("\nVerification:")
    checks = [
        musetalk_v15_dir / "unet.pth",
        musetalk_v15_dir / "musetalk.json",
        vae_dir / "config.json",
        whisper_dir / "config.json",
        dwpose_dir / "dw-ll_ucoco_384.pth",
        fp_dir / "79999_iter.pth",
        resnet_file,
    ]
    all_ok = True
    for p in checks:
        exists = p.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {p.relative_to(MODELS_DIR)}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\nAll model files verified!")
    else:
        print("\nSome files are missing. Please check the download logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
