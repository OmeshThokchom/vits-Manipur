"""
One-Click Setup Script for VITS Training (RunPod/Cloud GPU)
Automates:
1. Installing dependencies
2. Building Monotonic Align
3. Downloading Dataset from Hugging Face
4. Preparing Filelists
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

def run_command(cmd, cwd=None, shell=True):
    """Run a shell command and print output."""
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=shell, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        sys.exit(1)

def main():
    print("="*60)
    print("  VITS Training Setup - Meitei Mayek TTS")
    print("="*60)

    # 1. Install Dependencies
    print("\n[1/5] Installing Dependencies...")
    run_command("pip install --upgrade pip")
    run_command("pip install -r requirements.txt")
    
    # Check for PyTorch
    try:
        import torch
        print(f"Found PyTorch {torch.__version__}")
    except ImportError:
        print("PyTorch not found. Installing with CUDA support...")
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    # 2. Build Monotonic Align
    print("\n[2/5] Building Monotonic Align...")
    monotonic_path = Path("monotonic_align")
    if monotonic_path.exists():
        try:
            run_command("python setup.py build_ext --inplace", cwd=str(monotonic_path))
        except SystemExit:
            print("\n⚠️  Build failed. Attempting to fix by creating subdirectory...")
            # Sometimes setuptools wants the package dir to exist
            (monotonic_path / "monotonic_align").mkdir(exist_ok=True)
            try:
                run_command("python setup.py build_ext --inplace", cwd=str(monotonic_path))
            except:
                print("Build failed again. Please check the error message above.")
                sys.exit(1)
    else:
        print("Error: monotonic_align directory not found!")
        sys.exit(1)

    # 3. Download Dataset
    print("\n[3/5] Downloading Dataset from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError:
        print("Installing huggingface_hub...")
        run_command("pip install huggingface_hub")
        from huggingface_hub import hf_hub_download, login

    # Authentication
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="Hugging Face token for gated datasets")
    args, _ = parser.parse_known_args()

    hf_token = args.token or os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("\n⚠️  This dataset requires access.")
        print("Please enter your Hugging Face User Access Token (Write permission not needed).")
        hf_token = input("Token: ").strip()
        if not hf_token:
            print("No token provided. Attempting download without auth (might fail)...")
    
    if hf_token:
        print("Logging in to Hugging Face...")
        login(token=hf_token, add_to_git_credential=False)

    repo_id = "DayanandaThokchom/EmalonSpeech_V0.1"
    filename = "EmalonSpeech_V0.1.zip"
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    local_zip_path = dataset_dir / filename
    
    if not local_zip_path.exists():
        print(f"Downloading {filename}...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(dataset_dir),
                local_dir_use_symlinks=False,
                token=hf_token
            )
        except Exception as e:
            print(f"\nError downloading dataset: {e}")
            print("Double-check your token and access permissions.")
            sys.exit(1)
    else:
        print(f"{filename} already exists. Skipping download.")

    # 4. Extract Dataset
    print("\n[4/5] Extracting Dataset...")
    extract_path = dataset_dir / "EmalonSpeech"
    
    # Check if already extracted (look for metadata.txt or wavs folder)
    if (extract_path / "metadata.txt").exists() or (dataset_dir / "EmalonSpeech" / "metadata.txt").exists():
         print("Dataset appears to be already extracted.")
    else:
        print(f"Extracting {local_zip_path}...")
        # Try using system unzip first (much faster)
        try:
            run_command(f"unzip -q -o {local_zip_path} -d {dataset_dir}")
            print("Extraction complete (using unzip).")
        except:
            print("System unzip failed or not found. Falling back to Python zipfile (slower)...")
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print("Extraction complete (using zipfile).")

    # 5. Prepare Filelists
    print("\n[5/5] Preparing Filelists...")
    run_command("python prepare_filelists.py")

    # 6. Setup Logs Symlink (to use Container storage)
    print("\n[6/6] Setting up Logs Storage...")
    logs_path = Path("logs")
    container_logs = Path("/root/vits_logs")
    
    if not logs_path.exists() and not logs_path.is_symlink():
        print(f"Creating symlink: logs -> {container_logs}")
        print("⚠️  WARNING: Checkpoints will be stored in the Container disk.")
        print("    If you terminate the pod, these files will be LOST.")
        print("    Make sure to download your checkpoints before stopping the pod!")
        
        container_logs.mkdir(parents=True, exist_ok=True)
        os.symlink(container_logs, logs_path)
    else:
        print("Logs directory already exists. Skipping symlink.")

    print("\n" + "="*60)
    print("  Setup Complete! Ready to Train.")
    print("="*60)
    print("To start training:")
    print("  python train.py -c configs/meitei_prod.json -m meitei_v1")
    print("="*60)

if __name__ == "__main__":
    main()
