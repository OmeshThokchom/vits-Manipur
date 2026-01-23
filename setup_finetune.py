"""
One-Click Setup Script for VITS Fine-Tuning
Automates:
1. Installing dependencies
2. Building Monotonic Align
3. Downloading 240k Checkpoint from Hugging Face
4. Downloading Fine-tune Dataset from Hugging Face
5. Preparing Filelists
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
    print("  VITS Fine-Tuning Setup - Meitei Mayek TTS")
    print("="*60)

    # 1. Install Dependencies
    print("\n[1/7] Installing Dependencies...")
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
    print("\n[2/7] Building Monotonic Align...")
    monotonic_path = Path("monotonic_align")
    if monotonic_path.exists():
        try:
            run_command("python setup.py build_ext --inplace", cwd=str(monotonic_path))
        except SystemExit:
            print("\n⚠️  Build failed. Attempting to fix by creating subdirectory...")
            (monotonic_path / "monotonic_align").mkdir(exist_ok=True)
            try:
                run_command("python setup.py build_ext --inplace", cwd=str(monotonic_path))
            except:
                print("Build failed again. Please check the error message above.")
                sys.exit(1)
    else:
        print("Error: monotonic_align directory not found!")
        sys.exit(1)

    # 3. Download 240k Checkpoint from EmaLonTTS repo
    print("\n[3/7] Downloading 240k Checkpoint from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError:
        print("Installing huggingface_hub...")
        run_command("pip install huggingface_hub")
        from huggingface_hub import hf_hub_download, login

    # Authentication
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="Hugging Face token for private repos")
    parser.add_argument("--checkpoint-step", default="240000", 
                       help="Checkpoint step to download (default: 240000)")
    args, _ = parser.parse_known_args()

    hf_token = args.token or os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("\n⚠️  Private repos require authentication.")
        print("Please enter your Hugging Face User Access Token (or press Enter to skip).")
        hf_token = input("Token: ").strip()
    
    if hf_token:
        print("Logging in to Hugging Face...")
        login(token=hf_token, add_to_git_credential=False)

    # Checkpoint repo and files
    checkpoint_repo = "DayanandaThokchom/EmaLonTTS"
    checkpoint_step = args.checkpoint_step
    model_dir = Path("logs/meitei_finetune")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_files = [
        f"G_{checkpoint_step}.pth",
        f"D_{checkpoint_step}.pth",
        "config.json"
    ]
    
    for ckpt_file in checkpoint_files:
        local_path = model_dir / ckpt_file
        if not local_path.exists():
            print(f"Downloading {ckpt_file}...")
            try:
                hf_hub_download(
                    repo_id=checkpoint_repo,
                    filename=ckpt_file,
                    repo_type="model",
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                    token=hf_token
                )
                print(f"  ✅ {ckpt_file} downloaded")
            except Exception as e:
                print(f"  ❌ Error downloading {ckpt_file}: {e}")
                if ckpt_file != "config.json":  # config.json is optional
                    sys.exit(1)
        else:
            print(f"  ✅ {ckpt_file} already exists")

    # 4. Download Fine-tune Dataset
    print("\n[4/7] Downloading Fine-tune Dataset from Hugging Face...")
    
    # Fine-tune dataset repo
    dataset_repo = "DayanandaThokchom/finetune"
    dataset_dir = Path("dataset/finetune")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to find the zip file in the repo
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(dataset_repo, repo_type="dataset", token=hf_token)
        zip_files = [f for f in files if f.endswith('.zip')]
        
        if zip_files:
            filename = zip_files[0]
            local_zip_path = dataset_dir / filename
            print(f"Found zip file: {filename}")
            
            if not local_zip_path.exists():
                hf_hub_download(
                    repo_id=dataset_repo,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(dataset_dir),
                    local_dir_use_symlinks=False,
                    token=hf_token
                )
                print(f"  ✅ {filename} downloaded")
            else:
                print(f"  ✅ {filename} already exists")
        else:
            print("No zip file found in dataset repo!")
            sys.exit(1)
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Double-check your token and repository name.")
        sys.exit(1)

    # 5. Extract Dataset
    print("\n[5/7] Extracting Fine-tune Dataset...")
    extract_path = dataset_dir
    
    # Check if already extracted (look for fine_tune_data.txt or wavs folder)
    if (extract_path / "fine_tune_data.txt").exists() or (extract_path / "wavs").exists():
        print("Dataset appears to be already extracted.")
    else:
        # Find the zip file
        zip_files = list(dataset_dir.glob("*.zip"))
        if zip_files:
            local_zip_path = zip_files[0]
            print(f"Extracting {local_zip_path}...")
            try:
                run_command(f"unzip -q -o {local_zip_path} -d {extract_path}")
                print("Extraction complete (using unzip).")
            except:
                print("System unzip failed. Falling back to Python zipfile...")
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print("Extraction complete (using zipfile).")
        else:
            print("No zip file found to extract!")

    # 6. Prepare Filelists (for fine-tuning)
    print("\n[6/7] Preparing Filelists for Fine-tuning...")
    run_command("python prepare_filelists_finetune.py")

    # 7. Verify Checkpoint
    print("\n[7/7] Verifying Checkpoint...")
    g_checkpoint = model_dir / f"G_{checkpoint_step}.pth"
    d_checkpoint = model_dir / f"D_{checkpoint_step}.pth"
    
    if g_checkpoint.exists() and d_checkpoint.exists():
        print(f"✅ Generator checkpoint: {g_checkpoint}")
        print(f"✅ Discriminator checkpoint: {d_checkpoint}")
    else:
        print(f"⚠️  Checkpoint files missing in {model_dir}")

    print("\n" + "="*60)
    print("  Fine-Tune Setup Complete!")
    print("="*60)
    print(f"\nCheckpoints downloaded to: logs/meitei_finetune/")
    print(f"  - G_{checkpoint_step}.pth")
    print(f"  - D_{checkpoint_step}.pth")
    print("\nTo start fine-tuning from 240k checkpoint:")
    print("  python train.py -c configs/meitei_finetune.json -m meitei_finetune")
    print("="*60)

if __name__ == "__main__":
    main()

