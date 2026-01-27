"""
Unified Setup Script for VITS Training
Supports both fine-tuning (from checkpoint) and scratch training.

Usage:
  Interactive:    python setup.py
  Non-interactive: python setup.py --mode finetune --dataset-repo user/repo --token HF_TOKEN
"""

import os
import sys
import subprocess
import zipfile
import argparse
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
MODE_CONFIG = {
    "finetune": {
        "dataset_dir": "dataset/finetune",
        "config": "configs/meitei_finetune.json",
        "model_name": "meitei_finetune",
        "checkpoint_repo": "DayanandaThokchom/EmaLonTTS",
        "checkpoint_dir": "logs/meitei_finetune",
        "default_checkpoint_step": "240000",
    },
    "scratch": {
        "dataset_dir": "dataset/scratch",
        "config": "configs/meitei_prod.json",
        "model_name": "meitei_v1",
        "checkpoint_repo": None,
        "checkpoint_dir": None,
        "default_checkpoint_step": None,
    },
}


def run_command(cmd, cwd=None, shell=True):
    """Run a shell command and print output."""
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=shell, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        sys.exit(1)


def prompt_choice(prompt, choices, default=None):
    """Prompt user to choose from a list of choices."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        print(f"  [{i}] {choice}")
    
    while True:
        try:
            user_input = input(f"Enter choice (1-{len(choices)}): ").strip()
            if not user_input and default:
                return default
            idx = int(user_input) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")


def prompt_input(prompt, default=None, required=False):
    """Prompt user for text input."""
    suffix = f" [{default}]" if default else ""
    suffix += ": " if not required else " (required): "
    
    while True:
        user_input = input(f"{prompt}{suffix}").strip()
        if user_input:
            return user_input
        if default:
            return default
        if not required:
            return None
        print("This field is required. Please enter a value.")


def install_dependencies():
    """Install required dependencies."""
    print("\n[Step 1] Installing Dependencies...")
    run_command("pip install --upgrade pip")
    run_command("pip install -r requirements.txt")
    
    # Check for PyTorch
    try:
        import torch
        print(f"✅ Found PyTorch {torch.__version__}")
    except ImportError:
        print("PyTorch not found. Installing with CUDA support...")
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


def build_monotonic_align():
    """Build the Monotonic Align C extension."""
    print("\n[Step 2] Building Monotonic Align...")
    monotonic_path = Path("monotonic_align")
    
    if not monotonic_path.exists():
        print("Error: monotonic_align directory not found!")
        sys.exit(1)
    
    try:
        run_command("python setup.py build_ext --inplace", cwd=str(monotonic_path))
        print("✅ Monotonic Align built successfully")
    except SystemExit:
        print("\n⚠️  Build failed. Attempting to fix by creating subdirectory...")
        (monotonic_path / "monotonic_align").mkdir(exist_ok=True)
        try:
            run_command("python setup.py build_ext --inplace", cwd=str(monotonic_path))
            print("✅ Monotonic Align built successfully (with fix)")
        except:
            print("Build failed again. Please check the error message above.")
            sys.exit(1)


def setup_huggingface(token=None):
    """Setup Hugging Face authentication."""
    try:
        from huggingface_hub import login
    except ImportError:
        print("Installing huggingface_hub...")
        run_command("pip install huggingface_hub")
        from huggingface_hub import login
    
    hf_token = token or os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("\n⚠️  Private repos require authentication.")
        hf_token = prompt_input("Enter Hugging Face Token", required=False)
    
    if hf_token:
        print("Logging in to Hugging Face...")
        login(token=hf_token, add_to_git_credential=False)
    
    return hf_token


def download_dataset(dataset_repo, dataset_dir, hf_token):
    """Download and extract dataset from Hugging Face."""
    print(f"\n[Step 3] Downloading Dataset from {dataset_repo}...")
    
    from huggingface_hub import hf_hub_download, list_repo_files
    
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Find zip files in the repo
    try:
        files = list_repo_files(dataset_repo, repo_type="dataset", token=hf_token)
        zip_files = [f for f in files if f.endswith('.zip')]
        
        if not zip_files:
            print(f"No zip file found in {dataset_repo}!")
            sys.exit(1)
        
        filename = zip_files[0]
        local_zip_path = dataset_path / filename
        
        print(f"Found: {filename}")
        
        if not local_zip_path.exists():
            hf_hub_download(
                repo_id=dataset_repo,
                filename=filename,
                repo_type="dataset",
                local_dir=str(dataset_path),
                local_dir_use_symlinks=False,
                token=hf_token
            )
            print(f"✅ {filename} downloaded")
        else:
            print(f"✅ {filename} already exists")
        
        return local_zip_path
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("Double-check your token and repository name.")
        sys.exit(1)


def extract_dataset(zip_path, extract_dir):
    """Extract the dataset zip file."""
    print(f"\n[Step 4] Extracting Dataset...")
    
    extract_path = Path(extract_dir)
    
    # Check if already extracted
    if any(extract_path.glob("*.txt")) or (extract_path / "wavs").exists():
        print("Dataset appears to be already extracted.")
        return
    
    print(f"Extracting {zip_path}...")
    try:
        run_command(f'unzip -q -o "{zip_path}" -d "{extract_path}"')
        print("✅ Extraction complete (using unzip)")
    except:
        print("System unzip failed. Using Python zipfile...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Extraction complete (using zipfile)")


def download_checkpoint(checkpoint_repo, checkpoint_dir, checkpoint_step, hf_token):
    """Download pretrained checkpoint for fine-tuning."""
    print(f"\n[Step 5] Downloading Checkpoint (step {checkpoint_step})...")
    
    from huggingface_hub import hf_hub_download
    
    model_dir = Path(checkpoint_dir)
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
                if ckpt_file != "config.json":
                    sys.exit(1)
        else:
            print(f"  ✅ {ckpt_file} already exists")


def prepare_filelists(mode):
    """Prepare training filelists."""
    print(f"\n[Step 6] Preparing Filelists...")
    
    if mode == "finetune":
        run_command("python prepare_filelists_finetune.py")
    else:
        run_command("python prepare_filelists.py")
    
    print("✅ Filelists prepared")


def setup_logs_symlink():
    """Setup logs symlink for cloud training (RunPod etc.)."""
    logs_path = Path("logs")
    container_logs = Path("/root/vits_logs")
    
    # Only setup symlink on Linux (cloud environments)
    if sys.platform != "linux":
        return
    
    if not logs_path.exists() and not logs_path.is_symlink():
        print(f"\nSetting up logs symlink: logs -> {container_logs}")
        print("⚠️  WARNING: Checkpoints will be stored in Container disk.")
        print("   Make sure to download checkpoints before stopping the pod!")
        
        container_logs.mkdir(parents=True, exist_ok=True)
        os.symlink(container_logs, logs_path)


def main():
    print("=" * 60)
    print("  VITS Training Setup - Meitei Mayek TTS")
    print("=" * 60)
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="VITS Training Setup")
    parser.add_argument("--mode", choices=["finetune", "scratch"],
                        help="Training mode: finetune or scratch")
    parser.add_argument("--dataset-repo", 
                        help="Hugging Face dataset repo (e.g., username/dataset)")
    parser.add_argument("--token", 
                        help="Hugging Face token for private repos")
    parser.add_argument("--checkpoint-step", default="240000",
                        help="Checkpoint step to download (finetune only)")
    args = parser.parse_args()
    
    # ========================================================================
    # Interactive prompts (if CLI args not provided)
    # ========================================================================
    
    # 1. Training mode
    if args.mode:
        mode = args.mode
        print(f"\nMode: {mode}")
    else:
        mode_choice = prompt_choice(
            "Choose training mode:",
            ["Fine-tuning (resume from 240k checkpoint)", "Scratch training (train from zero)"]
        )
        mode = "finetune" if "Fine-tuning" in mode_choice else "scratch"
    
    config = MODE_CONFIG[mode]
    
    # 2. Dataset repo
    if args.dataset_repo:
        dataset_repo = args.dataset_repo
        print(f"Dataset repo: {dataset_repo}")
    else:
        default_repo = "DayanandaThokchom/finetune" if mode == "finetune" else "DayanandaThokchom/EmalonSpeech_V0.1"
        dataset_repo = prompt_input(
            "Enter Hugging Face dataset repo",
            default=default_repo,
            required=True
        )
    
    # ========================================================================
    # Setup Steps
    # ========================================================================
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Build Monotonic Align
    build_monotonic_align()
    
    # Step 3: Setup HF auth
    hf_token = setup_huggingface(args.token)
    
    # Step 4: Download dataset
    zip_path = download_dataset(dataset_repo, config["dataset_dir"], hf_token)
    
    # Step 5: Extract dataset
    extract_dataset(zip_path, config["dataset_dir"])
    
    # Step 6: Download checkpoint (finetune only)
    if mode == "finetune":
        download_checkpoint(
            config["checkpoint_repo"],
            config["checkpoint_dir"],
            args.checkpoint_step,
            hf_token
        )
    
    # Step 7: Prepare filelists
    prepare_filelists(mode)
    
    # Step 8: Setup logs symlink (cloud only)
    setup_logs_symlink()
    
    # ========================================================================
    # Done!
    # ========================================================================
    print("\n" + "=" * 60)
    print("  ✅ Setup Complete!")
    print("=" * 60)
    print(f"\nDataset downloaded to: {config['dataset_dir']}/")
    
    if mode == "finetune":
        print(f"Checkpoint downloaded to: {config['checkpoint_dir']}/")
    
    print(f"\nTo start training:")
    print(f"  python train.py -c {config['config']} -m {config['model_name']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
