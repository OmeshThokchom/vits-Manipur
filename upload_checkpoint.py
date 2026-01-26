import os
import sys
import glob
from huggingface_hub import HfApi, login

def upload_checkpoints(repo_id, log_dir):
    """
    Uploads the latest checkpoints and config to Hugging Face.
    """
    api = HfApi()
    
    # 1. Authentication
    print("Checking authentication...")
    try:
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
    except:
        print("⚠️  Not logged in.")
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("Please enter your Hugging Face WRITE Token.")
            token = input("Token: ").strip()
        
        if token:
            login(token=token, add_to_git_credential=True)
        else:
            print("Error: No token provided.")
            sys.exit(1)

    # 2. Verify Repo Exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"✅ Found repository: {repo_id}")
    except:
        print(f"⚠️  Repository {repo_id} not found.")
        create = input("Create it now? (y/n): ").lower().strip()
        if create == 'y':
            api.create_repo(repo_id=repo_id, repo_type="model", private=True)
            print("✅ Repository created.")
        else:
            print("Aborting.")
            sys.exit(1)

    # 3. Upload Files
    print(f"\nUploading checkpoints from {log_dir}...")
    
    # We upload: config.json, G_*.pth, D_*.pth, and events (for tensorboard)
    # Using upload_folder is smartest as it handles large files and syncing
    try:
        url = api.upload_folder(
            folder_path=log_dir,
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["G_*.pth", "D_*.pth", "config.json", "events*"],
            commit_message="Upload training checkpoints (Backup)"
        )
        print("\n🎉 Upload Complete!")
        print(f"View your model here: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"\n❌ Error uploading: {e}")

if __name__ == "__main__":
    import argparse
    
    # Configuration defaults
    DEFAULT_REPO = "DayanandaThokchom/finetune-model_emalonTTS"
    DEFAULT_LOG_DIR = "logs/meitei_finetune"  # Changed for fine-tuning
    
    parser = argparse.ArgumentParser(description="Upload checkpoints to Hugging Face")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Target repo ID (default: {DEFAULT_REPO})")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help=f"Log directory (default: {DEFAULT_LOG_DIR})")
    args = parser.parse_args()
    
    print("="*60)
    print("  Hugging Face Model Uploader")
    print("="*60)
    
    print(f"Repo: {args.repo}")
    print(f"Log Dir: {args.log_dir}")
    
    upload_checkpoints(args.repo, args.log_dir)

