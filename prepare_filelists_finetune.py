"""
Prepare Filelists for VITS Fine-Tuning
Handles the fine_tune_data.txt format: Sample_000001|ꯑꯦꯁꯤꯌꯥ ꯁ꯭ꯄꯦꯁꯤꯐꯤꯛ
"""
import os
import random
from pathlib import Path

def main():
    # Dataset paths for fine-tuning
    dataset_root = Path("dataset/finetune")
    wavs_dir = dataset_root / "wavs"
    
    # Input file (your fine-tune metadata)
    metadata_path = dataset_root / "fine_tune_data.txt"
    
    # Output files
    output_dir = Path("filelists")
    output_dir.mkdir(exist_ok=True)
    out_train_path = output_dir / "meitei_finetune_train.txt"
    out_val_path = output_dir / "meitei_finetune_val.txt"

    print(f"Looking for fine-tune dataset at: {dataset_root}")
    
    # Check for metadata file
    if not metadata_path.exists():
        # Try alternate names
        alt_names = ["metadata.txt", "transcripts.txt", "data.txt"]
        for alt in alt_names:
            alt_path = dataset_root / alt
            if alt_path.exists():
                metadata_path = alt_path
                break
        else:
            print(f"Error: Metadata file not found!")
            print(f"  Tried: {metadata_path}")
            print(f"  Also tried: {alt_names}")
            return

    print(f"Using metadata file: {metadata_path}")
    
    # Check wavs directory
    if not wavs_dir.exists():
        print(f"Error: wavs directory not found at {wavs_dir}")
        return

    # Process the metadata
    all_lines = []
    missing_count = 0
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('|')
            if len(parts) >= 2:
                file_id = parts[0].strip()
                text = parts[1].strip()
                
                # Handle with or without .wav extension
                if not file_id.endswith('.wav'):
                    file_name = f"{file_id}.wav"
                else:
                    file_name = file_id
                    
                wav_path = wavs_dir / file_name
                
                # Verify file exists
                if wav_path.exists():
                    # VITS expects: path/to/wav|text
                    rel_path = str(wav_path).replace("\\", "/")
                    all_lines.append(f"{rel_path}|{text}")
                else:
                    missing_count += 1
                    if missing_count <= 5:
                        print(f"Warning: Audio file not found: {wav_path}")

    if missing_count > 5:
        print(f"  ... and {missing_count - 5} more missing files")

    if not all_lines:
        print("Error: No valid audio files found!")
        return

    print(f"Found {len(all_lines)} valid audio-text pairs")
    
    # Split into train/val (95/5 split)
    random.seed(1234)
    random.shuffle(all_lines)
    
    val_size = max(1, int(len(all_lines) * 0.05))  # At least 1 for validation
    val_lines = all_lines[:val_size]
    train_lines = all_lines[val_size:]

    # Write output files
    with open(out_train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(out_val_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

    print(f"\n✅ Filelists created:")
    print(f"   Train: {out_train_path} ({len(train_lines)} samples)")
    print(f"   Val:   {out_val_path} ({len(val_lines)} samples)")

if __name__ == "__main__":
    main()
