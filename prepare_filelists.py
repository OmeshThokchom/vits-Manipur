"""
Prepare Filelists for VITS Training
Adapts the EmalonSpeech dataset structure to VITS format.
"""
import os
from pathlib import Path

def main():
    # Dataset paths
    dataset_root = Path("dataset/EmalonSpeech")
    wavs_dir = dataset_root / "wavs"
    
    # Input files (pre-split from dataset)
    train_txt_path = dataset_root / "train.txt"
    val_txt_path = dataset_root / "val.txt"
    
    # Output files
    output_dir = Path("filelists")
    output_dir.mkdir(exist_ok=True)
    out_train_path = output_dir / "meitei_train.txt"
    out_val_path = output_dir / "meitei_val.txt"

    print(f"Looking for dataset at: {dataset_root}")

    if not train_txt_path.exists() or not val_txt_path.exists():
        print("Error: train.txt or val.txt not found in dataset folder!")
        print(f"Expected: {train_txt_path}")
        return

    def process_filelist(input_path, output_path):
        lines = []
        missing_count = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    file_id = parts[0]
                    text = parts[1]
                    
                    # Construct path relative to project root
                    # Assuming file_id is just filename without extension or with .wav
                    if not file_id.endswith('.wav'):
                        file_name = f"{file_id}.wav"
                    else:
                        file_name = file_id
                        
                    wav_path = wavs_dir / file_name
                    
                    # Verify file exists
                    if wav_path.exists():
                        # VITS expects: path/to/wav|text
                        # We use forward slashes for cross-platform compatibility
                        rel_path = str(wav_path).replace("\\", "/")
                        lines.append(f"{rel_path}|{text}")
                    else:
                        # Try finding it directly in dataset root if not in wavs
                        wav_path_root = dataset_root / file_name
                        if wav_path_root.exists():
                             rel_path = str(wav_path_root).replace("\\", "/")
                             lines.append(f"{rel_path}|{text}")
                        else:
                            missing_count += 1
                            if missing_count <= 5:
                                print(f"Warning: Audio file not found: {wav_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            
        return len(lines), missing_count

    print("Processing train list...")
    train_count, train_missing = process_filelist(train_txt_path, out_train_path)
    print(f"  - Saved {train_count} lines to {out_train_path}")
    if train_missing > 0:
        print(f"  - Skipped {train_missing} missing files")

    print("Processing val list...")
    val_count, val_missing = process_filelist(val_txt_path, out_val_path)
    print(f"  - Saved {val_count} lines to {out_val_path}")
    if val_missing > 0:
        print(f"  - Skipped {val_missing} missing files")

    print("\nFilelist preparation complete!")

if __name__ == "__main__":
    main()
