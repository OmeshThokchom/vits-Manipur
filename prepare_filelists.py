"""
Convert VITS_TEST_WAV metadata to VITS filelist format
"""
import os

# Read metadata
with open('VITS_TEST_WAV/metadata.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

# Convert to VITS format: path|text
train_lines = []
val_lines = []

for i, line in enumerate(lines):
    parts = line.split('|')
    if len(parts) >= 2:
        sample_id = parts[0]  # e.g., Sample_000001
        text = parts[1]       # Meitei text
        
        # Create VITS format line
        wav_path = f"VITS_TEST_WAV/wavs/{sample_id}.wav"
        vits_line = f"{wav_path}|{text}"
        
        # Split: 90 train, 10 val
        if i < 90:
            train_lines.append(vits_line)
        else:
            val_lines.append(vits_line)

# Write train filelist
with open('filelists/meitei_train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_lines))

# Write val filelist
with open('filelists/meitei_val.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(val_lines))

print(f"Created filelists:")
print(f"  - meitei_train.txt: {len(train_lines)} samples")
print(f"  - meitei_val.txt: {len(val_lines)} samples")
print(f"\nSample train line:")
print(f"  {train_lines[0]}")
