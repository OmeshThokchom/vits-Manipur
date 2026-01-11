"""
VITS Inference with Word/Character Timestamps
Perfect for karaoke, subtitles, and lip-sync applications.
"""

import os
import json
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wavfile

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


def get_text(text, hps):
    """Convert text to sequence of symbol IDs."""
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def extract_timestamps(text, attn, hps):
    """
    Extract character and word timestamps from attention alignment.
    
    Args:
        text: Original input text
        attn: Attention matrix from model [1, 1, audio_frames, text_length]
        hps: Hyperparameters
    
    Returns:
        Dictionary with word and character timestamps
    """
    # Get alignment matrix
    alignment = attn.squeeze().cpu().numpy()  # [audio_frames, text_length]
    
    # Calculate time per frame
    hop_length = hps.data.hop_length
    sample_rate = hps.data.sampling_rate
    frame_duration = hop_length / sample_rate  # seconds per frame
    
    # Handle add_blank: if enabled, every other position is a blank
    add_blank = hps.data.add_blank
    
    # For each text position, find the frame range where it's active
    num_frames, num_positions = alignment.shape
    
    # Find start and end frame for each position
    char_timings = []
    
    for pos in range(num_positions):
        # Get the column for this position
        col = alignment[:, pos]
        
        # Find frames where this position is active (value > 0.5 for binary attention)
        active_frames = np.where(col > 0.5)[0]
        
        if len(active_frames) > 0:
            start_frame = active_frames[0]
            end_frame = active_frames[-1]
            start_time = start_frame * frame_duration
            end_time = (end_frame + 1) * frame_duration
        else:
            # Fallback: estimate from neighboring positions
            start_time = 0
            end_time = 0
        
        char_timings.append({
            "position": pos,
            "start": round(start_time, 3),
            "end": round(end_time, 3)
        })
    
    # Now map back to original text characters
    characters = []
    original_chars = list(text)
    
    if add_blank:
        # With add_blank, positions are: blank, char1, blank, char2, blank, ...
        # So actual characters are at odd positions (1, 3, 5, ...)
        for i, char in enumerate(original_chars):
            pos_in_sequence = i * 2 + 1  # Map to position in interspersed sequence
            if pos_in_sequence < len(char_timings):
                timing = char_timings[pos_in_sequence]
                characters.append({
                    "char": char,
                    "start": timing["start"],
                    "end": timing["end"]
                })
    else:
        # Direct mapping
        for i, char in enumerate(original_chars):
            if i < len(char_timings):
                timing = char_timings[i]
                characters.append({
                    "char": char,
                    "start": timing["start"],
                    "end": timing["end"]
                })
    
    # Group characters into words (split by space)
    words = []
    current_word = ""
    word_start = None
    word_end = None
    
    for char_info in characters:
        char = char_info["char"]
        
        if char == " ":
            # End current word
            if current_word:
                words.append({
                    "word": current_word,
                    "start": round(word_start, 3),
                    "end": round(word_end, 3)
                })
            current_word = ""
            word_start = None
            word_end = None
        else:
            # Add to current word
            current_word += char
            if word_start is None:
                word_start = char_info["start"]
            word_end = char_info["end"]
    
    # Don't forget the last word
    if current_word:
        words.append({
            "word": current_word,
            "start": round(word_start, 3),
            "end": round(word_end, 3)
        })
    
    # Calculate total duration
    total_duration = num_frames * frame_duration
    
    return {
        "text": text,
        "duration": round(total_duration, 3),
        "words": words,
        "characters": characters
    }


def synthesize(text, net_g, hps, device, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0):
    """
    Synthesize speech and extract timestamps.
    
    Returns:
        audio: numpy array of audio samples
        timestamps: dictionary with timing information
    """
    # Convert text to tensor
    stn_tst = get_text(text, hps)
    x_tst = stn_tst.unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    
    with torch.no_grad():
        # Generate audio and get attention
        audio, attn, mask, _ = net_g.infer(
            x_tst, 
            x_tst_lengths, 
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale
        )
        
        # Extract audio
        audio_np = audio[0, 0].data.cpu().numpy()
        
        # Extract timestamps from attention
        timestamps = extract_timestamps(text, attn, hps)
    
    return audio_np, timestamps


def main():
    parser = argparse.ArgumentParser(description="VITS Inference with Timestamps")
    parser.add_argument("-m", "--model_dir", type=str, default="./logs/meitei_test",
                        help="Model directory containing config.json and checkpoint")
    parser.add_argument("-t", "--text", type=str, default=None,
                        help="Text to synthesize (in Meitei Mayek)")
    parser.add_argument("-o", "--output_dir", type=str, default="./output",
                        help="Output directory for audio and timestamps")
    parser.add_argument("--noise_scale", type=float, default=0.667,
                        help="Noise scale for sampling")
    parser.add_argument("--length_scale", type=float, default=1.0,
                        help="Length scale (1.0 = normal speed)")
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  VITS Inference with Word Timestamps")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model_dir}")
    
    # Load config
    config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return
    
    hps = utils.get_hparams_from_file(config_path)
    
    # Load model
    print("\nLoading model...")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    net_g.eval()
    
    # Load checkpoint
    checkpoint_path = utils.latest_checkpoint_path(args.model_dir, "G_*.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_g.load_state_dict(checkpoint['model'])
    print(f"Loaded: {checkpoint_path}")
    
    # Test texts
    if args.text:
        test_texts = [args.text]
    else:
        # Default test texts in Meitei Mayek
        test_texts = [
            "ꯃꯅꯤꯄꯨꯔ",
            "ꯅꯣꯡꯄꯣꯛ ꯀꯌꯥ",
            "ꯁꯅꯥꯃꯅꯤ ꯃꯨꯇꯥ ꯊꯥꯎ",
        ]
    
    print("\n" + "-" * 60)
    print("Generating speech with timestamps...")
    print("-" * 60)
    
    all_results = []
    
    for i, text in enumerate(test_texts):
        print(f"\n[{i+1}/{len(test_texts)}] Text: {text}")
        
        try:
            # Synthesize
            audio, timestamps = synthesize(
                text, net_g, hps, device,
                noise_scale=args.noise_scale,
                length_scale=args.length_scale
            )
            
            # Save audio
            audio_path = os.path.join(args.output_dir, f"output_{i+1}.wav")
            wavfile.write(audio_path, hps.data.sampling_rate, audio)
            print(f"    Audio: {audio_path}")
            print(f"    Duration: {timestamps['duration']:.2f}s")
            
            # Save timestamps JSON
            json_path = os.path.join(args.output_dir, f"output_{i+1}_timestamps.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(timestamps, f, ensure_ascii=False, indent=2)
            print(f"    Timestamps: {json_path}")
            
            # Print word timings
            print(f"    Words:")
            for word_info in timestamps['words']:
                print(f"      [{word_info['start']:.2f}s - {word_info['end']:.2f}s] {word_info['word']}")
            
            all_results.append(timestamps)
            
        except Exception as e:
            print(f"    Error: {e}")
    
    # Save combined results
    combined_path = os.path.join(args.output_dir, "all_timestamps.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Done! Output saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
