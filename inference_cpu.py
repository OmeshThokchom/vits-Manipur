"""
VITS Inference Script for Meitei Mayek TTS
Tests the trained model by generating speech from text.
"""

import os
import json
import torch
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


def main():
    # Configuration
    model_dir = "./logs/meitei_test"  # Use the test model we just trained
    config_path = os.path.join(model_dir, "config.json")
    
    # Check if model exists
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        print("Make sure you've run training first!")
        return
    
    print("=" * 60)
    print("  VITS Inference - Meitei Mayek TTS")
    print("=" * 60)
    
    # Load config
    hps = utils.get_hparams_from_file(config_path)
    device = torch.device('cpu')
    
    print(f"Model directory: {model_dir}")
    print(f"Device: {device}")
    print(f"Number of symbols: {len(symbols)}")
    
    # Load model
    print("\n[1/3] Loading model...")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    net_g.eval()
    
    # Find latest checkpoint
    try:
        checkpoint_path = utils.latest_checkpoint_path(model_dir, "G_*.pth")
        print(f"Loading checkpoint: {checkpoint_path}")
    except:
        print("Error: No checkpoint found!")
        return
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_g.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint (iteration {checkpoint['iteration']})")
    
    # Test texts in Meitei Mayek
    test_texts = [
        "ꯅꯣꯡꯄꯣꯛ ꯀꯌꯥ",  # Simple text
        "ꯁꯅꯥꯃꯅꯤ ꯃꯨꯇꯥ ꯊꯥꯎ",  # Another sample
        "ꯃꯅꯤꯄꯨꯔ",  # Manipur
    ]
    
    print("\n[2/3] Generating speech...")
    output_dir = "./inference_output"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            print(f"\n  Text {i+1}: {text}")
            
            try:
                # Convert text to tensor
                stn_tst = get_text(text, hps)
                x_tst = stn_tst.unsqueeze(0).to(device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                
                # Generate audio
                audio = net_g.infer(
                    x_tst, 
                    x_tst_lengths, 
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0
                )[0][0, 0].data.cpu().numpy()
                
                # Save audio
                output_path = os.path.join(output_dir, f"output_{i+1}.wav")
                wavfile.write(output_path, hps.data.sampling_rate, audio)
                print(f"  Saved: {output_path}")
                print(f"  Duration: {len(audio) / hps.data.sampling_rate:.2f}s")
                
            except Exception as e:
                print(f"  Error generating audio: {e}")
    
    print("\n[3/3] Inference complete!")
    print(f"\nOutput files saved to: {output_dir}/")
    print("\nNote: The model was trained for only 5 epochs, so audio quality will be poor.")
    print("For production quality, train for 500-1000+ epochs on GPU.")


if __name__ == "__main__":
    main()
