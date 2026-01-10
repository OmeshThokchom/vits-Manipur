"""
CPU-only VITS Training Script
Optimized for laptops without GPU (i7 + 16GB RAM)
"""

import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import commons
import utils
from data_utils import (
    TextAudioLoader,
    TextAudioCollate,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

global_step = 0


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/meitei_laptop.json",
                        help='JSON config file path')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name/directory')
    
    args = parser.parse_args()
    
    model_dir = os.path.join("./logs", args.model)
    os.makedirs(model_dir, exist_ok=True)
    
    with open(args.config, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    class HParams:
        def __init__(self, **entries):
            for key, value in entries.items():
                if isinstance(value, dict):
                    self.__dict__[key] = HParams(**value)
                else:
                    self.__dict__[key] = value
    
    hps = HParams(**data)
    hps.model_dir = model_dir
    
    # Save config to model directory
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    return hps


def main():
    global global_step
    
    hps = get_hparams()
    device = torch.device('cpu')
    
    print("=" * 60)
    print("  VITS CPU Training for Meitei Mayek")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model directory: {hps.model_dir}")
    print(f"Batch size: {hps.train.batch_size}")
    print(f"Epochs: {hps.train.epochs}")
    print(f"Training files: {hps.data.training_files}")
    print(f"Validation files: {hps.data.validation_files}")
    print(f"Number of symbols: {len(symbols)}")
    print("=" * 60)
    
    # Setup logging
    logger = utils.get_logger(hps.model_dir)
    logger.info(f"Starting CPU training with config: {hps}")
    
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    
    torch.manual_seed(hps.train.seed)
    
    # Load datasets
    print("\n[1/4] Loading datasets...")
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    
    collate_fn = TextAudioCollate()
    
    train_loader = DataLoader(
        train_dataset, 
        num_workers=0,  # 0 for CPU to avoid multiprocessing issues
        shuffle=True, 
        batch_size=hps.train.batch_size,
        pin_memory=False,
        drop_last=True, 
        collate_fn=collate_fn
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        num_workers=0, 
        shuffle=False,
        batch_size=1,  # Smaller batch for eval
        pin_memory=False,
        drop_last=False, 
        collate_fn=collate_fn
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples: {len(eval_dataset)}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Initialize models
    print("\n[2/4] Initializing models...")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **vars(hps.model)
    ).to(device)
    
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    
    # Count parameters
    g_params = sum(p.numel() for p in net_g.parameters())
    d_params = sum(p.numel() for p in net_d.parameters())
    print(f"   Generator parameters: {g_params:,}")
    print(f"   Discriminator parameters: {d_params:,}")
    
    # Optimizers
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )
    
    # Try to load existing checkpoint
    epoch_start = 1
    try:
        g_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        d_path = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        _, _, _, epoch_start = utils.load_checkpoint(g_path, net_g, optim_g)
        utils.load_checkpoint(d_path, net_d, optim_d)
        global_step = (epoch_start - 1) * len(train_loader)
        print(f"\n[!] Resuming from epoch {epoch_start}, step {global_step}")
    except Exception as e:
        print(f"\n[!] Starting fresh training (no checkpoint found: {e})")
    
    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_start - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_start - 2
    )
    
    print("\n[3/4] Starting training...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(epoch_start, hps.train.epochs + 1):
        train_one_epoch(
            epoch, hps, device,
            net_g, net_d, 
            optim_g, optim_d,
            train_loader, eval_loader,
            logger, writer, writer_eval
        )
        scheduler_g.step()
        scheduler_d.step()
    
    print("\n[4/4] Training complete!")
    writer.close()
    writer_eval.close()


def train_one_epoch(epoch, hps, device, net_g, net_d, optim_g, optim_d, 
                    train_loader, eval_loader, logger, writer, writer_eval):
    global global_step
    
    net_g.train()
    net_d.train()
    
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
        # Move to device (CPU)
        x = x.to(device)
        x_lengths = x_lengths.to(device)
        spec = spec.to(device)
        spec_lengths = spec_lengths.to(device)
        y = y.to(device)
        y_lengths = y_lengths.to(device)
        
        # Forward pass through generator
        y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
        (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths)
        
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        
        y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)
        
        # ===== Discriminator =====
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
        
        optim_d.zero_grad()
        loss_disc_all.backward()
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        optim_d.step()
        
        # ===== Generator =====
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        
        optim_g.zero_grad()
        loss_gen_all.backward()
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        optim_g.step()
        
        # Logging
        if global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]['lr']
            losses = [loss_disc.item(), loss_gen.item(), loss_fm.item(), 
                     loss_mel.item(), loss_dur.item(), loss_kl.item()]
            
            print(f"Epoch {epoch} | Step {global_step} | "
                  f"Loss_D: {losses[0]:.4f} | Loss_G: {losses[1]:.4f} | "
                  f"Loss_mel: {losses[3]:.4f} | LR: {lr:.6f}")
            
            logger.info(f'Train Epoch: {epoch} [{100. * batch_idx / len(train_loader):.0f}%]')
            logger.info(f'Losses: {losses} | Step: {global_step} | LR: {lr}')
            
            # TensorBoard
            scalar_dict = {
                "loss/g/total": loss_gen_all.item(),
                "loss/d/total": loss_disc_all.item(),
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
                "loss/g/fm": loss_fm.item(),
                "loss/g/mel": loss_mel.item(),
                "loss/g/dur": loss_dur.item(),
                "loss/g/kl": loss_kl.item()
            }
            utils.summarize(writer=writer, global_step=global_step, scalars=scalar_dict)
        
        # Evaluation and checkpoint
        if global_step % hps.train.eval_interval == 0:
            evaluate(hps, device, net_g, eval_loader, writer_eval)
            
            # Save checkpoint
            utils.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, f"G_{global_step}.pth")
            )
            utils.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, f"D_{global_step}.pth")
            )
            print(f"  [Checkpoint saved at step {global_step}]")
        
        global_step += 1
    
    logger.info(f'====> Epoch: {epoch}')


def evaluate(hps, device, generator, eval_loader, writer_eval):
    global global_step
    generator.eval()
    
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
            x = x.to(device)
            x_lengths = x_lengths.to(device)
            spec = spec.to(device)
            spec_lengths = spec_lengths.to(device)
            y = y.to(device)
            y_lengths = y_lengths.to(device)
            
            # Just use first sample
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            break
        
        y_hat, attn, mask, *_ = generator.infer(x, x_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
        
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
    
    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
        "gen/audio": y_hat[0, :, :y_hat_lengths[0]]
    }
    
    if global_step == 0:
        image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})
    
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()
