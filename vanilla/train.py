import torch
import shutil
from pathlib import Path
import math
from dataclasses import dataclass
import os
from pathlib import Path

from model import Sona
from data_spliter import dataloader

# parameters
num_epochs = 1000
# Find the actual location
home = Path.home()  # /home/Rohan
data_dir = home / "gujrati_male_mono1" / "mono"

path = str(data_dir / "txt.done.data")
wav_dir = str(data_dir / "wave")

# Verify paths exist
if not Path(path).exists():
    raise FileNotFoundError(f"CSV file not found: {path}")
if not Path(wav_dir).exists():
    raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

print(f"Using CSV: {path}")
print(f"Using WAV dir: {wav_dir}")
# path = r"../../gujrati_male_mono1/mono/txt.done.data"
# wav_dir = r"../../gujrati_male_mono1/mono/wave"

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 100278
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    sample_rate: int = 48000
    out_channels: int = 768
    kernel_size: int = 251
    stride: int = 1
    padding: int = 0

# data loading
train, val = dataloader(path, wav_dir, 1)

# parameter saving
checkpoint_dir = Path("checkpoints")
model_dir = Path("best_model")
def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    checkpoint_dir = Path(checkpoint_dir)
    best_model_dir = Path(best_model_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    f_path = checkpoint_dir / "checkpoint.pt"
    torch.save(state, f_path)

    if is_best:
        best_fpath = best_model_dir / "best_model.pt"
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("using device:", device)

# model defining
model = Sona(Config.sample_rate,Config(vocab_size=100278))

model.to(device)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 1
max_steps = 5

# learning rate
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr -min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
torch.set_float32_matmul_precision('high')

# loading the best
checkpoint_path = "best_model/best_model.pt"
if Path(checkpoint_path).exists():
    model, optimizer, start_epoch = load_ckp(checkpoint_path, model, optimizer)
    print(f"Resume training from epoch {start_epoch}")

else:
    start_epoch = 0
    print("Start training from scratch")

# best_val_loss = -inf
best_val_loss = float("inf")

# training steps
for step in range(start_epoch, num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    for waveforms, transcripts in train:
        optimizer.zero_grad()
        waveform = waveforms.to(device)

        d_in = transcripts[:,:-1].to(device)
        d_out = transcripts[:,1:].to(device)

        logits, loss = model(waveform, d_in, targets=d_out)
        train_loss += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    model.eval()
    with torch.no_grad():
        for waveforms, transcripts in val:
            waveform = waveforms.to(device)

            d_in = transcripts[:,:-1].to(device)
            d_out = transcripts[:,1:].to(device)

            logits, loss = model(waveform, d_in, targets=d_out)
            
            val_loss += loss.detach()
    is_best = val_loss < best_val_loss
    best_val_loss = min(best_val_loss, val_loss)
    checkpoint = {
        'epoch': step + 1,
        'state_dict' : model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)

    print(f"Epoch {step+1}, Train Loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, shape: {logits.shape}")