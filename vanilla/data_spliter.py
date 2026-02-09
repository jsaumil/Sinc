import re
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchaudio
import tiktoken
import torch

def parse_transcript_file(path, wav_dir, audio_extension=".wav"):
    audio_files = []
    transcripts = []    
    pattern = re.compile(r'\(\s*(\S+)\s+"(.+?)"\s*\)')  
    with open(path, "r", encoding="utf-8") as f:
      for line in f:
        match = pattern.search(line)
        if match:
          audio_id = match.group(1)
          text = match.group(2) 
          audio_filename = audio_id + audio_extension
          audio_path = os.path.join(wav_dir, audio_filename)    
          audio_files.append(audio_path)
          transcripts.append(text)


    train_audio, val_audio, train_trans, val_trans = train_test_split(audio_files, transcripts, test_size=0.2, random_state=42)

    return train_audio, val_audio, train_trans, val_trans

class Data(Dataset):
    def __init__(self, file_list, transcript_list):
    
        self.file_list = file_list
        self.transcript_list = transcript_list

    def __len__(self):
      return len(self.file_list)

    def __getitem__(self, idx):
       audio_path = self.file_list[idx]
       transcript = self.transcript_list[idx]
       waveform, sample_rate = torchaudio.load(audio_path)
       return waveform, sample_rate, transcript
   
enc = tiktoken.get_encoding("cl100k_base")

def encode(text):
   return enc.encode(text, allowed_special="all")

PAD_ID = enc.encode("<|endoftext|>")
START_ID = enc.encode("<|endoftext|>")
END_ID = enc.encode("<|endoftext|>")

def decode(ids):
   ids = [i for i in ids if i !=PAD_ID]
   return enc.decode(ids)

def collate_fn(batch):
    waveforms, sample_rates, transcripts = zip(*batch)
    lengths = [w.size(1) for w in waveforms]
    max_len = max(lengths)  
    padded_waveforms = torch.zeros(len(waveforms), waveforms[0].size(0), max_len)
    for i,w in enumerate(waveforms):
       padded_waveforms[i, :, :w.size(1)] = w
      
    tokenized = []
    for t in transcripts:
       tokens = encode(t.lower())
       tokens = START_ID + tokens + END_ID
       tokenized.append(torch.tensor(tokens, dtype=torch.long))

    text_lengths = [len(t) for t in tokenized]
    max_text_len = max(text_lengths)

    padded_text = torch.full((len(tokenized), max_text_len), PAD_ID)

    for i, t in enumerate(tokenized):
       padded_text[i, :len(t)] = t

    return padded_waveforms, padded_text

def dataloader(path, wav_dir, batch_size):
   train_audio, val_audio, train_trans, val_trans = parse_transcript_file(path, wav_dir)
   train_dataset = Data(train_audio, train_trans)
   val_dataset = Data(val_audio, val_trans)

   train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
   val_loader = DataLoader(val_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
   return train_loader, val_loader