import json, math, re, wave
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; AUDIO=DATA/'recordings'; META=DATA/'metadata.jsonl'; CKPT=ROOT/'checkpoints'/'latest.pt'
SR=22050; HOP=256; NFFT=1024; MEL=80
# Character-level vocabulary created from the user's training transcripts; no pretrained tokenizer.
def normalize(text):
    return re.sub(r'\s+', ' ', text.strip())
def vocab_from_rows(rows):
    return {'<pad>':0,'<unk>':1,**{c:i+2 for i,c in enumerate(sorted(set(''.join(normalize(r['text']) for r in rows))))}}
def encode(text,vocab):
    return torch.tensor([vocab.get(c,1) for c in normalize(text)],dtype=torch.long)
def load_rows():
    if not META.exists(): return []
    return [json.loads(x) for x in META.read_text().splitlines() if x.strip()]
def save_wav(path,audio):
    x=np.clip(audio,-1,1); pcm=(x*32767).astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(SR);f.writeframes(pcm.tobytes())
def read_wav(path):
    with wave.open(str(path),'rb') as f:
        if f.getnchannels()!=1 or f.getsampwidth()!=2 or f.getframerate()!=SR:
            raise ValueError('Recording must be mono 16-bit PCM WAV at 22050 Hz')
        return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
# Small, scratch-initialized mel model: text encoder + frame expansion + convolutional decoder.
# A baseline, NOT a production TTS architecture.
class ScratchTTS(nn.Module):
    def __init__(self,vocab_size,hidden=192):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,hidden,padding_idx=0)
        self.encoder=nn.GRU(hidden,hidden,batch_first=True,bidirectional=True)
        self.decoder=nn.Sequential(nn.Conv1d(hidden*2,256,5,padding=2),nn.ReLU(),nn.Conv1d(256,256,5,padding=2),nn.ReLU(),nn.Conv1d(256,MEL,1))
    def forward(self,tokens,frames):
        x,_=self.encoder(self.embed(tokens.unsqueeze(0)))
        x=F.interpolate(x.transpose(1,2),size=frames,mode='linear',align_corners=False)
        return self.decoder(x).squeeze(0)
def mel_spec(audio):
    import torchaudio
    transform=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=NFFT,hop_length=HOP,n_mels=MEL,power=2)
    return torch.log(transform(torch.tensor(audio))+1e-5)
def reconstruct(logmel,iterations=48):
    import torchaudio
    mel=torch.exp(logmel.cpu()).clamp_min(0)
    inv=torchaudio.transforms.InverseMelScale(n_stft=NFFT//2+1,n_mels=MEL,sample_rate=SR)
    gl=torchaudio.transforms.GriffinLim(n_fft=NFFT,hop_length=HOP,n_iter=iterations)
    with torch.no_grad(): wav=gl(inv(mel))
    return wav.numpy()
