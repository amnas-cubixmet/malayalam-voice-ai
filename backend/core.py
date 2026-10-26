"""Scratch character-to-mel model and validated local data/checkpoint IO."""
import json
import math
import os
import re
import unicodedata
import uuid
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('VOICE_DATA_DIR', ROOT / 'data')).resolve()
AUDIO = DATA / 'recordings'
META = DATA / 'metadata.jsonl'
OUTPUT = DATA / 'generated'
CKPT = Path(os.environ.get('VOICE_CHECKPOINT_DIR', ROOT / 'checkpoints')).resolve() / 'latest.pt'
SR, HOP, NFFT, MEL = 22050, 256, 1024, 80
MAX_BYTES, MAX_SECONDS = 15_000_000, 30
FORMAT_VERSION = 1
CONFIG = {'sample_rate': SR, 'hop': HOP, 'n_fft': NFFT, 'n_mels': MEL, 'hidden': 192}


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text).strip())


def validate_text(text, limit=1000):
    text = normalize(text)
    if not text or len(text) > limit:
        raise ValueError(f'Text must contain 1–{limit} characters.')
    bad = [c for c in text if unicodedata.category(c).startswith('C') and c not in '\u200c\u200d']
    if bad:
        raise ValueError('Text contains unsupported control characters.')
    if not any(unicodedata.category(c).startswith(('L', 'N')) for c in text):
        raise ValueError('Text must contain spoken letters or numbers.')
    return text


def vocab_from_rows(rows):
    return {'<pad>': 0, '<unk>': 1, **{c: i + 2 for i, c in enumerate(sorted(set(''.join(normalize(r['text']) for r in rows))))}}


def encode(text, vocab):
    text = validate_text(text)
    unknown = sorted(set(text) - vocab.keys())
    if unknown:
        chars = ', '.join(f'{c!r} (U+{ord(c):04X})' for c in unknown)
        raise ValueError(f'Characters absent from training data: {chars}. Add recordings containing them and retrain.')
    return torch.tensor([vocab[c] for c in text], dtype=torch.long)


def recording_path(filename):
    if not isinstance(filename, str) or not re.fullmatch(r'[A-Za-z0-9_-]+\.wav', filename):
        raise ValueError('Invalid recording filename in metadata.')
    path = AUDIO / filename
    if path.is_symlink() or path.resolve().parent != AUDIO.resolve():
        raise ValueError('Recording path must stay inside the recordings directory.')
    return path


def load_rows():
    if not META.exists():
        return []
    rows = []
    for line in META.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        recording_path(row['file'])
        row['text'] = validate_text(row['text'])
        rows.append(row)
    return rows


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_wav(path, audio):
    audio = np.asarray(audio)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Synthesizer returned invalid audio samples.')
    pcm = (np.clip(audio, -1, 1) * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def read_wav(path):
    # A BytesIO is supported for validating uploads before writing them to disk.
    source = str(path) if isinstance(path, (str, Path)) else path
    with wave.open(source, 'rb') as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getcomptype()) != (1, 2, SR, 'NONE'):
            raise ValueError('Use mono 16-bit PCM WAV at 22050 Hz.')
        frames = f.getnframes()
        if not SR // 2 <= frames <= SR * MAX_SECONDS:
            raise ValueError(f'Recordings must be 0.5–{MAX_SECONDS} seconds long.')
        raw = f.readframes(frames)
        if len(raw) != frames * 2:
            raise ValueError('WAV is truncated: its audio data is shorter than its header.')
    audio = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768
    if float(np.max(np.abs(audio))) < 1e-4:
        raise ValueError('Recording is silent. Record audible speech before uploading.')
    return audio


class ScratchTTS(nn.Module):
    def __init__(self, vocab_size, hidden=192):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(nn.Conv1d(hidden * 2, 256, 5, padding=2), nn.ReLU(), nn.Conv1d(256, 256, 5, padding=2), nn.ReLU(), nn.Conv1d(256, MEL, 1))

    def forward(self, tokens, frames):
        if tokens.ndim != 1 or not tokens.numel() or frames < 2:
            raise ValueError('Expected nonempty 1D tokens and at least two output frames.')
        x, _ = self.encoder(self.embed(tokens.unsqueeze(0)))
        x = F.interpolate(x.transpose(1, 2), size=frames, mode='linear', align_corners=False)
        return self.decoder(x).squeeze(0)


@lru_cache(maxsize=1)
def audio_transforms():
    try:
        import torchaudio
    except (ImportError, OSError) as exc:
        raise RuntimeError('Cannot load torchaudio. Install matching torch/torchaudio builds; see README.') from exc
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=NFFT, hop_length=HOP, n_mels=MEL, power=2)
    # Pseudoinverse handles rank-deficient filterbanks without LAPACK gels failures.
    inverse = torch.linalg.pinv(mel.mel_scale.fb.T)
    return mel, inverse, torchaudio


def mel_spec(audio):
    transform, _, _ = audio_transforms()
    return torch.log(transform(torch.as_tensor(audio, dtype=torch.float32)) + 1e-5)


def reconstruct(logmel, iterations=48):
    if logmel.ndim != 2 or logmel.shape[0] != MEL or not torch.isfinite(logmel).all():
        raise ValueError('Expected finite [80, frames] log-mel tensor.')
    _, inverse, torchaudio = audio_transforms()
    with torch.no_grad():
        power_mel = logmel.detach().cpu().clamp(-11.51293, 20).exp()
        power_spec = (inverse @ power_mel).clamp_min(1e-10)
        audio = torchaudio.transforms.GriffinLim(n_fft=NFFT, hop_length=HOP, power=2, n_iter=iterations)(power_spec)
        if not torch.isfinite(audio).all() or audio.abs().max() <= 0:
            raise ValueError('Audio reconstruction produced invalid or silent samples.')
        audio = audio * (0.95 / audio.abs().max().clamp_min(0.95))
    return audio.numpy()


def load_checkpoint():
    if not CKPT.is_file():
        raise FileNotFoundError('No trained checkpoint. Upload at least two labeled recordings and complete training.')
    if CKPT.is_symlink():
        raise ValueError('Checkpoint symlinks are not allowed.')
    # Only our fixed local path is loaded; there is intentionally no checkpoint-upload API.
    cp = torch.load(CKPT, map_location='cpu', weights_only=True)
    if cp.get('format_version') != FORMAT_VERSION or cp.get('config') != CONFIG:
        raise ValueError('Incompatible or legacy checkpoint. Retrain with this version.')
    vocab = cp['vocab']
    if not isinstance(vocab, dict) or vocab.get('<pad>') != 0 or vocab.get('<unk>') != 1 or len(vocab) < 3:
        raise ValueError('Invalid checkpoint vocabulary.')
    if any(not isinstance(v, int) for v in vocab.values()) or sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError('Invalid checkpoint token indices.')
    if any(not isinstance(k, str) or len(k) != 1 for k in vocab if k not in ('<pad>', '<unk>')):
        raise ValueError('Invalid checkpoint character vocabulary.')
    if not math.isfinite(cp['frames_per_char']) or cp['frames_per_char'] <= 0 or cp['epoch'] < 1:
        raise ValueError('Invalid checkpoint training metadata.')
    model = ScratchTTS(len(vocab))
    model.load_state_dict(cp['state'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Checkpoint contains non-finite weights.')
    model.eval()
    return model, cp


def checkpoint_status():
    try:
        _, cp = load_checkpoint()
        return {'checkpoint': True, 'checkpoint_error': None, 'epoch': cp['epoch']}
    except FileNotFoundError as exc:
        return {'checkpoint': False, 'checkpoint_error': str(exc)}
    except Exception as exc:
        return {'checkpoint': False, 'checkpoint_error': f'Checkpoint cannot be loaded ({type(exc).__name__}). Retrain locally.'}
