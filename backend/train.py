"""One local training job at a time, shared by the API and CLI."""
import argparse
import json
import os
import traceback
import uuid
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from filelock import FileLock, Timeout
from core import (DATA, CKPT, CONFIG, FORMAT_VERSION, ScratchTTS, load_rows,
                  read_wav, recording_path, mel_spec, encode, vocab_from_rows, atomic_json)

STATUS = DATA / 'training_status.json'
STATUS_LOCK = DATA / 'training_status.lock'
LOCK = DATA / 'training.lock'
DATA_LOCK = DATA / 'dataset.lock'


def now():
    return datetime.now(timezone.utc).isoformat()


def acquire_training_lock():
    DATA.mkdir(parents=True, exist_ok=True)
    lock = FileLock(LOCK, thread_local=False)
    lock.acquire(timeout=0)
    return lock


def is_training():
    try:
        lock = acquire_training_lock()
    except Timeout:
        return True
    lock.release()
    return False


def read_status():
    # Coordinate Windows readers with the atomic writer; Windows may deny a
    # rename while another process has the status file open for reading.
    with FileLock(STATUS_LOCK):
        status = json.loads(STATUS.read_text(encoding='utf-8')) if STATUS.exists() else {'status': 'idle', 'history': []}
    active = is_training()
    if not active and status['status'] in ('starting', 'running'):
        status.update(status='interrupted', error='Training stopped before completion. Start a new run; any previous valid checkpoint is preserved.')
    status['active'] = active
    return status


def train(epochs=20, lr=0.0003, lock=None, job_id=None):
    if not 1 <= epochs <= 1000:
        raise ValueError('epochs must be 1–1000')
    lock = lock or acquire_training_lock()
    state = {'job_id': job_id or uuid.uuid4().hex, 'status': 'running', 'started_at': now(), 'epochs': epochs,
             'epoch': 0, 'step': 0, 'progress': 0, 'loss': None, 'history': [], 'error': None, 'log_tail': []}
    log_path = DATA / 'train_stdout.log'
    def report(message=None):
        if message:
            state['log_tail'] = (state['log_tail'] + [message])[-40:]
            with log_path.open('a', encoding='utf-8') as f:
                f.write(message + '\n')
        state['updated_at'] = now()
        with FileLock(STATUS_LOCK):
            atomic_json(STATUS, state)
    try:
        log_path.write_text('', encoding='utf-8')
        report('Validating dataset and audio dependencies…')
        with FileLock(DATA_LOCK):
            rows = load_rows()
        if len(rows) < 2:
            raise ValueError('At least two valid labeled recordings are required.')
        # Snapshot metadata; immutable uploaded recordings remain safe during further uploads.
        vocab = vocab_from_rows(rows)
        device_name = os.environ.get('VOICE_DEVICE', 'cpu')
        if device_name not in ('cpu', 'cuda'):
            raise ValueError('VOICE_DEVICE must be cpu or cuda.')
        device = torch.device(device_name)
        state.update(samples=len(rows), device=str(device), total_steps=epochs * len(rows))
        # Cache targets once, not again for every epoch/checkpoint.
        examples = [(encode(r['text'], vocab), mel_spec(read_wav(recording_path(r['file'])))) for r in rows]
        frames_per_char = sum(t.shape[-1] for _, t in examples) / sum(len(x) for x, _ in examples)
        torch.manual_seed(42)
        model = ScratchTTS(len(vocab)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        generator = torch.Generator().manual_seed(42)
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for index in torch.randperm(len(examples), generator=generator).tolist():
                tokens, target = examples[index]
                target = target.to(device)
                pred = model(tokens.to(device), target.shape[-1])
                loss = F.l1_loss(pred, target)
                if not torch.isfinite(loss):
                    raise ValueError('Training loss became non-finite; no checkpoint was published.')
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                opt.step()
                losses.append(float(loss.item()))
                state.update(step=state['step'] + 1, loss=losses[-1])
                state['progress'] = state['step'] / state['total_steps']
                report()
            value = sum(losses) / len(losses)
            state['epoch'] = epoch
            state['history'] = (state['history'] + [{'epoch': epoch, 'loss': value}])[-100:]
            report(f'Epoch {epoch}/{epochs}: mean L1 loss {value:.6f}')
        checkpoint = {'format_version': FORMAT_VERSION, 'config': CONFIG, 'state': {k: v.cpu() for k, v in model.state_dict().items()},
                      'vocab': vocab, 'frames_per_char': frames_per_char, 'epoch': epochs, 'samples': len(rows), 'job_id': state['job_id']}
        if any(not torch.isfinite(v).all() for v in checkpoint['state'].values()):
            raise ValueError('Training produced non-finite weights.')
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        temp = CKPT.with_name(state['job_id'] + '.tmp')
        try:
            torch.save(checkpoint, temp)
            # Verify the serialized weights before atomically replacing a prior working model.
            saved = torch.load(temp, map_location='cpu', weights_only=True)
            ScratchTTS(len(vocab)).load_state_dict(saved['state'], strict=True)
            os.replace(temp, CKPT)
        finally:
            temp.unlink(missing_ok=True)
        state.update(status='completed', completed_at=now(), progress=1)
        report('Training completed; checkpoint saved and verified. Audio quality has not been evaluated.')
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', completed_at=now())
        report(state['error'])
        with log_path.open('a', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
    finally:
        lock.release()
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('VOICE_TORCH_THREADS', '2')))
    try:
        result = train(args.epochs)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Timeout:
        parser.exit(1, 'A training job is already running.\n')
