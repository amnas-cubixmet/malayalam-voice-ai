"""Synthetic sine waves validate plumbing ONLY, never speech quality or identity."""
import io
import json
import time
import wave

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
import api
import core
import train


def wav(rate=22050, channels=1, seconds=.6, silent=False):
    data = io.BytesIO()
    samples = np.zeros(int(rate * seconds)) if silent else .15 * np.sin(2 * np.pi * 220 * np.arange(int(rate * seconds)) / rate)
    with wave.open(data, 'wb') as f:
        f.setnchannels(channels); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes(np.repeat((samples * 32767).astype('<i2'), channels).tobytes())
    return data.getvalue()


def upload(client, text='നമസ്കാരം', raw=None):
    return client.post('/dataset', data={'text': text}, files={'audio': ('../../private.wav', wav() if raw is None else raw, 'audio/wav')})


def wait_job(client):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = client.get('/training').json()
        if not state['active']:
            return state
        time.sleep(.05)
    pytest.fail('Synthetic training timed out')


def test_complete_synthetic_pipeline():
    with TestClient(api.app) as client:
        assert not client.get('/health').json()['checkpoint']
        assert client.post('/speak', json={'text': 'നമസ്കാരം'}).status_code == 409
        assert client.post('/train').status_code == 400
        for text in ('നമസ്കാരം', 'നമസ്കാരം നമസ്കാരം'):
            response = upload(client, text)
            assert response.status_code == 201, response.text
            assert '/' not in response.json()['file']
        assert client.get('/dataset').json()['count'] == 2
        started = client.post('/train?epochs=2')
        assert started.status_code == 202, started.text
        state = wait_job(client)
        assert state['status'] == 'completed', state
        assert state['checkpoint'] and state['epoch'] == 2
        assert len(state['history']) == 2 and state['progress'] == 1
        assert core.CKPT.is_file()
        model, cp = core.load_checkpoint()
        assert cp['samples'] == 2 and model.training is False
        response = client.post('/speak', json={'text': 'നമസ്കാരം'})
        assert response.status_code == 200, response.text
        audio = client.get(response.json()['url'])
        assert audio.status_code == 200
        assert audio.headers['content-type'] == 'audio/wav'
        with wave.open(io.BytesIO(audio.content)) as f:
            assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (22050, 1, 2)
            samples = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2')
            assert len(samples) > 0 and np.abs(samples).max() > 0
        response = client.post('/speak', json={'text': 'hello'})
        assert response.status_code == 400 and response.json()['detail']['code'] == 'UNSUPPORTED_CHARACTERS'
        assert 'U+' in response.json()['detail']['message']
        # A failed replacement run must preserve a genuinely loadable previous model.
        original = core.CKPT.read_bytes()
        core.recording_path(core.load_rows()[0]['file']).unlink()
        assert client.post('/train?epochs=1').status_code == 202
        assert wait_job(client)['status'] == 'failed'
        assert core.CKPT.read_bytes() == original
        assert client.get('/health').json()['checkpoint']


@pytest.mark.parametrize('raw', [b'not a wav', wav(rate=44100), wav(channels=2), wav(seconds=.1), wav(seconds=30.1), wav(silent=True), wav()[:-100]])
def test_invalid_audio(raw):
    with TestClient(api.app) as client:
        response = upload(client, raw=raw)
        assert response.status_code == 400, response.text
        assert client.get('/dataset').json()['count'] == 0


@pytest.mark.parametrize('text', ['  ', '\x00bad', '!', 'a' * 1001])
def test_invalid_transcripts(text):
    with TestClient(api.app) as client:
        assert upload(client, text).status_code == 400


def test_oversized_audio():
    with TestClient(api.app) as client:
        assert upload(client, raw=b'a' * (core.MAX_BYTES + 1)).status_code == 413


def test_training_lock_and_interrupted_state():
    with TestClient(api.app) as client:
        lock = train.acquire_training_lock()
        try:
            assert client.post('/train').status_code == 409
            with pytest.raises(train.Timeout):
                train.train(epochs=1)
        finally:
            lock.release()
        core.atomic_json(train.STATUS, {'status': 'running', 'history': []})
        assert client.get('/training').json()['status'] == 'interrupted'


def test_failed_training_preserves_previous_file():
    with TestClient(api.app) as client:
        upload(client); upload(client)
        core.CKPT.parent.mkdir()
        core.CKPT.write_bytes(b'previous-file')
        core.recording_path(core.load_rows()[0]['file']).unlink()
        assert client.post('/train?epochs=1').status_code == 202
        state = wait_job(client)
        assert state['status'] == 'failed' and state['error'] and state['log_tail']
        assert not state['checkpoint']
        assert core.CKPT.read_bytes() == b'previous-file'


def test_corrupt_and_legacy_checkpoints():
    core.CKPT.parent.mkdir()
    with TestClient(api.app) as client:
        for content in (b'corrupt pickle',):
            core.CKPT.write_bytes(content)
            assert not client.get('/health').json()['checkpoint']
            assert client.post('/speak', json={'text':'നമസ്കാരം'}).json()['detail']['code'] == 'CHECKPOINT_INVALID'
        torch.save({'vocab': {}, 'epoch': 1}, core.CKPT)
        assert not client.get('/health').json()['checkpoint']


def test_paths_and_request_validation(tmp_path):
    with TestClient(api.app) as client:
        for filename in ('metadata.jsonl', '..secret.wav', 'recording.wav'):
            assert client.get('/audio/' + filename).status_code == 400
        core.OUTPUT.mkdir()
        (core.OUTPUT / ('a'*32 + '.wav')).symlink_to(tmp_path / 'outside.wav')
        assert client.get('/audio/' + 'a'*32 + '.wav').status_code == 404
        for body in ({'text':'a','speed':0}, {'text':''}, {'text':'a'*501}):
            response = client.post('/speak', json=body)
            assert response.status_code == 422
            assert response.json()['detail']['code'] == 'INVALID_REQUEST'
        assert client.post('/speak', json={'text':'a','emotion':'happy'}).status_code == 400
        assert client.post('/train?epochs=0').status_code == 422
        core.META.write_text(json.dumps({'file':'../secret.wav','text':'hello'}) + '\n')
        assert client.get('/dataset').status_code == 409


def test_cors():
    with TestClient(api.app) as client:
        response = client.options('/speak', headers={'Origin':'http://localhost:3000','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
        assert response.status_code == 200
        assert response.headers['access-control-allow-origin'] == 'http://localhost:3000'
        assert client.options('/speak', headers={'Origin':'https://evil.example','Access-Control-Request-Method':'POST'}).status_code == 400


def test_mel_reconstruction_dimensions():
    audio = core.read_wav(io.BytesIO(wav()))
    mel = core.mel_spec(audio)
    assert mel.shape[0] == 80 and torch.isfinite(mel).all()
    restored = core.reconstruct(mel, iterations=2)
    assert restored.ndim == 1 and np.isfinite(restored).all()
    with pytest.raises(ValueError):
        core.reconstruct(torch.zeros(1, 80, 20))


def test_streamed_request_limit():
    with TestClient(api.app) as client:
        response = client.post('/dataset', content=b'x' * (core.MAX_BYTES + 65537), headers={'Content-Type':'multipart/form-data; boundary=test'})
        assert response.status_code == 413


def test_atomic_status_write_retries_transient_windows_lock(tmp_path, monkeypatch):
    path = tmp_path / 'training_status.json'
    original_replace = core.os.replace
    calls = 0

    def temporarily_locked(source, destination):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(5, 'Access is denied', str(destination))
        return original_replace(source, destination)

    monkeypatch.setattr(core.os, 'replace', temporarily_locked)
    monkeypatch.setattr(core.time, 'sleep', lambda _seconds: None)
    core.atomic_json(path, {'status': 'running', 'epoch': 6})
    assert calls == 3
    assert json.loads(path.read_text(encoding='utf-8')) == {'status': 'running', 'epoch': 6}
    assert list(tmp_path.glob('*.tmp')) == []


def test_nonfinite_checkpoint_and_reconstruction_failure(monkeypatch):
    with TestClient(api.app) as client:
        upload(client); upload(client)
        client.post('/train?epochs=1')
        assert wait_job(client)['status'] == 'completed'
        def fail(*args):
            raise RuntimeError('test reconstruction failure')
        monkeypatch.setattr(api, 'reconstruct', fail)
        response = client.post('/speak', json={'text':'നമസ്കാരം'})
        assert response.status_code == 500 and response.json()['detail']['code'] == 'SYNTHESIS_FAILED'
        cp = torch.load(core.CKPT, weights_only=True)
        cp['state']['embed.weight'][0, 0] = float('nan')
        torch.save(cp, core.CKPT)
        assert not client.get('/health').json()['checkpoint']


def test_lock_across_processes():
    import os
    import subprocess
    import sys
    from pathlib import Path
    lock = train.acquire_training_lock()
    try:
        env = {**os.environ, 'VOICE_DATA_DIR': str(core.DATA)}
        result = subprocess.run([sys.executable, str(Path(train.__file__)), '--epochs', '1'], env=env, capture_output=True, text=True, timeout=20)
        assert result.returncode == 1
        assert 'already running' in result.stderr
    finally:
        lock.release()
