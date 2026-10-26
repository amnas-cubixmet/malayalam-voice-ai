"""Local-only FastAPI service. Bind to loopback; no authentication is provided."""
import io
import json
import logging
import os
import re
import threading
import uuid
import wave
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from filelock import FileLock, Timeout
from pydantic import BaseModel, Field

from core import (DATA, AUDIO, META, OUTPUT, MAX_BYTES, SR, validate_text, read_wav,
                  load_rows, checkpoint_status, load_checkpoint, encode, reconstruct, save_wav, atomic_json)
from train import train, acquire_training_lock, read_status, DATA_LOCK, STATUS, STATUS_LOCK, now

logger = logging.getLogger(__name__)
workers = set()
synthesis_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(int(os.environ.get('VOICE_TORCH_THREADS', '2')))
    yield
    # Normal shutdown waits for an active training job; abrupt termination is reported on restart.
    for worker in list(workers):
        worker.join()


class UploadLimitMiddleware:
    """Bound the streamed body before multipart parsing spools it to disk."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        limit = MAX_BYTES + 65536 if scope['path'] == '/dataset' else 65536
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > limit:
                raise HTTPException(413, detail={'code': 'REQUEST_TOO_LARGE', 'message': 'Upload exceeds the request size limit (15 MB audio maximum).'})
            return message

        await self.app(scope, limited_receive, send)


app = FastAPI(title='Malayalam Voice AI — Scratch Baseline', lifespan=lifespan)
app.add_middleware(UploadLimitMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


def error(status, code, message):
    return HTTPException(status, detail={'code': code, 'message': message})


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc):
    message = '; '.join(f'{".".join(str(x) for x in e["loc"])}: {e["msg"]}' for e in exc.errors())
    return JSONResponse(status_code=422, content={'detail': {'code': 'INVALID_REQUEST', 'message': message}})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc):
    logger.exception('Request failed: %s', request.url.path)
    return JSONResponse(status_code=500, content={'detail': {'code': 'INTERNAL_ERROR', 'message': 'The operation failed. See the backend terminal for details.'}})


class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    emotion: str = 'neutral'
    speed: float = Field(default=1.0, ge=0.6, le=1.5, allow_inf_nan=False)


def dataset_rows():
    try:
        with FileLock(DATA_LOCK, timeout=10):
            return load_rows()
    except Exception as exc:
        raise error(409, 'DATASET_INVALID', 'Dataset metadata is invalid or unavailable. Check data/metadata.jsonl and the backend logs.') from exc


@app.get('/health')
def health():
    return {'ok': True, **checkpoint_status()}


@app.get('/dataset')
def dataset():
    DATA.mkdir(parents=True, exist_ok=True)
    rows = dataset_rows()
    return {'samples': rows, 'count': len(rows)}


@app.post('/dataset', status_code=201)
async def add_recording(text: str = Form(...), audio: UploadFile = File(...)):
    try:
        text = validate_text(text)
        raw = await audio.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise error(413, 'AUDIO_TOO_LARGE', 'WAV upload must be no larger than 15 MB.')
        try:
            samples = read_wav(io.BytesIO(raw))
        except (ValueError, wave.Error, EOFError) as exc:
            raise error(400, 'INVALID_AUDIO', str(exc) or 'Invalid WAV file.') from exc
        AUDIO.mkdir(parents=True, exist_ok=True)
        name = uuid.uuid4().hex + '.wav'
        row = {'file': name, 'text': text, 'duration': round(len(samples) / SR, 3)}
        with FileLock(DATA_LOCK, timeout=10):
            rows = load_rows()
            path = AUDIO / name
            path.write_bytes(raw)
            temp = META.with_suffix('.tmp')
            try:
                temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows + [row]), encoding='utf-8')
                os.replace(temp, META)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            finally:
                temp.unlink(missing_ok=True)
        return {**row, 'count': len(rows) + 1, 'message': 'Recording and transcript saved.'}
    except ValueError as exc:
        raise error(400, 'INVALID_DATASET', str(exc)) from exc
    finally:
        await audio.close()


@app.post('/train', status_code=202)
def start_training(epochs: int = Query(default=20, ge=1, le=1000)):
    try:
        lock = acquire_training_lock()
    except Timeout:
        raise error(409, 'TRAINING_ACTIVE', 'A training job is already running. Follow its progress below.')
    try:
        rows = dataset_rows()
        if len(rows) < 2:
            raise error(400, 'DATASET_TOO_SMALL', 'Upload at least two labeled recordings before training.')
        job_id = uuid.uuid4().hex
        with FileLock(STATUS_LOCK):
            atomic_json(STATUS, {'job_id': job_id, 'status': 'starting', 'epochs': epochs, 'epoch': 0,
                                 'progress': 0, 'history': [], 'log_tail': [], 'error': None, 'started_at': now()})
        def run():
            try:
                train(epochs, lock=lock, job_id=job_id)
            except Exception:
                logger.exception('Training job %s failed', job_id)
            finally:
                workers.discard(threading.current_thread())
        worker = threading.Thread(target=run, name=f'training-{job_id}', daemon=False)
        workers.add(worker)
        worker.start()
    except Exception:
        lock.release()
        raise
    return {'job_id': job_id, 'status': 'starting', 'message': 'Training started. Follow progress and errors in /training.'}


@app.get('/training')
def training():
    DATA.mkdir(parents=True, exist_ok=True)
    return {**read_status(), **checkpoint_status(), 'recordings': len(dataset_rows())}


@app.post('/speak')
def speak(request: Speech):
    if request.emotion != 'neutral':
        raise error(400, 'EMOTION_UNSUPPORTED', 'Emotion conditioning is not implemented. Use neutral.')
    if not synthesis_lock.acquire(blocking=False):
        raise error(409, 'SYNTHESIS_ACTIVE', 'Another audio generation is running. Try again when it finishes.')
    try:
        try:
            text = validate_text(request.text, limit=500)
        except ValueError as exc:
            raise error(400, 'INVALID_TEXT', str(exc)) from exc
        try:
            model, cp = load_checkpoint()
        except FileNotFoundError as exc:
            raise error(409, 'MODEL_NOT_TRAINED', str(exc)) from exc
        except Exception as exc:
            logger.exception('Checkpoint loading failed')
            raise error(409, 'CHECKPOINT_INVALID', 'The local checkpoint is corrupt or incompatible. Retrain with this version.') from exc
        try:
            tokens = encode(text, cp['vocab'])
        except ValueError as exc:
            raise error(400, 'UNSUPPORTED_CHARACTERS', str(exc)) from exc
        frames = max(40, int(len(tokens) * cp['frames_per_char'] / request.speed))
        if frames > 3000:
            raise error(400, 'SPEECH_TOO_LONG', 'Estimated speech exceeds 35 seconds. Generate shorter sentences separately.')
        try:
            with torch.inference_mode():
                mel = model(tokens, frames)
                audio = reconstruct(mel)
            OUTPUT.mkdir(parents=True, exist_ok=True)
            output = OUTPUT / (uuid.uuid4().hex + '.wav')
            save_wav(output, audio)
        except Exception as exc:
            logger.exception('Synthesis failed')
            raise error(500, 'SYNTHESIS_FAILED', 'Audio reconstruction failed. See backend logs; retrain if model output is invalid.') from exc
        return {'url': '/audio/' + output.name, 'sample_rate': SR, 'duration': len(audio) / SR,
                'emotion_supported': False, 'note': 'Generated by the scratch baseline. Intelligibility and voice similarity are not guaranteed.'}
    finally:
        synthesis_lock.release()


@app.get('/audio/{filename}')
def get_audio(filename: str):
    if not re.fullmatch(r'[0-9a-f]{32}\.wav', filename):
        raise error(400, 'INVALID_FILENAME', 'Invalid generated-audio filename.')
    path = OUTPUT / filename
    if path.is_symlink() or path.resolve().parent != OUTPUT.resolve() or not path.is_file():
        raise error(404, 'AUDIO_NOT_FOUND', 'Generated audio was not found.')
    return FileResponse(path, media_type='audio/wav', filename=filename, content_disposition_type='inline')
