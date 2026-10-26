# Pipeline audit and verification

Date: 26 October 2026 (India). Base commit: `de51a17`.

## Findings and implemented fixes

| Finding | Fix |
| --- | --- |
| `MODEL_NOT_TRAINED` was an accurate missing-file error, but there was no reliable observable path from training start to readiness | Training has job IDs, terminal states, per-step progress, epoch losses and actionable failure logs; readiness actually deserializes and validates the model |
| Repeated `/train` requests and CLI runs could overwrite each other's checkpoint/logs | Shared OS-backed file lock across processes; 409 on duplicate API training; CLI exits clearly |
| Checkpoints were overwritten directly after each epoch | Versioned checkpoint is verified and atomically replaced after a successful complete run; previous valid model survives failures |
| No persisted completion/failure/interruption state | Atomic job state, exception reporting, traceback log, restart/interruption detection |
| WAV validation trusted header without checking actual payload | Check sample rate/channels/PCM width/compression/duration, truncated data, silence and streamed request size |
| Upload filenames/hand-edited metadata could lead to unsafe paths | Server UUID upload names, strict metadata names, confined generated paths and symlink rejection |
| Metadata used platform-default encoding, risking Malayalam corruption on Windows | Explicit UTF-8, NFC text normalization, atomic metadata updates with a dataset lock |
| Empty/control-only text and unknown characters were inadequately described | Validated transcript/text and unknown character errors with Unicode code points |
| Torch and torchaudio were independently unbounded | Matching pinned 2.6.0 releases, documented matching CPU-wheel install and repair, actionable import failure |
| Inverse mel solve could fail or give invalid spectra; reconstruction lacked shape/finite checks | Filterbank pseudoinverse, non-negative power spectrum, bounded exponentiation, finite shape checks and finite mono PCM output |
| Targets and durations were recomputed repeatedly | Compute training targets once per run; snapshot dataset; deterministic initialization and sample order |
| `/speak` could throw during checkpoint loading, model execution or saving | Explicit validated checkpoint load; consistent operation errors; concurrent synthesis rejection; length limit instead of silent duration truncation |
| Emotion options appeared selectable despite not being implemented | UI labels neutral-only; API rejects unsupported emotions |
| UI displayed raw JSON, swallowed connectivity failures and enabled premature synthesis | Human-readable errors, connectivity feedback, readiness gate, training logs/progress, dataset list and disabled busy actions |
| Recorder lacked safe lifecycle handling | Start/stop guards, time limit, WAV conversion, preview, AudioContext/track/object-URL cleanup |
| Cross-origin download attribute was not sufficient for reliable download | Fetch audio and download a local blob URL |
| Malayalam glyphs depended on operating-system fonts | Bundle Noto Sans Malayalam locally; no runtime Google Fonts request |
| Only two model-helper tests existed | Isolated backend failure and real-training tests plus full-browser integration test and CI build/test workflow |
| README included trailing UTF-16/NUL text and incomplete Windows startup guidance | UTF-8 rewrite with exact PowerShell commands using `python` rather than requiring `py` |

## Executed checks

Environment: Linux, Python 3.12.14, torch/torchaudio 2.6.0+cpu, Node 24.22.1, Next.js 15.5.28. Windows commands are supplied but were not executed on Windows.

- Python dependencies installed; imports for torch, torchaudio, FastAPI and Uvicorn succeeded; `pip check` reported no broken requirements.
- `python -m compileall -q backend tests`: passed.
- `python -m pytest -q`: **24 passed**. One third-party Starlette/httpx deprecation warning does not affect results.
- Tests include two 0.6-second synthetic 220 Hz sine clips, with Malayalam labels, trained for two epochs. These labels deliberately exercise Malayalam tokenization; the sine waves are **not spoken Malayalam** and do not validate acoustic/linguistic quality.
- Actual model optimization, atomic checkpoint save, safe reload, `/health` readiness, `/speak`, and `/audio` retrieval passed. Returned WAV header is mono PCM16 at 22,050 Hz; decoded samples are nonempty and nonzero.
- Failure tests cover malformed/truncated/stereo/wrong-rate/short/long/silent/oversized audio; blank/control/overlength transcripts; missing/corrupt/legacy/non-finite checkpoints; unknown characters; invalid speed/emotion/epochs; metadata traversal; generated-file symlinks; reconstruction exceptions; CORS; lock contention including another process; interrupted and failed jobs.
- A failed retraining attempt was verified to preserve an actually loadable previous checkpoint.
- Frontend production build and TypeScript checks passed.
- `npm run test:e2e`: **1 full-flow Chromium test passed** against real Uvicorn and Next.js servers. Verified simulated microphone → WAV conversion/preview, two synthetic uploads, training, readiness gate, Malayalam request, audio playback with advancing playback time, WAV download, readable unknown-character errors, no browser runtime errors, and no horizontal overflow at 390px width.
- `git diff --check`: passed. No WAV, MP3, checkpoint, `.env` or private voice data was tracked.

The initial install test exposed a mismatched torchaudio binary; reinstalling the exact matching CPU release resolved it. The browser CDN initially returned invalid archives; Playwright 1.51.1's official fallback download supplied the browser. These installation failures were resolved before the final checks.

## Changed files

- `backend/core.py`: validated data/text IO, original model, audio transforms/reconstruction and checkpoint validation.
- `backend/train.py`: lock, training job lifecycle, progress/logs, CPU/CUDA selection, atomic checkpoints.
- `backend/api.py`: all seven requested endpoints, structured errors, secure uploads and audio retrieval.
- `backend/requirements.txt`, `backend/requirements-dev.txt`: runtime and test dependencies.
- `frontend/app/page.tsx`, `frontend/app/layout.tsx`, `frontend/app/globals.css`: responsive working interface.
- `frontend/package.json`, `frontend/package-lock.json`: reproducible dependencies, bundled Malayalam font and browser-test command.
- `frontend/playwright.config.ts`, `frontend/tests/pipeline.spec.ts`: isolated real-server browser smoke test.
- `tests/conftest.py`, `tests/test_pipeline.py`: isolated backend integration/failure tests; original `tests/test_core.py` retained.
- `.gitignore`: checkpoints, temporary files and test outputs protected from tracking.
- `.github/workflows/test.yml`: automated backend tests and frontend build.
- `README.md`, `AUDIT.md`: setup, operation, audit evidence and limitations.

## Remaining limits

No actual user voice recordings were available or used. Real-voice intelligibility, similarity, pronunciation, naturalness, emotional speech, GPU behavior and physical microphone behavior remain unverified. Browser microphone testing uses a simulated device. No pretrained weights or third-party voice service was used.

The preserved research architecture uses uniform character-to-frame interpolation and a signal-processing vocoder; it lacks learned alignment, learned durations, speaker conditioning, phoneme processing, validation split and quality metrics. More data and compute alone do not guarantee natural speech. Every training run restarts from random weights. Dataset targets are cached in RAM; large datasets may exhaust memory. There is no pause/resume/cancel API or automatic disk-retention cleanup.

This remains a local single-user service. Loopback binding is intentional; authentication, authorization and deployment hardening are outside this repair. Do not expose it publicly. Existing legacy checkpoints require retraining; they are not silently treated as compatible.
