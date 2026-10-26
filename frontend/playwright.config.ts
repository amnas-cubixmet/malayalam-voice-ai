import {defineConfig} from '@playwright/test';
import {mkdtempSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';

// Never use the user's real recordings or checkpoints for browser tests.
const scratch = mkdtempSync(join(tmpdir(), 'malayalam-voice-e2e-'));
const python = process.env.VOICE_TEST_PYTHON || resolve('..', '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
export default defineConfig({
  testDir: './tests', workers: 1, timeout: 60000,
  use: {baseURL: 'http://localhost:3000', browserName: 'chromium', permissions: ['microphone'],
    launchOptions: {args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']}},
  webServer: [
    {command: `"${python}" -m uvicorn api:app --app-dir ../backend --host 127.0.0.1 --port 8000`, url: 'http://127.0.0.1:8000/health', reuseExistingServer: false,
     env: {VOICE_DATA_DIR: join(scratch, 'data'), VOICE_CHECKPOINT_DIR: join(scratch, 'checkpoints')}, timeout: 30000},
    {command: 'npm run dev -- --hostname localhost', url: 'http://localhost:3000', reuseExistingServer: false,
     env: {NEXT_PUBLIC_API_URL: 'http://127.0.0.1:8000'}, timeout: 60000},
  ],
});
