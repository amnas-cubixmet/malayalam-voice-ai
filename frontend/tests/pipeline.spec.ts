import {test, expect} from '@playwright/test';

// This is a synthetic 220 Hz sine, NOT a recording of Malayalam speech.
function sineWav() {
  const frames = 13230;
  const b = Buffer.alloc(44 + frames * 2);
  b.write('RIFF', 0); b.writeUInt32LE(b.length - 8, 4); b.write('WAVEfmt ', 8);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(22050, 24); b.writeUInt32LE(44100, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(frames * 2, 40);
  for (let i = 0; i < frames; i++) b.writeInt16LE(Math.round(4000 * Math.sin(2 * Math.PI * 220 * i / 22050)), 44 + i * 2);
  return b;
}

test('recording conversion, synthetic dataset, training, playback, download and mobile layout', async ({page}, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await expect(page.getByText('No trained checkpoint.', {exact: false})).toBeVisible();
  await expect(page.getByRole('button', {name: 'Generate voice'})).toBeDisabled();
  await page.getByRole('button', {name: 'Start microphone'}).click();
  await expect(page.getByRole('button', {name: 'Stop recording'})).toBeEnabled();
  await page.waitForTimeout(850); // collect enough fake microphone frames to decode
  await page.getByRole('button', {name: 'Stop recording'}).click();
  await expect(page.getByLabel('Recording preview')).toBeVisible();
  await expect(page.getByText('Selected: recording.wav')).toBeVisible();
  // Do not upload the fake microphone recording; training uses explicitly synthetic fixtures below.
  for (let i = 0; i < 2; i++) {
    await page.getByLabel('Or choose a WAV file').setInputFiles({name: `synthetic-${i}.wav`, mimeType: 'audio/wav', buffer: sineWav()});
    await page.getByLabel('Exact Malayalam transcript').fill(i ? 'നമസ്കാരം നമസ്കാരം' : 'നമസ്കാരം');
    await page.getByRole('button', {name: 'Add to dataset'}).click();
    await expect(page.getByRole('heading', {name: `Dataset: ${i + 1} samples`})).toBeVisible();
    await expect(page.getByLabel('Or choose a WAV file')).toHaveValue('');
  }
  await page.getByLabel('Epochs (1–1000)').fill('2');
  await page.getByRole('button', {name: 'Start training'}).click();
  await expect(page.getByText('Model ready — checkpoint loaded and validated.')).toBeVisible({timeout: 30000});
  await page.getByLabel('Malayalam text', {exact: true}).fill('നമസ്കാരം');
  await page.getByRole('button', {name: 'Generate voice'}).click();
  const player = page.getByLabel('Generated speech');
  await expect(player).toBeVisible();
  await expect.poll(() => player.evaluate((e: HTMLAudioElement) => e.readyState)).toBeGreaterThanOrEqual(2);
  await player.evaluate((e: HTMLAudioElement) => e.play());
  await expect.poll(() => player.evaluate((e: HTMLAudioElement) => e.currentTime)).toBeGreaterThan(0);
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Download WAV'}).click();
  expect((await downloadEvent).suggestedFilename()).toBe('malayalam-voice.wav');
  await page.getByLabel('Malayalam text', {exact: true}).fill('hello');
  await page.getByRole('button', {name: 'Generate voice'}).click();
  await expect(page.getByRole('alert').filter({hasText: 'Characters absent'})).toContainText('Characters absent from training data');
  await expect(page.getByRole('status')).toHaveText('Ready to retry. See the error below.');
  await page.setViewportSize({width: 390, height: 844});
  await page.evaluate(() => document.fonts.ready);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path: testInfo.outputPath('mobile.png'), fullPage: true});
  expect(errors).toEqual([]);
});
