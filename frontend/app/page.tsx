"use client";
import { useCallback, useEffect, useRef, useState } from "react";
const API = (
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");
type Sample = { file: string; text: string; duration?: number };
type Training = {
  status: string;
  active: boolean;
  checkpoint: boolean;
  checkpoint_error?: string;
  epoch?: number;
  epochs?: number;
  progress?: number;
  loss?: number;
  error?: string;
  history: { epoch: number; loss: number }[];
  log_tail?: string[];
};
async function request(path: string, options?: RequestInit) {
  let response: Response;
  try {
    response = await fetch(API + path, { ...options, cache: "no-store" });
  } catch {
    throw new Error(
      "Cannot reach the backend. Start FastAPI on port 8000 and check NEXT_PUBLIC_API_URL.",
    );
  }
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : data.detail?.message || `Request failed (${response.status}).`,
    );
  return data;
}
async function wavFile(blob: Blob): Promise<File> {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer());
    const offline = new OfflineAudioContext(
      1,
      Math.ceil(decoded.duration * 22050),
      22050,
    );
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const pcm = (await offline.startRendering()).getChannelData(0);
    const buffer = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(buffer);
    const str = (at: number, s: string) => {
      for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i));
    };
    str(0, "RIFF");
    view.setUint32(4, 36 + pcm.length * 2, true);
    str(8, "WAVE");
    str(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 22050, true);
    view.setUint32(28, 44100, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    str(36, "data");
    view.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++)
      view.setInt16(
        44 + i * 2,
        Math.max(-32768, Math.min(32767, pcm[i] * 32767)),
        true,
      );
    return new File([buffer], "recording.wav", { type: "audio/wav" });
  } finally {
    await context.close();
  }
}
export default function Home() {
  const [text, setText] = useState("എല്ലാവർക്കും നമസ്കാരം"),
    [transcript, setTranscript] = useState("");
  const [message, setMessage] = useState(
    "Record short, clean sentences and enter exactly what you said.",
  );
  const [error, setError] = useState(""),
    [connectionError, setConnectionError] = useState("");
  const [url, setUrl] = useState(""),
    [preview, setPreview] = useState("");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [training, setTraining] = useState<Training>({
    status: "idle",
    active: false,
    checkpoint: false,
    history: [],
  });
  const [speed, setSpeed] = useState(1),
    [epochs, setEpochs] = useState(20);
  const [file, setFile] = useState<File | null>(null),
    [busy, setBusy] = useState("");
  const [recording, setRecording] = useState(false),
    [micPending, setMicPending] = useState(false);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const recorder = useRef<MediaRecorder | null>(null),
    stream = useRef<MediaStream | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null),
    mounted = useRef(true),
    polling = useRef(false);
  const refresh = useCallback(async () => {
    if (polling.current) return;
    polling.current = true;
    try {
      const [dataset, status] = await Promise.all([
        request("/dataset"),
        request("/training"),
      ]);
      if (mounted.current) {
        setSamples(dataset.samples);
        setTraining(status);
        setConnectionError("");
      }
    } catch (e) {
      if (mounted.current) setConnectionError((e as Error).message);
    } finally {
      polling.current = false;
    }
  }, []);
  useEffect(() => {
    mounted.current = true;
    void refresh();
    const id = setInterval(refresh, 2000);
    return () => {
      mounted.current = false;
      clearInterval(id);
      if (timer.current) clearTimeout(timer.current);
      if (recorder.current?.state === "recording") recorder.current.stop();
      stream.current?.getTracks().forEach((t) => t.stop());
    };
  }, [refresh]);
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const u = URL.createObjectURL(file);
    setPreview(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  async function action(name: string, fn: () => Promise<void>) {
    setBusy(name);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
      setMessage("Ready to retry. See the error below.");
    } finally {
      setBusy("");
    }
  }
  async function start() {
    setError("");
    setMicPending(true);
    try {
      if (
        !navigator.mediaDevices?.getUserMedia ||
        typeof MediaRecorder === "undefined"
      )
        throw new Error(
          "Recording requires a supported browser on localhost or HTTPS. You can also upload a WAV.",
        );
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!mounted.current) {
        media.getTracks().forEach((t) => t.stop());
        return;
      }
      stream.current = media;
      const r = new MediaRecorder(media);
      recorder.current = r;
      const chunks: Blob[] = [];
      r.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      r.onerror = () => {
        setError("Microphone recording failed. Try again or upload a WAV.");
        media.getTracks().forEach((t) => t.stop());
        setRecording(false);
      };
      r.onstop = async () => {
        if (timer.current) clearTimeout(timer.current);
        media.getTracks().forEach((t) => t.stop());
        if (!mounted.current) return;
        setRecording(false);
        await action("conversion", async () => {
          const converted = await wavFile(
            new Blob(chunks, { type: r.mimeType }),
          );
          if (!mounted.current) return;
          setFile(converted);
          setMessage(
            "Recording ready. Listen, enter the exact transcript, then add it to the dataset.",
          );
        });
      };
      r.start();
      setRecording(true);
      setMessage(
        "Recording… Stop after one sentence (automatically stops after 29 seconds).",
      );
      timer.current = setTimeout(() => {
        if (r.state === "recording") r.stop();
      }, 29000);
    } catch (e) {
      stream.current?.getTracks().forEach((t) => t.stop());
      setError((e as Error).message);
    } finally {
      setMicPending(false);
    }
  }
  const unavailable = !!busy || recording || micPending;
  return (
    <main>
      <header>
        <p className="eyebrow">LOCAL · SCRATCH-TRAINED · NO VOICE API</p>
        <h1>Malayalam Voice AI</h1>
        <p>
          Record your dataset, train a small research model, and generate a WAV.
        </p>
      </header>
      <aside className="notice">
        This baseline may produce noisy or unintelligible audio, even after
        training. A saved checkpoint means the model can run; it does not prove
        natural speech or voice cloning.
      </aside>
      <div role="status" aria-live="polite" className="status">
        {message}
      </div>
      {(error || connectionError) && (
        <div role="alert" className="error">
          {error || connectionError}
        </div>
      )}
      <section>
        <h2>1. Record & upload</h2>
        <p>
          Mono, 22,050 Hz, 16-bit PCM WAV · 0.5–30 seconds · maximum 15 MB. The
          microphone recorder converts to this format.
        </p>
        <div className="buttons">
          <button disabled={unavailable} onClick={start}>
            {micPending ? "Requesting microphone…" : "Start microphone"}
          </button>
          <button
            disabled={!recording}
            onClick={() => {
              if (recorder.current?.state === "recording")
                recorder.current.stop();
            }}
          >
            Stop recording
          </button>
        </div>
        <label htmlFor="upload">Or choose a WAV file</label>
        <input
          ref={fileInput}
          id="upload"
          type="file"
          accept=".wav,audio/wav"
          disabled={unavailable}
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        {file && <p>Selected: {file.name}</p>}
        {preview && (
          <audio aria-label="Recording preview" controls src={preview} />
        )}
        <label htmlFor="transcript">Exact Malayalam transcript</label>
        <textarea
          id="transcript"
          maxLength={1000}
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          placeholder="നിങ്ങൾ പറഞ്ഞ വാക്കുകൾ ഇവിടെ എഴുതുക"
        />
        <button
          disabled={
            unavailable || !file || !transcript.trim() || !!connectionError
          }
          onClick={() =>
            action("upload", async () => {
              const f = new FormData();
              f.append("audio", file!);
              f.append("text", transcript);
              const d = await request("/dataset", { method: "POST", body: f });
              setMessage(d.message);
              setFile(null);
              if (fileInput.current) fileInput.current.value = "";
              setTranscript("");
              await refresh();
            })
          }
        >
          {busy === "upload" ? "Uploading…" : "Add to dataset"}
        </button>
        <h3>Dataset: {samples.length} samples</h3>
        {!samples.length ? (
          <p>
            No recordings yet. Add at least two to test training; useful speech
            requires substantially more real recordings.
          </p>
        ) : (
          <ul className="samples">
            {samples.map((s) => (
              <li key={s.file}>
                <span lang="ml">{s.text}</span>
                <small>
                  {s.duration ? `${s.duration.toFixed(1)} seconds · ` : ""}
                  {s.file}
                </small>
              </li>
            ))}
          </ul>
        )}
      </section>
      <section>
        <h2>2. Train from scratch</h2>
        <p>
          Each run starts from random weights. The previous working checkpoint
          stays available until a new run completes successfully.
        </p>
        <label htmlFor="epochs">Epochs (1–1000)</label>
        <input
          id="epochs"
          type="number"
          min={1}
          max={1000}
          value={epochs}
          onChange={(e) => setEpochs(Number(e.target.value))}
        />
        <button
          disabled={
            unavailable ||
            training.active ||
            samples.length < 2 ||
            !Number.isInteger(epochs) ||
            epochs < 1 ||
            epochs > 1000 ||
            !!connectionError
          }
          onClick={() =>
            action("train", async () => {
              const d = await request(`/train?epochs=${epochs}`, {
                method: "POST",
              });
              setMessage(d.message);
              await refresh();
            })
          }
        >
          {training.active ? "Training in progress…" : "Start training"}
        </button>
        <p>
          Status: <strong>{training.status}</strong> · Epoch{" "}
          {training.epoch || 0}/{training.epochs || "—"}
        </p>
        <progress
          aria-label="Training progress"
          max={1}
          value={training.progress || 0}
        />
        <p>
          {Math.round((training.progress || 0) * 100)}% · Latest loss:{" "}
          {training.loss == null ? "—" : training.loss.toFixed(5)}
        </p>
        {training.error && (
          <p className="error" role="alert">
            {training.error}
          </p>
        )}
        <details>
          <summary>Epoch loss history</summary>
          <ul>
            {training.history.map((h) => (
              <li key={h.epoch}>
                Epoch {h.epoch}: {h.loss.toFixed(6)}
              </li>
            ))}
          </ul>
        </details>
        <details open>
          <summary>Training logs</summary>
          <pre>{training.log_tail?.join("\n") || "No training logs yet."}</pre>
        </details>
      </section>
      <section>
        <h2>3. Generate Malayalam audio</h2>
        <p
          className={
            training.checkpoint && !connectionError ? "ready" : "notice"
          }
        >
          {connectionError
            ? "Model readiness unknown — reconnect the backend."
            : training.checkpoint
              ? "Model ready — checkpoint loaded and validated."
              : training.checkpoint_error ||
                "Model not ready. Upload recordings and complete training first."}
        </p>
        <label htmlFor="speech">Malayalam text</label>
        <textarea
          id="speech"
          maxLength={500}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <label htmlFor="speed">
          Speed: {speed.toFixed(1)}× (duration heuristic)
        </label>
        <input
          id="speed"
          type="range"
          min="0.6"
          max="1.5"
          step="0.1"
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
        />
        <p>Emotion: neutral only. Emotion conditioning is not implemented.</p>
        <button
          disabled={
            unavailable ||
            !training.checkpoint ||
            !text.trim() ||
            !!connectionError
          }
          onClick={() =>
            action("speak", async () => {
              setUrl("");
              setMessage("Generating audio…");
              const d = await request("/speak", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, speed, emotion: "neutral" }),
              });
              setUrl(API + d.url);
              setMessage(d.note);
            })
          }
        >
          {busy === "speak" ? "Generating…" : "Generate voice"}
        </button>
        {url && (
          <div>
            <audio aria-label="Generated speech" controls src={url} />
            <button
              disabled={!!busy}
              onClick={() =>
                action("download", async () => {
                  const r = await fetch(url);
                  if (!r.ok)
                    throw new Error(
                      "Audio download failed. Generate it again.",
                    );
                  const u = URL.createObjectURL(await r.blob());
                  const a = document.createElement("a");
                  a.href = u;
                  a.download = "malayalam-voice.wav";
                  a.click();
                  setTimeout(() => URL.revokeObjectURL(u), 1000);
                })
              }
            >
              Download WAV
            </button>
          </div>
        )}
      </section>
      <footer>
        Local research prototype. Record only voices you have permission to use.
        Keep voice recordings and checkpoints private.
      </footer>
    </main>
  );
}
