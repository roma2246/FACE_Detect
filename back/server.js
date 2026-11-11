// Node-бэкенд: инференс модели эмоций (ONNX Runtime), сбор своих снимков и запуск дообучения.
import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3001;
const MODELS = path.join(__dirname, 'models');
const ML = path.join(__dirname, 'ml');
const CUSTOM = path.join(ML, 'data', 'custom');
const PYTHON = fs.existsSync(path.join(ML, '.venv', 'bin', 'python'))
  ? path.join(ML, '.venv', 'bin', 'python')
  : 'python3';

// Порядок должен совпадать с CLASSES в ml/model.py
export const CLASSES = ['happy', 'sad', 'angry', 'surprise', 'neutral'];
const FACE = 48 * 48;

// ---------- Модель ----------
let session = null;
let modelInfo = { name: null, finetuned: false, loadedAt: null };

async function loadModel() {
  const ft = path.join(MODELS, 'emotion_cnn_ft.onnx');
  const base = path.join(MODELS, 'emotion_cnn.onnx');
  const file = fs.existsSync(ft) ? ft : base;
  if (!fs.existsSync(file)) {
    session = null;
    console.warn('Модель не найдена. Сначала обучите: cd ml && .venv/bin/python train.py');
    return;
  }
  session = await ort.InferenceSession.create(file);
  modelInfo = { name: path.basename(file), finetuned: file === ft, loadedAt: new Date().toISOString() };
  console.log('Модель загружена:', modelInfo.name);
}

// Выравнивание гистограммы — точная копия PIL.ImageOps.equalize (используется при обучении).
function equalize(src, off) {
  const h = new Uint32Array(256);
  for (let i = 0; i < FACE; i++) h[src[off + i]]++;
  let last = 255;
  while (last > 0 && h[last] === 0) last--;
  const step = Math.floor((FACE - h[last]) / 255);
  const out = new Float32Array(FACE);
  if (step === 0) {
    for (let i = 0; i < FACE; i++) out[i] = src[off + i] / 255;
    return out;
  }
  const lut = new Uint8Array(256);
  let n = Math.floor(step / 2);
  for (let i = 0; i < 256; i++) {
    lut[i] = Math.min(255, Math.floor(n / step));
    n += h[i];
  }
  for (let i = 0; i < FACE; i++) out[i] = lut[src[off + i]] / 255;
  return out;
}

function softmax(logits, off, k) {
  let max = -Infinity;
  for (let i = 0; i < k; i++) max = Math.max(max, logits[off + i]);
  const e = [];
  let s = 0;
  for (let i = 0; i < k; i++) { e.push(Math.exp(logits[off + i] - max)); s += e[i]; }
  return e.map((v) => v / s);
}

// faces: Uint8Array из N подряд идущих серых лиц 48x48
async function predict(faces) {
  const n = Math.floor(faces.length / FACE);
  if (!session || n === 0) return [];
  const input = new Float32Array(n * FACE);
  for (let i = 0; i < n; i++) input.set(equalize(faces, i * FACE), i * FACE);
  const tensor = new ort.Tensor('float32', input, [n, 1, 48, 48]);
  const out = await session.run({ input: tensor });
  const logits = out.logits.data;
  const k = CLASSES.length;
  return Array.from({ length: n }, (_, i) => softmax(logits, i * k, k));
}

// ---------- HTTP API ----------
const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' }));

app.get('/api/health', (_req, res) => res.json({ ok: true, model: modelInfo, ready: !!session }));

app.get('/api/model', (_req, res) => {
  let metrics = null;
  try { metrics = JSON.parse(fs.readFileSync(path.join(ML, 'reports', 'metrics.json'), 'utf8')); } catch { /* нет отчёта */ }
  res.json({ ...modelInfo, ready: !!session, classes: CLASSES, testAcc: metrics?.test_acc ?? null });
});

app.post('/api/model/reset', async (_req, res) => {
  for (const f of ['emotion_cnn_ft.onnx', 'emotion_cnn_ft.pth']) fs.rmSync(path.join(MODELS, f), { force: true });
  await loadModel();
  res.json(modelInfo);
});

function sampleStats() {
  const stats = {};
  for (const c of CLASSES) {
    const dir = path.join(CUSTOM, c);
    stats[c] = fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => f.endsWith('.png')).length : 0;
  }
  return stats;
}

app.get('/api/samples', (_req, res) => res.json(sampleStats()));

// { label: 'happy', images: ['data:image/png;base64,...', ...] }
app.post('/api/samples', (req, res) => {
  const { label, images } = req.body || {};
  if (!CLASSES.includes(label) || !Array.isArray(images)) return res.status(400).json({ error: 'bad request' });
  const dir = path.join(CUSTOM, label);
  fs.mkdirSync(dir, { recursive: true });
  let saved = 0;
  for (const img of images.slice(0, 200)) {
    const m = /^data:image\/png;base64,(.+)$/.exec(img);
    if (!m) continue;
    fs.writeFileSync(path.join(dir, `${Date.now()}_${saved}.png`), Buffer.from(m[1], 'base64'));
    saved++;
  }
  res.json({ saved, stats: sampleStats() });
});

app.delete('/api/samples', (_req, res) => {
  fs.rmSync(CUSTOM, { recursive: true, force: true });
  res.json(sampleStats());
});

// Дообучение запускается как дочерний Python-процесс, прогресс читаем из stdout (JSON-строки).
let job = { running: false, log: [], result: null, error: null };

app.post('/api/finetune', (req, res) => {
  if (job.running) return res.status(409).json({ error: 'Уже идёт дообучение' });
  const epochs = String(Math.min(30, Math.max(1, Number(req.body?.epochs) || 8)));
  job = { running: true, log: [], result: null, error: null, startedAt: Date.now() };
  const p = spawn(PYTHON, ['finetune.py', '--epochs', epochs], { cwd: ML, env: { ...process.env, PYTHONWARNINGS: 'ignore' } });
  let buf = '';
  p.stdout.on('data', (d) => {
    buf += d;
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      try {
        const msg = JSON.parse(line);
        job.log.push(msg);
        if (msg.type === 'done') job.result = msg;
        if (msg.type === 'error') job.error = msg.message;
      } catch { /* не-JSON строки (служебный вывод) пропускаем */ }
    }
  });
  let stderr = '';
  p.stderr.on('data', (d) => { stderr = (stderr + d).slice(-2000); });
  p.on('close', async (code) => {
    job.running = false;
    if (code === 0) await loadModel();
    else if (!job.error) job.error = `Процесс завершился с кодом ${code}: ${stderr.split('\n').filter(Boolean).slice(-3).join(' | ')}`;
  });
  res.json({ started: true });
});

app.get('/api/finetune', (_req, res) => res.json(job));

// ---------- WebSocket: поток лиц -> вероятности ----------
const server = createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

wss.on('connection', (ws) => {
  ws.on('message', async (data, isBinary) => {
    if (!isBinary) return;
    const t0 = performance.now();
    try {
      const probs = await predict(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      ws.send(JSON.stringify({ probs, ms: +(performance.now() - t0).toFixed(2), ready: !!session, finetuned: modelInfo.finetuned }));
    } catch (e) {
      ws.send(JSON.stringify({ error: String(e) }));
    }
  });
});

await loadModel();
server.listen(PORT, () => console.log(`Бэкенд: http://localhost:${PORT}  (WebSocket /ws)`));
