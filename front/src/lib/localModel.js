// Инференс прямо в браузере (ONNX Runtime Web) — используется, когда Node-бэкенд недоступен,
// например в статическом деплое на Vercel. Предобработка идентична back/server.js.
import * as ort from 'onnxruntime-web/wasm'

const FACE = 48 * 48
ort.env.wasm.wasmPaths = { wasm: '/ort/ort-wasm-simd-threaded.wasm' }
ort.env.wasm.numThreads = 1

export const FEATURES = 128
const HEAD_KEY = 'emocam-head'

let sessionPromise = null
export const loadLocalModel = () =>
  (sessionPromise ??= ort.InferenceSession.create('/models/emotion_cnn_web.onnx', { executionProviders: ['wasm'] }))

// ---------- Последний слой (128 -> 5): базовый или дообученный в браузере ----------
let baseHeadPromise = null
export const loadBaseHead = () =>
  (baseHeadPromise ??= fetch('/models/emotion_head.json').then((r) => {
    if (!r.ok) throw new Error('не найден emotion_head.json')
    return r.json()
  }))

let customHead // undefined — ещё не читали; null — нет
export function getCustomHead() {
  if (customHead === undefined) {
    try { customHead = JSON.parse(localStorage.getItem(HEAD_KEY)) } catch { customHead = null }
  }
  return customHead
}
export function setCustomHead(head) {
  customHead = head
  try {
    if (head) localStorage.setItem(HEAD_KEY, JSON.stringify(head))
    else localStorage.removeItem(HEAD_KEY)
  } catch { /* хранилище недоступно — слой действует до перезагрузки страницы */ }
}

export function softmax(row) {
  const max = Math.max(...row)
  const e = row.map((v) => Math.exp(v - max))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** Логиты по признакам одного лица: W·f + b. */
export function applyHead(head, feats, off = 0) {
  const k = head.b.length
  const out = new Array(k)
  for (let c = 0; c < k; c++) {
    let s = head.b[c]
    for (let j = 0; j < FEATURES; j++) s += head.w[c * FEATURES + j] * feats[off + j]
    out[c] = s
  }
  return out
}

// Выравнивание гистограммы — копия PIL.ImageOps.equalize (так обучалась модель).
function equalize(src, off, out) {
  const h = new Uint32Array(256)
  for (let i = 0; i < FACE; i++) h[src[off + i]]++
  let last = 255
  while (last > 0 && h[last] === 0) last--
  const step = Math.floor((FACE - h[last]) / 255)
  if (step === 0) {
    for (let i = 0; i < FACE; i++) out[off + i] = src[off + i] / 255
    return
  }
  const lut = new Uint8Array(256)
  let n = Math.floor(step / 2)
  for (let i = 0; i < 256; i++) {
    lut[i] = Math.min(255, Math.floor(n / step))
    n += h[i]
  }
  for (let i = 0; i < FACE; i++) out[off + i] = lut[src[off + i]] / 255
}

async function run(faces) {
  const session = await loadLocalModel()
  const n = faces.length / FACE
  const input = new Float32Array(faces.length)
  for (let i = 0; i < n; i++) equalize(faces, i * FACE, input)
  const out = await session.run({ input: new ort.Tensor('float32', input, [n, 1, 48, 48]) })
  return { n, logits: out.logits.data, features: out.features.data }
}

/** faces: Uint8Array из N серых лиц 48x48 -> Float32Array из N*128 признаков предпоследнего слоя. */
export async function extractFeatures(faces) {
  const BATCH = 32 * FACE
  const all = new Float32Array((faces.length / FACE) * FEATURES)
  for (let off = 0; off < faces.length; off += BATCH) {
    const { features } = await run(faces.subarray(off, off + BATCH))
    all.set(features, (off / FACE) * FEATURES)
  }
  return all
}

/** faces: Uint8Array из N серых лиц 48x48 -> массив из N массивов вероятностей. */
export async function predictLocal(faces) {
  const { n, logits, features } = await run(faces)
  const head = getCustomHead()
  const k = logits.length / n
  return Array.from({ length: n }, (_, i) =>
    softmax(head ? applyHead(head, features, i * FEATURES) : Array.from(logits.subarray(i * k, (i + 1) * k))))
}
