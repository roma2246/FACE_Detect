// Инференс прямо в браузере (ONNX Runtime Web) — используется, когда Node-бэкенд недоступен,
// например в статическом деплое на Vercel. Предобработка идентична back/server.js.
import * as ort from 'onnxruntime-web/wasm'

const FACE = 48 * 48
ort.env.wasm.wasmPaths = { wasm: '/ort/ort-wasm-simd-threaded.wasm' }
ort.env.wasm.numThreads = 1

let sessionPromise = null
export const loadLocalModel = () =>
  (sessionPromise ??= ort.InferenceSession.create('/models/emotion_cnn.onnx', { executionProviders: ['wasm'] }))

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

/** faces: Uint8Array из N серых лиц 48x48 -> массив из N массивов вероятностей. */
export async function predictLocal(faces) {
  const session = await loadLocalModel()
  const n = faces.length / FACE
  const input = new Float32Array(faces.length)
  for (let i = 0; i < n; i++) equalize(faces, i * FACE, input)
  const { logits } = await session.run({ input: new ort.Tensor('float32', input, [n, 1, 48, 48]) })
  const k = logits.dims[1]
  return Array.from({ length: n }, (_, i) => {
    const row = Array.from(logits.data.subarray(i * k, (i + 1) * k))
    const max = Math.max(...row)
    const e = row.map((v) => Math.exp(v - max))
    const s = e.reduce((a, b) => a + b, 0)
    return e.map((v) => v / s)
  })
}
