// Копирует wasm-рантаймы (MediaPipe, ONNX Runtime) из node_modules в public/ и
// синхронизирует модель эмоций из back/models (если бэкенд лежит рядом).
import { copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const copy = (from, to) => {
  mkdirSync(dirname(to), { recursive: true })
  copyFileSync(from, to)
}

for (const f of ['vision_wasm_internal.js', 'vision_wasm_internal.wasm']) {
  copy(join(root, 'node_modules/@mediapipe/tasks-vision/wasm', f), join(root, 'public/mediapipe', f))
}
for (const f of ['ort-wasm-simd-threaded.wasm']) {
  copy(join(root, 'node_modules/onnxruntime-web/dist', f), join(root, 'public/ort', f))
}
// Браузерная версия модели (логиты + признаки) и веса последнего слоя — см. back/ml/export_browser.py
for (const f of ['emotion_cnn_web.onnx', 'emotion_head.json']) {
  const src = join(root, '../back/models', f)
  if (existsSync(src)) copy(src, join(root, 'public/models', f))
}
console.log('assets copied')
