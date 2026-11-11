// Дообучение прямо в браузере (режим без бэкенда, например на Vercel).
// Свёрточная часть сети заморожена: она превращает лицо в 128 признаков, а обучается
// только последний слой 128 -> 5 (softmax-регрессия) на своих кадрах. Это transfer learning
// в самом лёгком виде — считается за секунду на обычном JS, без PyTorch.
import { CLASSES } from './emotions'
import { FEATURES, applyHead, extractFeatures, loadBaseHead, setCustomHead, softmax } from './localModel'
import { allSessions } from './localStore'

const EPOCHS = 60
const LR = 0.01
const KEEP_CLOSE = 0.02 // штраф за уход весов от базовых — чтобы не разучиться на других людях
const TEST_SHARE = 0.2 // последние 20% кадров каждой записи — только для проверки

function accuracy(head, set) {
  if (!set.y.length) return 0
  let ok = 0
  set.y.forEach((y, i) => {
    const logits = applyHead(head, set.x, i * FEATURES)
    if (logits.indexOf(Math.max(...logits)) === y) ok++
  })
  return ok / set.y.length
}

/** onEpoch({epoch, epochs, custom_acc}) вызывается после каждой эпохи. */
export async function browserFinetune(onEpoch) {
  const sessions = await allSessions()
  const train = { x: [], y: [] }
  const test = { x: [], y: [] }
  for (const s of sessions) {
    const feats = await extractFeatures(s.faces)
    const nTest = s.n >= 5 ? Math.max(1, Math.round(s.n * TEST_SHARE)) : 0
    const y = CLASSES.indexOf(s.label)
    for (let i = 0; i < s.n; i++) {
      const part = i < s.n - nTest ? train : test
      part.x.push(feats.subarray(i * FEATURES, (i + 1) * FEATURES))
      part.y.push(y)
    }
  }
  const flat = (rows) => {
    const a = new Float32Array(rows.length * FEATURES)
    rows.forEach((r, i) => a.set(r, i * FEATURES))
    return a
  }
  train.x = flat(train.x)
  test.x = flat(test.x)
  const heldOut = test.y.length > 0
  const evalSet = heldOut ? test : train

  const base = await loadBaseHead()
  const K = base.b.length
  const head = { w: Float64Array.from(base.w), b: Float64Array.from(base.b) }
  const baseAcc = accuracy(base, evalSet)

  // Adam по всему набору сразу (он маленький)
  const n = train.y.length
  const size = K * FEATURES + K
  const m = new Float64Array(size)
  const v = new Float64Array(size)
  const grad = new Float64Array(size)
  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    grad.fill(0)
    for (let i = 0; i < n; i++) {
      const p = softmax(applyHead(head, train.x, i * FEATURES))
      for (let c = 0; c < K; c++) {
        const d = (p[c] - (train.y[i] === c ? 1 : 0)) / n
        for (let j = 0; j < FEATURES; j++) grad[c * FEATURES + j] += d * train.x[i * FEATURES + j]
        grad[K * FEATURES + c] += d
      }
    }
    for (let q = 0; q < K * FEATURES; q++) grad[q] += KEEP_CLOSE * (head.w[q] - base.w[q])
    for (let q = 0; q < size; q++) {
      m[q] = 0.9 * m[q] + 0.1 * grad[q]
      v[q] = 0.999 * v[q] + 0.001 * grad[q] * grad[q]
      const step = (LR * (m[q] / (1 - 0.9 ** epoch))) / (Math.sqrt(v[q] / (1 - 0.999 ** epoch)) + 1e-8)
      if (q < K * FEATURES) head.w[q] -= step
      else head.b[q - K * FEATURES] -= step
    }
    onEpoch?.({ epoch, epochs: EPOCHS, custom_acc: accuracy(head, evalSet) })
    if (epoch % 5 === 0) await new Promise((r) => setTimeout(r, 0)) // дать интерфейсу перерисоваться
  }

  setCustomHead({ w: Array.from(head.w, (x) => +x.toFixed(6)), b: Array.from(head.b, (x) => +x.toFixed(6)) })
  return {
    custom_acc: accuracy(head, evalSet), base_custom_acc: baseAcc,
    held_out: heldOut, n_train: n, n_test: test.y.length,
  }
}
