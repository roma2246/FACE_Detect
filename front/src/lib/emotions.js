// Порядок классов совпадает с моделью (back/ml/model.py)
export const CLASSES = ['happy', 'sad', 'angry', 'surprise', 'neutral']

export const EMOTIONS = {
  happy: { emoji: '😊', label: 'Радость', color: '#facc15' },
  sad: { emoji: '😢', label: 'Грусть', color: '#60a5fa' },
  angry: { emoji: '😡', label: 'Злость', color: '#f87171' },
  surprise: { emoji: '😮', label: 'Удивление', color: '#c084fc' },
  neutral: { emoji: '😐', label: 'Нейтрально', color: '#94a3b8' },
}

export const UNKNOWN = { emoji: '🤔', label: 'Не уверен', color: '#64748b' }

export const SMOOTH_WINDOW = 8 // кадров в скользящем окне
export const CONFIDENCE_THRESHOLD = 0.4

export const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0)

export function average(history) {
  const out = new Array(CLASSES.length).fill(0)
  for (const p of history) for (let i = 0; i < out.length; i++) out[i] += p[i] / history.length
  return out
}
