// Жесты по 21 точке руки (MediaPipe HandLandmarker). Нейросеть находит точки,
// а сам жест определяется простыми геометрическими правилами — отдельного обучения не нужно.
//   ✌️ мир    — указательный и средний выпрямлены, безымянный и мизинец согнуты
//   🎸 рок    — указательный и мизинец выпрямлены, средний и безымянный согнуты
//   👋 привет — раскрытая ладонь, которая качается влево-вправо
// Для игры «камень, ножницы, бумага» те же позы: кулак, знак мира, раскрытая ладонь.

export const GESTURE_ORDER = ['peace', 'rock', 'wave']

export const GESTURES = {
  peace: { emoji: '✌️', label: 'Мир', hint: 'знак мира' },
  rock: { emoji: '🎸', label: 'Рок', hint: '«коза»' },
  wave: { emoji: '👋', label: 'Привет!', hint: 'помахать' },
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y)

// [кончик, средний сустав] для указательного, среднего, безымянного и мизинца
const FINGERS = [[8, 6], [12, 10], [16, 14], [20, 18]]
const EXTENDED = 1.2 // кончик заметно дальше от запястья, чем сустав
const FOLDED = 1.05 // кончик не дальше сустава

/** Поза кисти в одном кадре: 'peace' | 'rock' | 'palm' | null. pts — 21 точка в пикселях. */
export function handPose(pts) {
  const wrist = pts[0]
  const r = FINGERS.map(([tip, pip]) => dist(wrist, pts[tip]) / (dist(wrist, pts[pip]) || 1))
  const up = r.map((v) => v > EXTENDED)
  const down = r.map((v) => v < FOLDED)
  const [index, middle, ring, pinky] = [0, 1, 2, 3]
  if (up[index] && up[middle] && down[ring] && down[pinky]) return 'peace'
  if (up[index] && up[pinky] && down[middle] && down[ring]) return 'rock'
  if (up.every(Boolean)) return 'palm'
  if (down.every(Boolean)) return 'fist'
  return null
}

// ---------- Камень, ножницы, бумага ----------
export const RPS = {
  rock: { emoji: '✊', label: 'Камень', beats: 'scissors' },
  scissors: { emoji: '✌️', label: 'Ножницы', beats: 'paper' },
  paper: { emoji: '✋', label: 'Бумага', beats: 'rock' },
}
const POSE_TO_RPS = { fist: 'rock', peace: 'scissors', palm: 'paper' }

/** Устойчивая фигура для игры по последним позам руки: 'rock' | 'scissors' | 'paper' | null. */
export function rpsShape(state) {
  for (const pose of Object.keys(POSE_TO_RPS)) {
    if (state.poses.filter((p) => p === pose).length >= POSE_VOTES) return POSE_TO_RPS[pose]
  }
  return null
}

/** Сколько раз движение по X сменило направление (размах не меньше thr). */
export function countSwings(xs, thr) {
  let ext = xs[0]
  let dir = 0
  let n = 0
  for (const x of xs) {
    if (dir === 0) {
      if (Math.abs(x - ext) > thr) { dir = Math.sign(x - ext); ext = x }
    } else if ((x - ext) * dir > 0) {
      ext = x
    } else if ((ext - x) * dir > thr) {
      n++
      dir = -dir
      ext = x
    }
  }
  return n
}

const POSE_WINDOW = 5 // последние измерения позы
const POSE_VOTES = 3 // сколько из них должны совпасть
const WAVE_WINDOW_MS = 1400
const WAVE_HOLD_MS = 900
const WAVE_SWINGS = 2 // туда-обратно-туда
const WAVE_AMPLITUDE = 0.18 // доля размера кисти

export function newHandState() {
  return { poses: [], xs: [], waveUntil: 0 }
}

/**
 * Обновляет состояние руки и возвращает устойчивый жест: 'peace' | 'rock' | 'wave' | null.
 * hand: { pts, cx, size } в пикселях видео.
 */
export function updateGesture(state, hand, now) {
  const pose = handPose(hand.pts)
  state.poses.push(pose)
  if (state.poses.length > POSE_WINDOW) state.poses.shift()

  // Махание: следим за горизонтальным положением раскрытой ладони
  if (pose === 'palm') {
    state.xs.push({ t: now, x: hand.cx })
    while (state.xs.length && now - state.xs[0].t > WAVE_WINDOW_MS) state.xs.shift()
    const swings = countSwings(state.xs.map((p) => p.x), hand.size * WAVE_AMPLITUDE)
    if (swings >= WAVE_SWINGS) state.waveUntil = now + WAVE_HOLD_MS
  } else if (pose !== null) {
    state.xs = [] // другой чёткий жест — это уже не махание
  }
  if (now < state.waveUntil) return 'wave'

  for (const g of ['peace', 'rock']) {
    if (state.poses.filter((p) => p === g).length >= POSE_VOTES) return g
  }
  return null
}
