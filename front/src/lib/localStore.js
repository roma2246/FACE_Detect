// Свой датасет в режиме без бэкенда: записи хранятся в IndexedDB этого браузера
// и никуда не отправляются. Одна запись = одна серия кадров одной эмоции.
import { CLASSES } from './emotions'

const DB = 'emocam'
const STORE = 'sessions'

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1)
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true })
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

async function tx(mode, fn) {
  const db = await open()
  try {
    return await new Promise((resolve, reject) => {
      const t = db.transaction(STORE, mode)
      const req = fn(t.objectStore(STORE))
      t.oncomplete = () => resolve(req.result)
      t.onerror = () => reject(t.error)
      t.onabort = () => reject(t.error)
    })
  } finally {
    db.close()
  }
}

/** Все записи: [{id, label, person, n, faces: Uint8Array(n*2304)}] в порядке создания. */
export const allSessions = () => tx('readonly', (s) => s.getAll())

/** grays — массив Uint8Array(2304), кадры одной записи по порядку. */
export async function addSession(label, person, grays) {
  const faces = new Uint8Array(grays.length * 2304)
  grays.forEach((g, i) => faces.set(g, i * 2304))
  await tx('readwrite', (s) => s.add({ label, person: person.trim() || 'me', n: grays.length, faces, t: Date.now() }))
  return sampleStats()
}

export async function clearSessions() {
  await tx('readwrite', (s) => s.clear())
  return sampleStats()
}

/** Та же форма, что у GET /api/samples на бэкенде. */
export async function sampleStats() {
  const sessions = await allSessions()
  const classes = Object.fromEntries(CLASSES.map((c) => [c, 0]))
  const people = {}
  let total = 0
  for (const s of sessions) {
    classes[s.label] += s.n
    people[s.person] = (people[s.person] || 0) + s.n
    total += s.n
  }
  return { classes, people, total, sessions: sessions.length }
}
