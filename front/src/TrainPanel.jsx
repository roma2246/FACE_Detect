// Сбор собственного датасета (кадры по эмоциям и людям) и дообучение модели на нём.
// mode="server"  — снимки и обучение на Node-бэкенде (PyTorch, дообучается вся сеть);
// mode="browser" — бэкенда нет (Vercel): снимки лежат в IndexedDB, в браузере дообучается последний слой.
import { useCallback, useEffect, useRef, useState } from 'react'
import { browserFinetune } from './lib/browserTrain'
import { CLASSES, EMOTIONS } from './lib/emotions'
import { grayToPng } from './lib/faceCrop'
import { getCustomHead, setCustomHead } from './lib/localModel'
import { addSession, clearSessions, sampleStats } from './lib/localStore'

const SHOT_OPTIONS = [40, 80, 150]
const INTERVAL = 120

async function api(path, opts = {}) {
  const r = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.error || r.statusText)
  return data
}

export default function TrainPanel({ captureMainFace, hasFace, mode = 'server' }) {
  const browser = mode === 'browser'
  const [label, setLabel] = useState('happy')
  const [person, setPerson] = useState(() => {
    try { return localStorage.getItem('emocam-person') || '' } catch { return '' }
  })
  const [shots, setShots] = useState(SHOT_OPTIONS[0])
  const [stats, setStats] = useState(null)
  const [rec, setRec] = useState(null) // {phase:'count'|'rec', n}
  const [job, setJob] = useState(null)
  const [msg, setMsg] = useState(null)
  const poll = useRef(0)
  const [tuned, setTuned] = useState(() => !!getCustomHead()) // есть ли дообученный слой в этом браузере

  const refresh = useCallback(
    () => (browser ? sampleStats() : api('/samples')).then(setStats).catch(() => setStats(null)),
    [browser],
  )
  useEffect(() => { refresh() }, [refresh])

  const pollJob = useCallback(() => {
    clearInterval(poll.current)
    const tick = async () => {
      try {
        const j = await api('/finetune')
        setJob(j)
        if (!j.running) clearInterval(poll.current)
      } catch { /* бэкенд недоступен */ }
    }
    tick()
    poll.current = setInterval(tick, 800)
  }, [])
  useEffect(() => {
    if (browser) return undefined
    pollJob()
    return () => clearInterval(poll.current)
  }, [browser, pollJob])

  const record = async () => {
    setMsg(null)
    for (let i = 3; i > 0; i--) {
      setRec({ phase: 'count', n: i })
      await new Promise((r) => setTimeout(r, 700))
    }
    const grays = []
    for (let i = 0; i < shots * 3 && grays.length < shots; i++) {
      setRec({ phase: 'rec', n: grays.length })
      const g = captureMainFace()
      if (g) grays.push(g.slice())
      await new Promise((r) => setTimeout(r, INTERVAL))
    }
    setRec(null)
    if (!grays.length) return setMsg('Лицо не найдено — снимки не записаны.')
    try {
      const r = browser
        ? { saved: grays.length, stats: await addSession(label, person, grays) }
        : await api('/samples', { method: 'POST', body: { label, images: grays.map(grayToPng), person } })
      setStats(r.stats)
      setMsg(`Сохранено ${r.saved} снимков «${EMOTIONS[label].label}». Для разнообразия меняйте наклон головы, расстояние и освещение.`)
    } catch (e) { setMsg(`Ошибка: ${e.message}`) }
  }

  const finetune = async () => {
    setMsg(null)
    if (browser) {
      setJob({ running: true, log: [] })
      try {
        const result = await browserFinetune((e) => setJob({ running: true, log: [{ type: 'epoch', ...e }] }))
        setJob({ running: false, result })
        setTuned(true)
      } catch (e) { setJob({ running: false, error: e.message }) }
      return
    }
    try {
      await api('/finetune', { method: 'POST', body: { epochs: 8 } })
      pollJob()
    } catch (e) { setMsg(`Ошибка: ${e.message}`) }
  }

  const reset = async () => {
    if (browser) {
      setCustomHead(null)
      setTuned(false)
      setJob(null)
    } else {
      await api('/model/reset', { method: 'POST' })
    }
    setMsg('Возвращена базовая модель.')
  }

  const clear = async () => {
    if (!confirm('Удалить все записанные снимки?')) return
    setStats(browser ? await clearSessions() : await api('/samples', { method: 'DELETE' }))
  }

  const savePerson = (v) => {
    setPerson(v)
    try { localStorage.setItem('emocam-person', v) } catch { /* хранилище недоступно */ }
  }

  const total = stats?.total ?? 0
  const peopleCount = stats ? Object.keys(stats.people).length : 0
  const res = job?.result
  const lastEpoch = job?.log?.filter((l) => l.type === 'epoch').at(-1)
  const busy = !!rec || job?.running
  const classCount = stats ? CLASSES.filter((c) => stats.classes[c] > 0).length : 0
  // В браузере учится только последний слой: с одной эмоцией он просто начнёт отвечать ею на всё
  const canTrain = total >= 10 && (!browser || classCount >= 2)

  return (
    <section className="card">
      <div className="card-title">
        <h3>Свой датасет и дообучение</h3>
        {total > 0 && <button className="link" onClick={clear} disabled={busy}>очистить</button>}
      </div>
      <p className="muted small">
        1) впишите, кто в кадре · 2) выберите эмоцию и покажите её · 3) «Записать» · 4) повторите для других эмоций и людей · 5) «Дообучить»
      </p>
      <div className="fields">
        <input className="input" placeholder="Кто в кадре (имя)" value={person} maxLength={24}
          onChange={(e) => savePerson(e.target.value)} disabled={busy} />
        <select className="input" value={shots} onChange={(e) => setShots(Number(e.target.value))} disabled={busy}>
          {SHOT_OPTIONS.map((n) => <option key={n} value={n}>{n} кадров</option>)}
        </select>
      </div>
      <div className="chips">
        {CLASSES.map((c) => (
          <button key={c} className={`chip ${label === c ? 'sel' : ''}`} onClick={() => setLabel(c)} disabled={busy}
            style={{ '--c': EMOTIONS[c].color }}>
            {EMOTIONS[c].emoji} <b>{stats?.classes[c] ?? 0}</b>
          </button>
        ))}
      </div>
      <div className="row">
        <button className="btn" onClick={record} disabled={busy || !hasFace}>
          {rec?.phase === 'count' ? `Приготовьтесь… ${rec.n}` :
            rec?.phase === 'rec' ? `Запись ${rec.n}/${shots}` :
              `● Записать «${EMOTIONS[label].label}»`}
        </button>
        <button className="btn primary" onClick={finetune} disabled={busy || !canTrain}
          title={browser && total >= 10 && classCount < 2 ? 'Запишите хотя бы две разные эмоции' : undefined}>
          {job?.running ? 'Обучение…' : '🚀 Дообучить'}
        </button>
      </div>

      {total > 0 && (
        <div className="dataset-line muted small">
          <span>В датасете: <b>{total}</b> кадров · людей: <b>{peopleCount}</b> · записей: <b>{stats.sessions}</b></span>
          {!browser && <a className="link" href="/api/samples/export" download>скачать .zip</a>}
        </div>
      )}

      {job?.running && (
        <div className="progress">
          <div className="progress-fill" style={{ width: `${lastEpoch ? (lastEpoch.epoch / lastEpoch.epochs) * 100 : 3}%` }} />
          <span>{lastEpoch ? `эпоха ${lastEpoch.epoch}/${lastEpoch.epochs} · точность на отложенных кадрах ${Math.round(lastEpoch.custom_acc * 100)}%` : 'подготовка данных…'}</span>
        </div>
      )}
      {res && !job.running && (
        <div className="result">
          ✅ Готово. {res.held_out ? `На ${res.n_test} отложенных своих кадрах (в обучении не участвовали)` : 'На своих кадрах'}:{' '}
          {Math.round(res.base_custom_acc * 100)}% → <b>{Math.round(res.custom_acc * 100)}%</b>
          {browser
            ? '. На общем тесте в браузере не проверяется.'
            : `, на общем тесте: ${Math.round(res.base_test_acc * 100)}% → ${Math.round(res.test_acc * 100)}%`}
          <button className="link" onClick={reset}>вернуть базовую</button>
        </div>
      )}
      {browser && tuned && !res && !job?.running && (
        <div className="result">
          Сейчас работает слой, дообученный в этом браузере.
          <button className="link" onClick={reset}>вернуть базовую</button>
        </div>
      )}
      {browser && (
        <p className="muted small" style={{ margin: '12px 0 0' }}>
          Режим без сервера: снимки хранятся только в этом браузере, дообучается последний слой сети.
          Полное дообучение всей сети и выгрузка датасета — при локальном запуске с бэкендом.
        </p>
      )}
      {job?.error && !job.running && <div className="result error">⚠️ {job.error}</div>}
      {msg && <div className="result">{msg}</div>}
    </section>
  )
}
