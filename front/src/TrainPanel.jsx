// Сбор собственного датасета (кадры по эмоциям и людям) и дообучение модели на нём.
import { useCallback, useEffect, useRef, useState } from 'react'
import { CLASSES, EMOTIONS } from './lib/emotions'
import { grayToPng } from './lib/faceCrop'

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

export default function TrainPanel({ captureMainFace, hasFace }) {
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

  const refresh = useCallback(() => api('/samples').then(setStats).catch(() => setStats(null)), [])
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
  useEffect(() => { pollJob(); return () => clearInterval(poll.current) }, [pollJob])

  const record = async () => {
    setMsg(null)
    for (let i = 3; i > 0; i--) {
      setRec({ phase: 'count', n: i })
      await new Promise((r) => setTimeout(r, 700))
    }
    const images = []
    for (let i = 0; i < shots * 3 && images.length < shots; i++) {
      setRec({ phase: 'rec', n: images.length })
      const g = captureMainFace()
      if (g) images.push(grayToPng(g))
      await new Promise((r) => setTimeout(r, INTERVAL))
    }
    setRec(null)
    if (!images.length) return setMsg('Лицо не найдено — снимки не записаны.')
    try {
      const r = await api('/samples', { method: 'POST', body: { label, images, person } })
      setStats(r.stats)
      setMsg(`Сохранено ${r.saved} снимков «${EMOTIONS[label].label}». Для разнообразия меняйте наклон головы, расстояние и освещение.`)
    } catch (e) { setMsg(`Ошибка: ${e.message}`) }
  }

  const finetune = async () => {
    setMsg(null)
    try {
      await api('/finetune', { method: 'POST', body: { epochs: 8 } })
      pollJob()
    } catch (e) { setMsg(`Ошибка: ${e.message}`) }
  }

  const reset = async () => {
    await api('/model/reset', { method: 'POST' })
    setMsg('Возвращена базовая модель.')
  }

  const clear = async () => {
    if (!confirm('Удалить все записанные снимки?')) return
    setStats(await api('/samples', { method: 'DELETE' }))
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
        <button className="btn primary" onClick={finetune} disabled={busy || total < 10}>
          {job?.running ? 'Обучение…' : '🚀 Дообучить'}
        </button>
      </div>

      {total > 0 && (
        <div className="dataset-line muted small">
          <span>В датасете: <b>{total}</b> кадров · людей: <b>{peopleCount}</b> · записей: <b>{stats.sessions}</b></span>
          <a className="link" href="/api/samples/export" download>скачать .zip</a>
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
          {Math.round(res.base_custom_acc * 100)}% → <b>{Math.round(res.custom_acc * 100)}%</b>,
          на общем тесте: {Math.round(res.base_test_acc * 100)}% → {Math.round(res.test_acc * 100)}%
          <button className="link" onClick={reset}>вернуть базовую</button>
        </div>
      )}
      {job?.error && !job.running && <div className="result error">⚠️ {job.error}</div>}
      {msg && <div className="result">{msg}</div>}
    </section>
  )
}
