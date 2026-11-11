// Дообучение модели на собственных снимках: запись кадров по эмоциям -> запуск fine-tuning на бэкенде.
import { useCallback, useEffect, useRef, useState } from 'react'
import { CLASSES, EMOTIONS } from './lib/emotions'
import { grayToPng } from './lib/faceCrop'

const SHOTS = 40
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
    for (let i = 0; i < SHOTS * 3 && images.length < SHOTS; i++) {
      setRec({ phase: 'rec', n: images.length })
      const g = captureMainFace()
      if (g) images.push(grayToPng(g))
      await new Promise((r) => setTimeout(r, INTERVAL))
    }
    setRec(null)
    if (!images.length) return setMsg('Лицо не найдено — снимки не записаны.')
    try {
      const r = await api('/samples', { method: 'POST', body: { label, images } })
      setStats(r.stats)
      setMsg(`Сохранено ${r.saved} снимков «${EMOTIONS[label].label}». Меняйте немного наклон головы и освещение.`)
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
    setMsg('Возвращена базовая модель (FER-2013).')
  }

  const clear = async () => {
    if (!confirm('Удалить все записанные снимки?')) return
    setStats(await api('/samples', { method: 'DELETE' }))
  }

  const total = stats ? Object.values(stats).reduce((a, b) => a + b, 0) : 0
  const lastEpoch = job?.log?.filter((l) => l.type === 'epoch').at(-1)
  const busy = !!rec || job?.running

  return (
    <section className="card">
      <div className="card-title">
        <h3>Дообучить на моём лице</h3>
        {total > 0 && <button className="link" onClick={clear} disabled={busy}>очистить</button>}
      </div>
      <p className="muted small">
        1) выберите эмоцию и покажите её в камеру · 2) нажмите «Записать» · 3) повторите для других эмоций · 4) «Дообучить»
      </p>
      <div className="chips">
        {CLASSES.map((c) => (
          <button key={c} className={`chip ${label === c ? 'sel' : ''}`} onClick={() => setLabel(c)} disabled={busy}
            style={{ '--c': EMOTIONS[c].color }}>
            {EMOTIONS[c].emoji} <b>{stats?.[c] ?? 0}</b>
          </button>
        ))}
      </div>
      <div className="row">
        <button className="btn" onClick={record} disabled={busy || !hasFace}>
          {rec?.phase === 'count' ? `Приготовьтесь… ${rec.n}` :
            rec?.phase === 'rec' ? `Запись ${rec.n}/${SHOTS}` :
              `● Записать «${EMOTIONS[label].label}»`}
        </button>
        <button className="btn primary" onClick={finetune} disabled={busy || total < 10}>
          {job?.running ? 'Обучение…' : '🚀 Дообучить'}
        </button>
      </div>

      {job?.running && (
        <div className="progress">
          <div className="progress-fill" style={{ width: `${lastEpoch ? (lastEpoch.epoch / lastEpoch.epochs) * 100 : 3}%` }} />
          <span>{lastEpoch ? `эпоха ${lastEpoch.epoch}/${lastEpoch.epochs} · точность на ваших снимках ${Math.round(lastEpoch.custom_acc * 100)}%` : 'подготовка данных…'}</span>
        </div>
      )}
      {job?.result && !job.running && (
        <div className="result">
          ✅ Готово. На ваших снимках: {Math.round(job.result.base_custom_acc * 100)}% → <b>{Math.round(job.result.custom_acc * 100)}%</b>,
          на FER-2013: {Math.round(job.result.base_test_acc * 100)}% → {Math.round(job.result.test_acc * 100)}%
          <button className="link" onClick={reset}>вернуть базовую</button>
        </div>
      )}
      {job?.error && !job.running && <div className="result error">⚠️ {job.error}</div>}
      {msg && <div className="result">{msg}</div>}
    </section>
  )
}
