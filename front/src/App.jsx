import { useEffect, useRef, useState } from 'react'
import { CLASSES, EMOTIONS, UNKNOWN } from './lib/emotions'
import { useEmotionEngine } from './lib/useEmotionEngine'
import TrainPanel from './TrainPanel'
import './App.css'

function emotionOf(face) {
  return face?.stable == null ? UNKNOWN : EMOTIONS[CLASSES[face.stable]]
}

// Эмодзи над головой + рамка лица. Координаты в пикселях видео, зеркалим по X.
function FaceOverlay({ face, scale, videoW, showBox }) {
  const { box } = face
  const em = emotionOf(face)
  const x = (videoW - box.originX - box.width) * scale
  const y = box.originY * scale
  const w = box.width * scale
  const h = box.height * scale
  const size = Math.max(48, w * 0.62)
  const conf = face.stable != null && face.avg ? Math.round(face.avg[face.stable] * 100) : null
  // Эмодзи не должен уходить за верхний край кадра
  const emojiTop = Math.max(4, y - size * 1.25)

  return (
    <>
      {showBox && (
        <div className="face-box" style={{ left: x, top: y, width: w, height: h, '--c': em.color }}>
          <span className="face-label">
            {em.label}{conf != null && ` · ${conf}%`}
          </span>
        </div>
      )}
      <div className="emoji-anchor" style={{ left: x + w / 2, top: emojiTop, fontSize: size }}>
        <span key={em.emoji} className="emoji">{em.emoji}</span>
      </div>
    </>
  )
}

function Stage({ videoRef, engine, showBox }) {
  const wrapRef = useRef(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(wrapRef.current)
    return () => ro.disconnect()
  }, [])
  const { frame, status, error } = engine
  const scale = frame.videoW ? width / frame.videoW : 0
  const loading = !error && (status.camera !== 'on' || status.detector !== 'ready')

  return (
    <div className="stage" ref={wrapRef} style={{ aspectRatio: frame.videoW ? `${frame.videoW}/${frame.videoH}` : '16/9' }}>
      <video ref={videoRef} playsInline muted className="video" />
      {scale > 0 && frame.faces.map((f) => (
        <FaceOverlay key={f.id} face={f} scale={scale} videoW={frame.videoW} showBox={showBox} />
      ))}
      <div className="hud">
        <span>{frame.fps} FPS</span>
        <span>{frame.ms} мс</span>
      </div>
      {loading && <div className="stage-msg"><div className="spinner" />Запускаем камеру и детектор лиц…</div>}
      {error && <div className="stage-msg error">⚠️ {error}</div>}
      {!loading && !error && frame.faces.length === 0 && (
        <div className="stage-hint">Лицо не найдено — посмотрите в камеру 👀</div>
      )}
    </div>
  )
}

function Probabilities({ face }) {
  const em = emotionOf(face)
  return (
    <section className="card">
      <div className="current">
        <span className="current-emoji" key={em.emoji}>{em.emoji}</span>
        <div>
          <div className="muted small">Сейчас</div>
          <div className="current-label" style={{ color: em.color }}>{face ? em.label : 'Нет лица'}</div>
        </div>
      </div>
      <div className="bars">
        {CLASSES.map((c, i) => {
          const p = face?.avg ? face.avg[i] : 0
          const e = EMOTIONS[c]
          return (
            <div key={c} className={`bar-row ${face?.stable === i ? 'active' : ''}`}>
              <span className="bar-emoji">{e.emoji}</span>
              <span className="bar-name">{e.label}</span>
              <div className="bar"><div className="bar-fill" style={{ width: `${p * 100}%`, background: e.color }} /></div>
              <span className="bar-val">{Math.round(p * 100)}%</span>
            </div>
          )
        })}
      </div>
    </section>
  )
}

// Статистика эмоций за сессию (доля времени по главному лицу)
const ZERO = Object.fromEntries(CLASSES.map((c) => [c, 0]))

function SessionStats({ face }) {
  const [acc, setAcc] = useState(ZERO)
  const stable = face?.stable ?? null
  const stableRef = useRef(stable)
  useEffect(() => { stableRef.current = stable }, [stable])
  useEffect(() => {
    const TICK = 250
    const id = setInterval(() => {
      const s = stableRef.current
      if (s != null) setAcc((a) => ({ ...a, [CLASSES[s]]: a[CLASSES[s]] + TICK }))
    }, TICK)
    return () => clearInterval(id)
  }, [])
  const total = Object.values(acc).reduce((a, b) => a + b, 0)

  return (
    <section className="card">
      <div className="card-title">
        <h3>Статистика сессии</h3>
        <button className="link" onClick={() => setAcc(ZERO)}>сбросить</button>
      </div>
      <div className="stack">
        {total > 0 ? CLASSES.map((c) => acc[c] > 0 && (
          <div key={c} style={{ flex: acc[c], background: EMOTIONS[c].color }} title={EMOTIONS[c].label} />
        )) : <div className="stack-empty" />}
      </div>
      <div className="legend">
        {CLASSES.map((c) => (
          <span key={c}>{EMOTIONS[c].emoji} {total ? Math.round((acc[c] / total) * 100) : 0}%</span>
        ))}
      </div>
    </section>
  )
}

function StatusDot({ ok, label }) {
  return <span className={`status ${ok ? 'ok' : 'bad'}`}><i />{label}</span>
}

export default function App() {
  const videoRef = useRef(null)
  const engine = useEmotionEngine(videoRef)
  const [showBox, setShowBox] = useState(true)
  const main = engine.frame.faces[0]
  const { status } = engine

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">🎭</span>
          <div>
            <h1>EmoCam</h1>
            <p className="muted small">Распознавание эмоций по лицу в реальном времени</p>
          </div>
        </div>
        <div className="statuses">
          <StatusDot ok={status.camera === 'on'} label="Камера" />
          <StatusDot ok={status.detector === 'ready'} label="Детектор" />
          <StatusDot ok={status.socket === 'online' && status.model !== false} label={status.model === false ? 'Модель не обучена' : 'Модель'} />
          {engine.frame.finetuned && <span className="badge">дообучена на вас</span>}
          {status.mode === 'browser' && <span className="badge">в браузере</span>}
        </div>
      </header>

      <main className="layout">
        <div className="left">
          <Stage videoRef={videoRef} engine={engine} showBox={showBox} />
          <div className="toolbar">
            <label className="switch">
              <input type="checkbox" checked={showBox} onChange={(e) => setShowBox(e.target.checked)} />
              <span>Рамка и подпись</span>
            </label>
            <span className="muted small">Лиц в кадре: {engine.frame.faces.length}</span>
          </div>
        </div>
        <aside className="right">
          <Probabilities face={main} />
          <SessionStats face={main} />
          {status.mode === 'browser' ? (
            <section className="card">
              <h3>Свой датасет и дообучение</h3>
              <p className="muted small" style={{ margin: '8px 0 0' }}>
                Сейчас модель работает прямо в вашем браузере. Сбор своего датасета и дообучение доступны при локальном запуске
                с Node-бэкендом — инструкция в README репозитория.
              </p>
            </section>
          ) : (
            <TrainPanel captureMainFace={engine.captureMainFace} hasFace={!!main} />
          )}
        </aside>
      </main>
      <footer className="muted small footer">
        Модель предназначена для демонстрации и не оценивает реальное психологическое состояние человека.
      </footer>
    </div>
  )
}
