import { useEffect, useRef, useState } from 'react'
import { CLASSES, EMOTIONS, UNKNOWN } from './lib/emotions'
import { GESTURES, GESTURE_ORDER } from './lib/gestures'
import { useEmotionEngine } from './lib/useEmotionEngine'
import { useRps } from './lib/useRps'
import { RpsCard, RpsOverlay } from './RpsGame'
import TrainPanel from './TrainPanel'
import ParticleBackground from './ParticleBackground'
import ThemeToggle from './ThemeToggle'
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

// Эмодзи жеста над рукой
function HandOverlay({ hand, scale, videoW, showLabel }) {
  const g = GESTURES[hand.gesture]
  const size = Math.min(120, Math.max(44, hand.size * scale * 0.55))
  const x = (videoW - hand.cx) * scale
  const top = Math.max(4, hand.top * scale - size * 1.3)
  return (
    <div className="emoji-anchor hand-anchor" style={{ left: x, top, fontSize: size }}>
      <span key={hand.gesture} className="emoji">{g.emoji}</span>
      {showLabel && <span className="hand-label">{g.label}</span>}
    </div>
  )
}

function Gestures({ hands, ready }) {
  const active = new Set(hands.map((h) => h.gesture))
  return (
    <section className="card">
      <div className="card-title">
        <h3>Жесты</h3>
        <span className="muted small">{ready ? 'покажите рукой' : 'загрузка…'}</span>
      </div>
      <div className="gestures">
        {GESTURE_ORDER.map((k) => (
          <div key={k} className={`gesture ${active.has(k) ? 'on' : ''}`}>
            <span className="gesture-emoji">{GESTURES[k].emoji}</span>
            <span>{GESTURES[k].hint}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function Stage({ videoRef, engine, showBox, game }) {
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
      {scale > 0 && frame.hands.map((h) => (
        <HandOverlay key={h.id} hand={h} scale={scale} videoW={frame.videoW} showLabel={showBox} />
      ))}
      <RpsOverlay game={game} />
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
          <div className="eyebrow">Сейчас</div>
          <div className="current-label">{face ? em.label : 'Нет лица'}</div>
        </div>
      </div>
      <div className="bars">
        {CLASSES.map((c, i) => {
          const p = face?.avg ? face.avg[i] : 0
          const e = EMOTIONS[c]
          return (
            <div key={c} className={`bar-row ${face?.stable === i ? 'active' : ''}`} style={{ '--c': e.color }}>
              <span className="bar-emoji">{e.emoji}</span>
              <span className="bar-name">{e.label}</span>
              <div className="bar"><div className="bar-fill" style={{ width: `${p * 100}%` }} /></div>
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

function initialTheme() {
  try {
    const saved = localStorage.getItem('emocam-theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch { /* хранилище недоступно */ }
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export default function App() {
  const videoRef = useRef(null)
  const engine = useEmotionEngine(videoRef)
  const rps = useRps(engine.frame.shape)
  const [showBox, setShowBox] = useState(true)
  const [theme, setTheme] = useState(initialTheme)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('emocam-theme', theme) } catch { /* хранилище недоступно */ }
  }, [theme])
  const main = engine.frame.faces[0]
  const { status } = engine

  return (
    <div className="app">
      <ParticleBackground emotion={main?.stable != null ? CLASSES[main.stable] : null} theme={theme} />
      <header className="header">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
              <path d="M8 9v1.5M16 9v1.5M7.5 14.5c1.2 1.6 2.7 2.4 4.5 2.4s3.3-.8 4.5-2.4" />
            </svg>
          </span>
          <h1>EmoCam</h1>
        </div>
        <div className="header-right">
        <div className="statuses">
          <StatusDot ok={status.camera === 'on'} label="Камера" />
          <StatusDot ok={status.detector === 'ready'} label="Детектор" />
          <StatusDot ok={status.socket === 'online' && status.model !== false} label={status.model === false ? 'Модель не обучена' : 'Модель'} />
          {engine.frame.finetuned && <span className="badge">дообучена на вас</span>}
          {status.mode === 'browser' && <span className="badge">в браузере</span>}
        </div>
          <ThemeToggle theme={theme} onChange={setTheme} />
        </div>
      </header>

      <section className="intro">
        <p className="lead">Покажите эмоцию или жест в камеру — эмодзи появится над головой или рукой.</p>
        <p className="muted">Пять эмоций и три жеста в реальном времени. Видео не покидает ваш компьютер.</p>
      </section>

      <main className="layout">
        <div className="left">
          <Stage videoRef={videoRef} engine={engine} showBox={showBox} game={rps.game} />
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
          <Gestures hands={engine.frame.hands} ready={status.hands === 'ready'} />
          <RpsCard rps={rps} shape={engine.frame.shape} ready={status.hands === 'ready'} />
          <SessionStats face={main} />
          {status.mode && <TrainPanel captureMainFace={engine.captureMainFace} hasFace={!!main} mode={status.mode} />}
        </aside>
      </main>
      <footer className="muted small footer">
        Модель предназначена для демонстрации и не оценивает реальное психологическое состояние человека.
      </footer>
    </div>
  )
}
