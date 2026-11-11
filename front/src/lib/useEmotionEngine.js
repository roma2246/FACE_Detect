// Камера + детекция лиц (MediaPipe) + классификация эмоций + сглаживание.
// Эмоции считает Node-бэкенд (WebSocket); если он недоступен — модель запускается прямо в браузере.
import { useCallback, useEffect, useRef, useState } from 'react'
import { FaceDetector, FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import { CONFIDENCE_THRESHOLD, SMOOTH_WINDOW, argmax, average } from './emotions'
import { cropGray } from './faceCrop'
import { newHandState, rpsShape, updateGesture } from './gestures'
import { getCustomHead, loadLocalModel, predictLocal } from './localModel'

const MAX_FACES = 4
const MAX_HANDS = 2
const HAND_EVERY = 2 // руки ищем через кадр — этого хватает, а FPS почти не падает

// Только для разработки: ?demo=/face.png подставляет картинку вместо камеры (проверка без веб-камеры).
const demoUrl = import.meta.env.DEV ? new URLSearchParams(location.search).get('demo') : null

async function demoStream(url) {
  const img = new Image()
  img.src = url
  await img.decode()
  const c = document.createElement('canvas')
  c.width = 1280
  c.height = 720
  const g = c.getContext('2d')
  const w = (1100 * img.naturalWidth) / img.naturalHeight
  const params = new URLSearchParams(location.search)
  const fit = params.has('fit') // вписать картинку целиком (для фото рук)
  const sway = params.has('sway') // качать влево-вправо (проверка махания)
  const draw = () => {
    g.fillStyle = '#333'
    g.fillRect(0, 0, 1280, 720)
    const dx = sway ? Math.sin(performance.now() / 160) * 140 : 0
    if (fit) {
      const fw = (680 * img.naturalWidth) / img.naturalHeight
      g.drawImage(img, 640 - fw / 2 + dx, 20, fw, 680)
    } else {
      g.drawImage(img, 640 - w / 2 + dx, 20, w, 1100)
    }
  }
  draw()
  setInterval(draw, 33)
  return c.captureStream(30)
}
let nextTrackId = 1

export function useEmotionEngine(videoRef) {
  const [status, setStatus] = useState({ camera: 'starting', detector: 'loading', socket: 'connecting' })
  const [error, setError] = useState(null)
  const [frame, setFrame] = useState({ faces: [], hands: [], shape: null, fps: 0, ms: 0, videoW: 0, videoH: 0, finetuned: false })

  const tracks = useRef([]) // [{id, box, history, stable, lastSeen}]
  const hands = useRef([]) // [{id, cx, top, size, state, gesture, lastSeen}]
  const lastCrops = useRef(new Map()) // trackId -> Uint8Array (для сбора своих снимков)

  useEffect(() => {
    let cancelled = false
    let raf = 0
    let stream = null
    let detector = null
    let handLm = null
    let tick = 0
    let ws = null
    let wsTimer = 0
    let everConnected = false
    let local = false // true — бэкенда нет, инференс в браузере
    let inflight = null // id треков, отправленных в последнем запросе
    let ms = 0
    let finetuned = false
    let frames = 0
    let fps = 0
    let fpsT = performance.now()

    // ---------- WebSocket с автопереподключением ----------
    const connect = () => {
      if (cancelled) return
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws`)
      ws.binaryType = 'arraybuffer'
      ws.onopen = () => {
        everConnected = true
        setStatus((s) => ({ ...s, socket: 'online', mode: 'server' }))
      }
      ws.onclose = () => {
        inflight = null
        if (!everConnected) return startLocal() // бэкенда нет (например, деплой на Vercel)
        setStatus((s) => ({ ...s, socket: 'offline' }))
        wsTimer = setTimeout(connect, 1500)
      }
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data)
        const ids = inflight || []
        inflight = null
        if (!msg.probs) return
        ms = msg.ms
        finetuned = msg.finetuned
        setStatus((s) => (s.model === msg.ready ? s : { ...s, model: msg.ready }))
        applyProbs(ids, msg.probs)
      }
    }

    const startLocal = async () => {
      ws = null
      setStatus((s) => ({ ...s, mode: 'browser' }))
      try {
        await loadLocalModel()
        if (cancelled) return
        local = true
        setStatus((s) => ({ ...s, socket: 'online', model: true }))
      } catch (e) {
        setStatus((s) => ({ ...s, socket: 'offline', model: false }))
        setError(`Не удалось загрузить модель эмоций: ${e.message}`)
      }
    }

    // Сглаживание: усредняем вероятности по последним кадрам
    const applyProbs = (ids, probs) => {
      probs.forEach((p, i) => {
        const t = tracks.current.find((tr) => tr.id === ids[i])
        if (!t) return
        t.history.push(p)
        if (t.history.length > SMOOTH_WINDOW) t.history.shift()
        t.avg = average(t.history)
        const top = argmax(t.avg)
        // Ниже порога уверенности — оставляем предыдущую устойчивую эмоцию (или «?»)
        if (t.avg[top] >= CONFIDENCE_THRESHOLD) t.stable = top
      })
    }

    // ---------- Сопоставление детекций с треками (по ближайшему центру) ----------
    const updateTracks = (boxes, now) => {
      const free = [...tracks.current]
      const next = []
      for (const box of boxes) {
        const cx = box.originX + box.width / 2
        const cy = box.originY + box.height / 2
        let best = null
        let bestD = Infinity
        for (const t of free) {
          const d = Math.hypot(t.box.originX + t.box.width / 2 - cx, t.box.originY + t.box.height / 2 - cy)
          if (d < bestD && d < box.width * 0.6) { best = t; bestD = d }
        }
        if (best) {
          free.splice(free.indexOf(best), 1)
          // лёгкое сглаживание рамки, чтобы эмодзи не дрожал
          const k = 0.5
          best.box = {
            originX: best.box.originX + (box.originX - best.box.originX) * k,
            originY: best.box.originY + (box.originY - best.box.originY) * k,
            width: best.box.width + (box.width - best.box.width) * k,
            height: best.box.height + (box.height - best.box.height) * k,
          }
          best.lastSeen = now
          next.push(best)
        } else {
          next.push({ id: nextTrackId++, box, history: [], avg: null, stable: null, lastSeen: now })
        }
      }
      // кратковременно потерянные лица держим 400 мс
      for (const t of free) if (now - t.lastSeen < 400) next.push(t)
      tracks.current = next
    }

    // ---------- Руки и жесты ----------
    const updateHands = (video, now) => {
      const W = video.videoWidth
      const H = video.videoHeight
      const res = handLm.detectForVideo(video, now)
      const free = [...hands.current]
      const next = []
      for (const lm of res.landmarks) {
        const pts = lm.map((p) => ({ x: p.x * W, y: p.y * H }))
        const xs = pts.map((p) => p.x)
        const ys = pts.map((p) => p.y)
        const minX = Math.min(...xs)
        const maxX = Math.max(...xs)
        const top = Math.min(...ys)
        const size = Math.max(maxX - minX, Math.max(...ys) - top)
        const cx = (minX + maxX) / 2
        const cy = top + size / 2
        let best = null
        let bestD = Infinity
        for (const h of free) {
          const d = Math.hypot(h.cx - cx, h.cy - cy)
          if (d < bestD && d < size * 1.2) { best = h; bestD = d }
        }
        if (best) free.splice(free.indexOf(best), 1)
        else best = { id: nextTrackId++, state: newHandState(), gesture: null }
        Object.assign(best, { cx, cy, top, size, lastSeen: now })
        best.gesture = updateGesture(best.state, { pts, cx, size }, now)
        best.shape = rpsShape(best.state)
        next.push(best)
      }
      for (const h of free) if (now - h.lastSeen < 300) next.push(h)
      hands.current = next
    }

    // ---------- Главный цикл ----------
    const loop = () => {
      if (cancelled) return
      raf = requestAnimationFrame(loop)
      const video = videoRef.current
      if (!video || video.readyState < 2 || !detector) return
      const now = performance.now()
      if (handLm && tick++ % HAND_EVERY === 0) updateHands(video, now)
      const res = detector.detectForVideo(video, now)
      const boxes = res.detections
        .map((d) => d.boundingBox)
        .filter((b) => b.width > 40)
        .sort((a, b) => b.width * b.height - a.width * a.height) // приоритет — самое крупное лицо
        .slice(0, MAX_FACES)
      updateTracks(boxes, now)

      const visible = tracks.current.filter((t) => t.lastSeen === now)
      if (!visible.length) lastCrops.current = new Map()
      const canSend = local || ws?.readyState === WebSocket.OPEN
      if (visible.length && canSend && !inflight) {
        lastCrops.current = new Map()
        const buf = new Uint8Array(visible.length * 48 * 48)
        visible.forEach((t, i) => {
          const g = cropGray(video, t.box)
          lastCrops.current.set(t.id, g)
          buf.set(g, i * 48 * 48)
        })
        inflight = visible.map((t) => t.id)
        if (local) {
          const ids = inflight
          const t0 = performance.now()
          predictLocal(buf)
            .then((probs) => { ms = +(performance.now() - t0).toFixed(2); finetuned = !!getCustomHead(); applyProbs(ids, probs) })
            .catch((e) => console.error(e))
            .finally(() => { inflight = null })
        } else {
          ws.send(buf)
        }
      }

      frames++
      if (now - fpsT > 500) {
        fps = Math.round((frames * 1000) / (now - fpsT))
        frames = 0
        fpsT = now
      }
      setFrame({
        faces: tracks.current.map((t) => ({ id: t.id, box: t.box, avg: t.avg, stable: t.stable })),
        hands: hands.current.filter((h) => h.gesture).map((h) => ({ id: h.id, cx: h.cx, top: h.top, size: h.size, gesture: h.gesture })),
        // фигура для «камень, ножницы, бумага» — по самой крупной руке в кадре
        shape: hands.current.reduce((a, h) => (!a || h.size > a.size ? h : a), null)?.shape ?? null,
        fps, ms, finetuned,
        videoW: video.videoWidth, videoH: video.videoHeight,
      })
    }

    ;(async () => {
      connect()
      setError(null)
      try {
        stream = demoUrl ? await demoStream(demoUrl) : await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false,
        })
        if (cancelled) return stream.getTracks().forEach((t) => t.stop())
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        setStatus((s) => ({ ...s, camera: 'on' }))
      } catch (e) {
        setStatus((s) => ({ ...s, camera: 'error' }))
        setError(e.name === 'NotAllowedError'
          ? 'Нет доступа к камере. Разрешите доступ в настройках браузера и перезагрузите страницу.'
          : e.name === 'NotFoundError' ? 'Камера не найдена. Подключите веб-камеру.'
            : `Камера недоступна: ${e.message}`)
        return
      }
      try {
        const vision = await FilesetResolver.forVisionTasks('/mediapipe')
        // CPU-делегат: BlazeFace занимает ~3 мс, а GPU-делегат в части браузеров молча не находит лица
        detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/mediapipe/blaze_face_short_range.tflite', delegate: 'CPU' },
          runningMode: 'VIDEO', minDetectionConfidence: 0.5,
        })
        if (cancelled) return detector.close()
        setStatus((s) => ({ ...s, detector: 'ready' }))
        loop()
        // Жесты — дополнение: если модель руки не загрузилась, эмоции продолжают работать
        HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/mediapipe/hand_landmarker.task', delegate: 'CPU' },
          runningMode: 'VIDEO', numHands: MAX_HANDS,
        }).then((h) => {
          if (cancelled) return h.close()
          handLm = h
          setStatus((s) => ({ ...s, hands: 'ready' }))
        }).catch((e) => {
          console.warn('Модель руки не загрузилась:', e)
          setStatus((s) => ({ ...s, hands: 'error' }))
        })
      } catch (e) {
        setStatus((s) => ({ ...s, detector: 'error' }))
        setError(`Не удалось загрузить детектор лиц: ${e.message}`)
      }
    })()

    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
      clearTimeout(wsTimer)
      if (ws) { ws.onclose = null; ws.close() }
      stream?.getTracks().forEach((t) => t.stop())
      detector?.close()
      handLm?.close()
    }
  }, [videoRef])

  /** Серый 48x48 кроп самого крупного лица в текущем кадре (или null). */
  const captureMainFace = useCallback(() => {
    const main = tracks.current[0]
    return main ? lastCrops.current.get(main.id) ?? null : null
  }, [])

  return { status, error, frame, captureMainFace }
}
