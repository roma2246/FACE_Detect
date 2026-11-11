// Фон из частиц (three.js): поле точек, которое плавно меняет форму под текущую эмоцию.
//   нет лица / нейтрально — спокойная волна      радость   — волна «улыбкой», края вверх
//   грусть — медленная волна, края вниз          злость    — резкие острые гребни
//   удивление — круги, расходящиеся из центра
// Все формы считаются в вершинном шейдере; на CPU только плавно меняются веса форм и цвет.
import { memo, useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EMOTIONS } from './lib/emotions'

const COLS = 170
const ROWS = 80
const FORMS = ['calm', 'happy', 'sad', 'angry', 'surprise']
const FORM_OF = { happy: 'happy', sad: 'sad', angry: 'angry', surprise: 'surprise', neutral: 'calm' }

const IDLE_COLOR = { dark: '#b4b4bb', light: '#52525b' }

const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uCalm, uHappy, uSad, uAngry, uSurprise;
  uniform float uSize;
  varying float vFade;
  varying float vHeight;

  void main() {
    vec3 p = position;
    float x = p.x, z = p.z, t = uTime;

    float calm = sin(x * 0.45 + t * 0.6) * 0.35 + cos(z * 0.55 + t * 0.45) * 0.3;
    float happy = sin(x * 0.7 + t * 1.5) * 0.4 + cos(z * 0.9 + t * 1.2) * 0.3 + x * x * 0.022;
    float sad = sin(x * 0.3 + t * 0.3) * 0.25 + cos(z * 0.35 + t * 0.25) * 0.2 - x * x * 0.02;
    float saw = abs(fract(x * 0.28 + z * 0.12 + t * 0.55) * 2.0 - 1.0);
    float angry = (1.0 - saw * saw) * 0.95 - 0.45 + sin(z * 1.6 + t * 2.4) * 0.12;
    float r = length(vec2(x, z + 2.0));
    float surprise = sin(r * 1.5 - t * 2.6) * 0.75 / (1.0 + r * 0.16);

    float h = calm * uCalm + happy * uHappy + sad * uSad + angry * uAngry + surprise * uSurprise;
    p.y += h;
    vHeight = h;

    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = -mv.z;
    gl_PointSize = uSize * (1.0 + h * 0.35) / depth;
    // дальние точки и края поля плавно исчезают
    vFade = smoothstep(26.0, 7.0, depth) * smoothstep(17.0, 11.0, abs(x));
  }
`

const fragmentShader = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity;
  uniform float uGlow;
  varying float vFade;
  varying float vHeight;

  void main() {
    float d = length(gl_PointCoord - 0.5);
    float dot = smoothstep(0.5, 0.12, d);
    vec3 col = uColor * (1.0 + vHeight * uGlow);
    gl_FragColor = vec4(col, dot * vFade * uOpacity);
  }
`

function ParticleBackground({ emotion, theme }) {
  const canvasRef = useRef(null)
  const target = useRef({ form: 'calm', color: new THREE.Color(IDLE_COLOR.dark), theme })

  // Цель (форма и цвет) обновляется без пересоздания сцены
  useEffect(() => {
    target.current.form = FORM_OF[emotion] || 'calm'
    target.current.color.set(emotion ? EMOTIONS[emotion].color : IDLE_COLOR[theme])
    // на светлом фоне яркие цвета (жёлтый) плохо видны — затемняем
    if (theme === 'light') target.current.color.multiplyScalar(emotion ? 0.72 : 1)
    target.current.theme = theme
  }, [emotion, theme])

  useEffect(() => {
    const canvas = canvasRef.current
    let renderer
    try {
      renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false, powerPreference: 'low-power' })
    } catch {
      return // WebGL недоступен — остаётся обычный фон
    }
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 60)
    camera.position.set(0, 3.2, 9)
    camera.lookAt(0, 0.4, -3)

    const positions = new Float32Array(COLS * ROWS * 3)
    let i = 0
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        positions[i++] = (c / (COLS - 1) - 0.5) * 36
        positions[i++] = -1.6
        positions[i++] = 5 - (r / (ROWS - 1)) * 24
      }
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 100)

    const uniforms = {
      uTime: { value: 0 }, uSize: { value: 60 },
      uCalm: { value: 1 }, uHappy: { value: 0 }, uSad: { value: 0 }, uAngry: { value: 0 }, uSurprise: { value: 0 },
      uColor: { value: new THREE.Color(IDLE_COLOR.dark) }, uOpacity: { value: 0.8 }, uGlow: { value: 0.5 },
    }
    const material = new THREE.ShaderMaterial({ vertexShader, fragmentShader, uniforms, transparent: true, depthWrite: false })
    const points = new THREE.Points(geometry, material)
    scene.add(points)

    const resize = () => {
      const w = window.innerWidth
      const h = window.innerHeight
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      renderer.setPixelRatio(dpr)
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      uniforms.uSize.value = 34 * dpr * (h / 900)
    }
    resize()
    window.addEventListener('resize', resize)

    const pointer = { x: 0, y: 0 }
    const onMove = (e) => {
      pointer.x = e.clientX / window.innerWidth - 0.5
      pointer.y = e.clientY / window.innerHeight - 0.5
    }
    window.addEventListener('pointermove', onMove)

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const key = (f) => `u${f[0].toUpperCase()}${f.slice(1)}`
    let raf = 0
    let last = performance.now()
    let appliedTheme = null

    const frame = (now) => {
      raf = requestAnimationFrame(frame)
      const dt = Math.min((now - last) / 1000, 0.1)
      last = now
      const t = target.current
      if (t.theme !== appliedTheme) {
        appliedTheme = t.theme
        const dark = t.theme === 'dark'
        material.blending = dark ? THREE.AdditiveBlending : THREE.NormalBlending
        uniforms.uOpacity.value = dark ? 0.75 : 0.85
        uniforms.uGlow.value = dark ? 0.55 : 0.0
        material.needsUpdate = true
      }
      // плавный переход между формами и цветами
      const k = 1 - Math.exp(-dt * 2.2)
      for (const f of FORMS) {
        const u = uniforms[key(f)]
        u.value += ((t.form === f ? 1 : 0) - u.value) * k
      }
      uniforms.uColor.value.lerp(t.color, k)
      if (!reduced) uniforms.uTime.value += dt
      points.rotation.y += (pointer.x * 0.12 - points.rotation.y) * 0.04
      camera.position.y += (3.2 - pointer.y * 0.8 - camera.position.y) * 0.04
      camera.lookAt(0, 0.4, -3)
      renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(frame)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      window.removeEventListener('pointermove', onMove)
      geometry.dispose()
      material.dispose()
      renderer.dispose()
    }
  }, [])

  return <canvas ref={canvasRef} className="particles" aria-hidden="true" />
}

export default memo(ParticleBackground)
