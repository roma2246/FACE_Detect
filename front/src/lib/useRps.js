// Игра «камень, ножницы, бумага» против компьютера.
// Отсчёт 3-2-1, затем фиксируется фигура, которую рука показывает в этот момент;
// компьютер выбирает свою случайно и только после этого.
import { useCallback, useEffect, useRef, useState } from 'react'
import { RPS } from './gestures'

const SHAPES = Object.keys(RPS)
const STEP_MS = 800
const ZERO = { win: 0, lose: 0, draw: 0 }

export function useRps(shape) {
  const [game, setGame] = useState({ phase: 'idle', count: 0, round: null, score: ZERO })
  const shapeRef = useRef(shape)
  useEffect(() => { shapeRef.current = shape }, [shape])
  const timer = useRef(0)
  useEffect(() => () => clearTimeout(timer.current), [])

  const play = useCallback(() => {
    clearTimeout(timer.current)
    const step = (n) => {
      if (n > 0) {
        setGame((g) => ({ ...g, phase: 'count', count: n, round: null }))
        timer.current = setTimeout(() => step(n - 1), STEP_MS)
        return
      }
      const you = shapeRef.current
      if (!you) return setGame((g) => ({ ...g, phase: 'result', round: { you: null } }))
      const cpu = SHAPES[Math.floor(Math.random() * SHAPES.length)]
      const outcome = you === cpu ? 'draw' : RPS[you].beats === cpu ? 'win' : 'lose'
      setGame((g) => ({
        ...g, phase: 'result', round: { you, cpu, outcome },
        score: { ...g.score, [outcome]: g.score[outcome] + 1 },
      }))
    }
    step(3)
  }, [])

  const reset = useCallback(() => {
    clearTimeout(timer.current)
    setGame({ phase: 'idle', count: 0, round: null, score: ZERO })
  }, [])

  return { game, play, reset }
}
