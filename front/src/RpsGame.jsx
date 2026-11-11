// Интерфейс игры «камень, ножницы, бумага»: карточка в правой панели и надпись поверх видео.
import { RPS } from './lib/gestures'

const COUNT_WORDS = { 3: 'Камень…', 2: 'Ножницы…', 1: 'Бумага!' }
const OUTCOME = { win: 'Вы выиграли!', lose: 'Компьютер выиграл', draw: 'Ничья' }

/** Отсчёт и результат раунда поверх видео. */
export function RpsOverlay({ game }) {
  const { phase, count, round } = game
  if (phase === 'count') {
    return (
      <div className="rps-overlay">
        <span key={count} className="rps-count">{count}</span>
        <span className="rps-word">{COUNT_WORDS[count]}</span>
      </div>
    )
  }
  if (phase === 'result' && round?.you) {
    return (
      <div className="rps-overlay result">
        <span className="rps-versus">{RPS[round.you].emoji} <i>vs</i> {RPS[round.cpu].emoji}</span>
        <span className="rps-word">{OUTCOME[round.outcome]}</span>
      </div>
    )
  }
  return null
}

export function RpsCard({ rps, shape, ready }) {
  const { game, play, reset } = rps
  const { phase, round, score } = game
  const counting = phase === 'count'
  const played = phase === 'result' && round?.you
  const you = played ? RPS[round.you] : shape ? RPS[shape] : null
  const cpu = played ? RPS[round.cpu] : null

  let text = 'Нажмите «Играть» и на счёт «Бумага!» покажите фигуру рукой.'
  if (!ready) text = 'Загружается модель руки…'
  else if (counting) text = 'Готовьте руку…'
  else if (played) text = OUTCOME[round.outcome]
  else if (phase === 'result') text = 'Не вижу руку — покажите фигуру в камеру и сыграйте ещё раз.'

  return (
    <section className="card">
      <div className="card-title">
        <h3>Камень, ножницы, бумага</h3>
        <button className="link" onClick={reset}>сбросить</button>
      </div>
      <div className="rps-row">
        <div className="rps-side">
          <span className="rps-emoji" key={you?.emoji}>{you ? you.emoji : '🖐️'}</span>
          <span className="muted small">{you ? you.label : 'Вы'}</span>
        </div>
        <div className="rps-score">{score.win} : {score.lose}</div>
        <div className="rps-side">
          <span className="rps-emoji" key={cpu?.emoji}>{cpu ? cpu.emoji : '🤖'}</span>
          <span className="muted small">{cpu ? cpu.label : 'Компьютер'}</span>
        </div>
      </div>
      <p className={`rps-text small ${played ? round.outcome : 'muted'}`}>{text}</p>
      <button className="btn primary rps-play" onClick={play} disabled={!ready || counting}>
        {counting ? 'Идёт отсчёт…' : phase === 'result' ? 'Ещё раунд' : 'Играть'}
      </button>
      <div className="legend">
        <span>✊ камень</span><span>✌️ ножницы</span><span>✋ бумага</span><span>ничьих: {score.draw}</span>
      </div>
    </section>
  )
}
