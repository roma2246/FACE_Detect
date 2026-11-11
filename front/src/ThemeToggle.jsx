// Тумблер светлой/тёмной темы: солнце с облаками <-> луна со звёздами.
import './ThemeToggle.css'

export default function ThemeToggle({ theme, onChange }) {
  const dark = theme === 'dark'
  return (
    <button
      type="button"
      role="switch"
      aria-checked={dark}
      aria-label={dark ? 'Включить светлую тему' : 'Включить тёмную тему'}
      title={dark ? 'Светлая тема' : 'Тёмная тема'}
      className={`theme-toggle ${dark ? 'is-dark' : ''}`}
      onClick={() => onChange(dark ? 'light' : 'dark')}
    >
      <span className="tt-sky">
        <span className="tt-stars"><i /><i /><i /><i /><i /></span>
        <span className="tt-clouds"><i /><i /><i /></span>
      </span>
      <span className="tt-knob">
        <span className="tt-craters"><i /><i /><i /></span>
      </span>
    </button>
  )
}
