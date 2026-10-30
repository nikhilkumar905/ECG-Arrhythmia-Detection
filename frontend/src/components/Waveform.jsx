import { useMemo } from 'react'
import { CLASSES } from '../classes'

const FS = 125

function toPath(values, width, height, pad = 8, min = Math.min(...values, 0), max = Math.max(...values, 1)) {
  const span = max - min || 1
  const dx = width / Math.max(values.length - 1, 1)
  let d = ''
  for (let i = 0; i < values.length; i++) {
    const y = pad + (1 - (values[i] - min) / span) * (height - pad * 2)
    d += `${i ? 'L' : 'M'}${(i * dx).toFixed(1)},${y.toFixed(1)}`
  }
  return d
}

function Grid({ width, height, minor = 10 }) {
  const lines = []
  for (let x = 0; x <= width; x += minor) {
    lines.push(<line key={`x${x}`} x1={x} x2={x} y1={0} y2={height} className={x % (minor * 5) ? 'grid-minor' : 'grid-major'} />)
  }
  for (let y = 0; y <= height; y += minor) {
    lines.push(<line key={`y${y}`} y1={y} y2={y} x1={0} x2={width} className={y % (minor * 5) ? 'grid-minor' : 'grid-major'} />)
  }
  return <g>{lines}</g>
}

// A single 187-sample heartbeat on ECG paper, optionally with the model's attention as a heat overlay
export function BeatWaveform({ values, color = 'var(--accent)', attention, height = 240 }) {
  const width = 750
  const path = useMemo(() => toPath(values, width, height), [values, height])
  const step = width / 187
  // Each of the 23 LSTM time steps sees ~8 input samples (187 -> 93 -> 46 -> 23 after pooling)
  const span = 187 / (attention?.length || 23)

  return (
    <div className="ecg-paper">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="Heartbeat waveform">
        <Grid width={width} height={height} />
        {attention && (
          <g className="attention">
            {attention.map((w, t) => (
              <rect key={t} x={t * span * step} width={span * step + 0.5} y={0} height={height}
                fill={color} fillOpacity={0.05 + 0.45 * w * w} />
            ))}
          </g>
        )}
        <path key={path.slice(0, 60)} d={path} className="trace" style={{ stroke: color }} />
      </svg>
      {attention && (
        <div className="attention-strip" aria-hidden>
          {attention.map((w, t) => <i key={t} style={{ opacity: 0.08 + 0.92 * w, background: color }} />)}
        </div>
      )}
      <div className="axis-note">
        <span>0 ms</span>
        <span>187 samples · 125 Hz</span>
        <span>1496 ms</span>
      </div>
    </div>
  )
}

const WINDOW_SEC = 12

// A window of the recording around the selected beat, with beat markers colored by prediction,
// plus an overview of the whole recording for navigation
export function SignalStrip({ signal, peaks, results, selected, onSelect }) {
  const height = 210
  const width = 1200
  const winLen = WINDOW_SEC * FS
  const center = peaks[selected] ?? 0
  const start = Math.max(0, Math.min(center - winLen / 2, signal.length - winLen))
  const end = Math.min(signal.length, start + winLen)

  const path = useMemo(
    () => toPath(signal.slice(start, end), width * ((end - start) / winLen), height, 14, 0, 1),
    [signal, start, end, winLen],
  )
  const xOf = (p) => ((p - start) / winLen) * width

  const visible = []
  for (let i = 0; i < peaks.length; i++) {
    if (peaks[i] >= start && peaks[i] < end) visible.push(i)
  }

  const total = signal.length
  const pickFromOverview = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const target = ((e.clientX - rect.left) / rect.width) * total
    let best = 0
    for (let i = 0; i < peaks.length; i++) {
      if (Math.abs(peaks[i] - target) < Math.abs(peaks[best] - target)) best = i
    }
    onSelect(best)
  }
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

  return (
    <div>
      <div className="ecg-paper strip">
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="ECG recording window">
          <Grid width={width} height={height} minor={width / (WINDOW_SEC * 5)} />
          {visible.map((i) => {
            const r = results[i]
            const cls = CLASSES[r.prediction]
            const x = xOf(peaks[i])
            const isSel = selected === i
            return (
              <g key={i} className="peak" onClick={() => onSelect(i)}>
                <title>{`Beat ${i + 1}: ${cls.name} (${(r.confidence * 100).toFixed(1)}%)${r.needs_review ? ' · needs review' : ''}`}</title>
                <rect x={x - 14} y={0} width={28} height={height} className={isSel ? 'peak-hit selected' : 'peak-hit'} />
                <line x1={x} x2={x} y1={0} y2={height} stroke={cls.color} strokeOpacity={isSel ? 0.9 : 0.3} strokeDasharray="3 4" vectorEffect="non-scaling-stroke" />
                {r.needs_review && <circle cx={x} cy={11} r={9} fill="none" stroke="var(--c-supra)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />}
                <circle cx={x} cy={11} r={isSel ? 6 : 4.5} fill={cls.color} />
              </g>
            )
          })}
          <path d={path} className="trace" style={{ stroke: 'var(--trace)' }} />
        </svg>
        <div className="axis-note">
          <span>{fmt(start / FS)}</span>
          <span>{WINDOW_SEC} s window · 125 Hz</span>
          <span>{fmt(end / FS)}</span>
        </div>
      </div>

      {total > winLen && (
        <div className="overview" onClick={pickFromOverview} title="Click to jump to a beat">
          <svg viewBox={`0 0 ${total} 30`} preserveAspectRatio="none">
            {peaks.map((p, i) => {
              const r = results[i]
              if (r.prediction === 0 && !r.needs_review) return null
              return (
                <rect key={i} x={p} y={r.prediction === 0 ? 20 : 4} width={total / 1500} height={r.prediction === 0 ? 10 : 26}
                  fill={r.prediction === 0 ? 'var(--c-supra)' : CLASSES[r.prediction].color} />
              )
            })}
          </svg>
          <div className="overview-window" style={{ left: `${(start / total) * 100}%`, width: `${((end - start) / total) * 100}%` }} />
          <div className="overview-labels"><span>0:00</span><span>Abnormal &amp; flagged beats across the recording</span><span>{fmt(total / FS)}</span></div>
        </div>
      )}
    </div>
  )
}
