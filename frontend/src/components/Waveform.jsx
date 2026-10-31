import { useMemo } from 'react'

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
