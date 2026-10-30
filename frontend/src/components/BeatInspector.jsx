import { useState } from 'react'
import { CLASSES, pct, reviewReason } from '../classes'
import { BeatWaveform } from './Waveform'

function ConfidenceRing({ value, color }) {
  const r = 42
  const c = 2 * Math.PI * r
  return (
    <div className="ring">
      <svg viewBox="0 0 100 100" width="108" height="108">
        <circle cx="50" cy="50" r={r} className="ring-track" />
        <circle cx="50" cy="50" r={r} className="ring-value" stroke={color}
          strokeDasharray={c} strokeDashoffset={c * (1 - value)} />
      </svg>
      <div className="ring-label">
        <strong>{(value * 100).toFixed(1)}<small>%</small></strong>
        <span>confidence</span>
      </div>
    </div>
  )
}

export default function BeatInspector({ beat, values, total, onPrev, onNext }) {
  const [showAttention, setShowAttention] = useState(true)
  const cls = CLASSES[beat.prediction]
  const truth = beat.true_label != null ? CLASSES[beat.true_label] : null
  const correct = truth && truth.id === beat.prediction

  // Where the model looked: the time step with the highest attention, in ms from the R-peak
  const peakStep = beat.attention.indexOf(Math.max(...beat.attention))
  const stepMs = (187 / beat.attention.length / 125) * 1000
  const focusFrom = Math.round(peakStep * stepMs)

  return (
    <section className="card inspector" style={{ '--cls': cls.color }}>
      <div className="inspector-head">
        <div className="verdict">
          <span className="eyebrow">{total > 1 ? `Beat ${beat.index + 1} of ${total}` : 'Prediction'}</span>
          <h2 className="verdict-name">
            <span className="code-badge">{cls.code}</span>
            {cls.name}
          </h2>
          <p className="muted">{cls.description}</p>
          <div className="tags">
            {truth && (
              <span className={`truth ${correct ? 'ok' : 'bad'}`}>
                {correct ? '✓ Matches' : '✗ Differs from'} {beat.symbol ? 'annotation' : 'true label'}: <b>{truth.name}</b>
                {beat.symbol && <span className="mono"> ({beat.symbol})</span>}
              </span>
            )}
            {beat.needs_review && (
              <span className="review-badge" title="Predictions like this one should be checked by a person.">
                ⚠ Needs review · {reviewReason(beat)}
              </span>
            )}
          </div>
        </div>
        <ConfidenceRing value={beat.confidence} color={beat.needs_review ? 'var(--c-supra)' : cls.color} />
      </div>

      <div className="row-between tight">
        <h3>Waveform</h3>
        <label className="switch">
          <input type="checkbox" checked={showAttention} onChange={(e) => setShowAttention(e.target.checked)} />
          <span className="switch-track"><span /></span>
          Show model attention
        </label>
      </div>
      <BeatWaveform values={values} color={cls.color} attention={showAttention ? beat.attention : null} />
      {showAttention && (
        <p className="attention-note muted">
          Shaded regions show where the attention layer focused, strongest around{' '}
          <b>{focusFrom}–{Math.round(focusFrom + stepMs)} ms</b> after the R-peak. Each band covers ~8 samples
          (64 ms); because the LSTM mixes context, treat this as an approximate guide, not an exact explanation.
        </p>
      )}

      <div className="probs">
        <h3>Class probabilities</h3>
        {beat.probabilities.map((p, i) => {
          const c = CLASSES[i]
          return (
            <div key={c.id} className={`prob-row ${c.id === beat.prediction ? 'top' : ''}`} style={{ '--bar': c.color }}>
              <span className="prob-name"><i className="dot" />{c.name}</span>
              <div className="prob-track"><div className="prob-fill" style={{ width: `${Math.max(p * 100, 0.4)}%` }} /></div>
              <span className="prob-val mono">{pct(p, p < 0.001 ? 2 : 1)}</span>
            </div>
          )
        })}
      </div>

      {total > 1 && (
        <div className="pager">
          <button className="btn btn-ghost" onClick={onPrev} disabled={beat.index === 0}>← Previous beat</button>
          <button className="btn btn-ghost" onClick={onNext} disabled={beat.index === total - 1}>Next beat →</button>
        </div>
      )}
    </section>
  )
}
