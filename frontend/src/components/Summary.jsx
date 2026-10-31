import { useMemo, useState } from 'react'
import { CLASSES, pct, reviewReason } from '../classes'
import ConfusionMatrix from './ConfusionMatrix'

function Stat({ label, value, hint, tone }) {
  return (
    <div className={`stat ${tone ?? ''}`}>
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  )
}

export function SummaryStats({ data }) {
  const { summary, num_beats } = data
  return (
    <div className="stats">
      <Stat label="Beats analyzed" value={num_beats.toLocaleString()} hint={`${data.inference_ms} ms inference`} />
      <Stat label="Abnormal beats" value={summary.abnormal_beats.toLocaleString()}
        hint={pct(summary.abnormal_ratio)} tone={summary.abnormal_beats ? 'warn' : 'good'} />
      <Stat label="Need review" value={summary.review_beats.toLocaleString()}
        hint={`mean confidence ${pct(summary.mean_confidence)}`} tone={summary.review_beats ? 'warn' : 'good'} />
      {summary.accuracy != null ? (
        <Stat label="Accuracy vs labels" value={pct(summary.accuracy)}
          hint={`${summary.labelled_beats.toLocaleString()} labelled beats`} tone={summary.accuracy > 0.95 ? 'good' : 'warn'} />
      ) : (
        <Stat label="Dominant class" value={CLASSES[summary.dominant_class].name} />
      )}
    </div>
  )
}

export function Distribution({ counts, total }) {
  return (
    <section className="card">
      <h3>Beat distribution</h3>
      <div className="dist-bar">
        {counts.map((n, i) => n > 0 && (
          <div key={i} style={{ flexGrow: n, background: CLASSES[i].color }} title={`${CLASSES[i].name}: ${n}`} />
        ))}
      </div>
      <div className="dist-legend">
        {counts.map((n, i) => (
          <div key={i} className={`dist-item ${n ? '' : 'zero'}`} style={{ '--chip': CLASSES[i].color }}>
            <i className="dot" />
            <span>{CLASSES[i].name}</span>
            <b className="mono">{n}</b>
            <span className="muted mono">{pct(n / total)}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

export function LabelledConfusion({ data, onCellClick }) {
  return (
    <section className="card">
      <div className="row-between">
        <h3>Predictions vs true labels</h3>
        <span className="muted small">Click a cell to list those beats</span>
      </div>
      <ConfusionMatrix matrix={data.summary.confusion_matrix} onCellClick={onCellClick} />
    </section>
  )
}

const PAGE = 40

export function BeatTable({ results, selected, onSelect, cellFilter, onClearCell }) {
  const [chosenFilter, setFilter] = useState('all')
  const filter = cellFilter ? 'cell' : chosenFilter
  const [limit, setLimit] = useState(PAGE)
  const hasLabels = results.some((r) => r.true_label != null)
  const hasReview = results.some((r) => r.needs_review)

  const rows = useMemo(() => results.filter((r) => {
    if (filter === 'abnormal') return r.prediction !== 0
    if (filter === 'review') return r.needs_review
    if (filter === 'errors') return r.true_label != null && r.true_label !== r.prediction
    if (filter === 'cell' && cellFilter) return r.true_label === cellFilter[0] && r.prediction === cellFilter[1]
    return true
  }), [results, filter, cellFilter])

  const setF = (f) => { setFilter(f); setLimit(PAGE); onClearCell?.() }

  return (
    <section className="card">
      <div className="row-between">
        <h3>All beats <span className="muted small">· {rows.length.toLocaleString()} shown</span></h3>
        <div className="segmented">
          <button className={filter === 'all' ? 'active' : ''} onClick={() => setF('all')}>All</button>
          <button className={filter === 'abnormal' ? 'active' : ''} onClick={() => setF('abnormal')}>Abnormal</button>
          {hasReview && <button className={filter === 'review' ? 'active' : ''} onClick={() => setF('review')}>Needs review</button>}
          {hasLabels && <button className={filter === 'errors' ? 'active' : ''} onClick={() => setF('errors')}>Mismatches</button>}
          {cellFilter && (
            <button className="active" onClick={() => setF('all')} title="Clear">
              {CLASSES[cellFilter[0]].short} → {CLASSES[cellFilter[1]].short} ×
            </button>
          )}
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Prediction</th><th>Confidence</th>{hasLabels && <th>True label</th>}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, limit).map((r) => {
              const c = CLASSES[r.prediction]
              const t = r.true_label != null ? CLASSES[r.true_label] : null
              return (
                <tr key={r.index} className={selected === r.index ? 'selected' : ''} onClick={() => onSelect(r.index)}>
                  <td className="mono">{r.index + 1}</td>
                  <td>
                    <span className="pill" style={{ '--chip': c.color }}><i className="dot" />{c.name}</span>
                    {r.needs_review && <span className="review-dot" title={reviewReason(r)}>⚠</span>}
                  </td>
                  <td>
                    <div className="mini-conf"><div style={{ width: pct(r.confidence), background: c.color }} /></div>
                    <span className="mono small">{pct(r.confidence)}</span>
                  </td>
                  {hasLabels && (
                    <td className={t && t.id !== r.prediction ? 'mismatch' : 'muted'}>
                      {t ? t.name : '—'} {t && (t.id === r.prediction ? '✓' : '✗')}
                    </td>
                  )}
                </tr>
              )
            })}
            {!rows.length && <tr><td colSpan={4} className="muted center">No beats match this filter.</td></tr>}
          </tbody>
        </table>
      </div>
      {rows.length > limit && (
        <button className="btn btn-ghost block" onClick={() => setLimit(limit + PAGE)}>
          Show more ({(rows.length - limit).toLocaleString()} remaining)
        </button>
      )}
    </section>
  )
}
