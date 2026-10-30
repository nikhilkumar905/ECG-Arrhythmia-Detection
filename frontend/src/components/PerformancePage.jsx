import { useEffect, useState } from 'react'
import { figureUrl, getPerformance } from '../api'
import { CLASSES, pct } from '../classes'
import ConfusionMatrix from './ConfusionMatrix'

function MetricBar({ value, color }) {
  return (
    <div className="metric-cell">
      <div className="mini-conf wide"><div style={{ width: pct(value), background: color }} /></div>
      <span className="mono">{pct(value)}</span>
    </div>
  )
}

export default function PerformancePage() {
  const [perf, setPerf] = useState(null)
  const [error, setError] = useState('')
  const [zoom, setZoom] = useState(null)

  useEffect(() => {
    getPerformance().then(setPerf).catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="alert" role="alert"><b>Could not load performance.</b> {error}</div>
  if (!perf) return <div className="card empty"><p className="muted">Loading model performance…</p></div>

  const errors = perf.total - Math.round(perf.accuracy * perf.total)
  const worst = [...perf.per_class].sort((a, b) => a.recall - b.recall)[0]

  return (
    <div className="result-stack perf">
      <div className="result-bar">
        <div>
          <span className="eyebrow">Held-out test set</span>
          <h2 className="file-name">Model performance</h2>
          <p className="muted small">{perf.model} · {perf.dataset} · beats never seen during training or model selection</p>
        </div>
      </div>

      <div className="stats">
        <div className="stat good"><span className="stat-label">Accuracy</span><strong className="stat-value">{pct(perf.accuracy, 2)}</strong><span className="stat-hint">{errors} errors in {perf.total.toLocaleString()} beats</span></div>
        <div className="stat"><span className="stat-label">Macro F1</span><strong className="stat-value">{pct(perf.macro_f1, 1)}</strong><span className="stat-hint">every class weighted equally</span></div>
        <div className="stat"><span className="stat-label">Weighted F1</span><strong className="stat-value">{pct(perf.weighted_f1, 2)}</strong><span className="stat-hint">weighted by class size</span></div>
        <div className="stat warn"><span className="stat-label">Weakest class</span><strong className="stat-value">{CLASSES[worst.id].name}</strong><span className="stat-hint">recall {pct(worst.recall)}</span></div>
      </div>

      <section className="card">
        <h3>Per-class metrics</h3>
        <div className="table-wrap">
          <table className="metrics-table">
            <thead>
              <tr><th>Class</th><th>Precision</th><th>Recall (sensitivity)</th><th>Specificity</th><th>F1</th><th>Test beats</th></tr>
            </thead>
            <tbody>
              {perf.per_class.map((m) => {
                const c = CLASSES[m.id]
                return (
                  <tr key={m.id}>
                    <td><span className="pill" style={{ '--chip': c.color }}><i className="dot" />{c.name}</span></td>
                    <td><MetricBar value={m.precision} color={c.color} /></td>
                    <td><MetricBar value={m.recall} color={c.color} /></td>
                    <td><MetricBar value={m.specificity} color={c.color} /></td>
                    <td><MetricBar value={m.f1} color={c.color} /></td>
                    <td className="mono">{m.support.toLocaleString()}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <ul className="explain">
          <li><b>Precision</b> — when the model says this class, how often it is right.</li>
          <li><b>Recall</b> — of all true beats of this class, how many it finds.</li>
          <li><b>Specificity</b> — of all beats <i>not</i> in this class, how many it correctly rejects.</li>
        </ul>
      </section>

      <section className="card">
        <h3>Confusion matrix</h3>
        <ConfusionMatrix matrix={perf.confusion_matrix} />
        <p className="muted small note">
          The test set is heavily imbalanced (83% Normal), so overall accuracy flatters the model — the
          Supraventricular row is where most mistakes happen, with ~12% of those beats called Normal.
        </p>
      </section>

      {perf.figures.length > 0 && (
        <section className="card">
          <h3>Training figures</h3>
          <div className="figures">
            {perf.figures.map((f) => (
              <button key={f.file} className="figure" onClick={() => setZoom(f)}>
                <img src={figureUrl(f.file)} alt={f.title} loading="lazy" />
                <span>{f.title}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)} role="dialog" aria-label={zoom.title}>
          <img src={figureUrl(zoom.file)} alt={zoom.title} />
          <span>{zoom.title} · click anywhere to close</span>
        </div>
      )}
    </div>
  )
}
