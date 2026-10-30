import { useRef, useState } from 'react'
import { CLASSES } from '../classes'

const ACCEPT = '.csv,.txt,.npy,.hea,.dat,.atr'

export default function UploadPanel({ samples, records, loading, onFiles, onSample, onRecord }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [samplingRate, setSamplingRate] = useState(360)
  const [filter, setFilter] = useState('all')
  const [tab, setTab] = useState('beats')

  const handleFiles = (files) => {
    if (files?.length) onFiles([...files], samplingRate)
  }

  const shown = samples.filter((s) => filter === 'all' || s.label === Number(filter))
  const randomSample = () => {
    if (shown.length) onSample(shown[Math.floor(Math.random() * shown.length)].name)
  }

  return (
    <section className="card upload-card">
      <div className="card-head">
        <h2>Analyze an ECG</h2>
        <p className="muted">Upload a file or pick one of the examples.</p>
      </div>

      <div
        className={`dropzone ${dragging ? 'dragging' : ''} ${loading ? 'busy' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files) }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
      >
        <input ref={inputRef} type="file" accept={ACCEPT} multiple hidden
          onChange={(e) => { handleFiles(e.target.files); e.target.value = '' }} />
        <div className="drop-icon" aria-hidden>
          <svg viewBox="0 0 48 48" width="44" height="44">
            <path d="M24 31V13m0 0-7 7m7-7 7 7" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M8 31v4a5 5 0 0 0 5 5h22a5 5 0 0 0 5-5v-4" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
          </svg>
        </div>
        <strong>{loading ? 'Analyzing…' : 'Drop your ECG file here'}</strong>
        <span className="muted">CSV, TXT, NPY · or a PhysioNet record (.hea + .dat + .atr)</span>
      </div>

      <details className="formats">
        <summary>Supported formats</summary>
        <ul>
          <li><b>Heartbeats:</b> one beat per row, 187 values (125 Hz, scaled 0–1). An optional 188th column is used as the true label.</li>
          <li><b>Continuous recording:</b> a single column (or row) of raw ECG samples. Beats are detected automatically.</li>
          <li><b>WFDB record:</b> select the <code>.hea</code> and <code>.dat</code> files together. Add the <code>.atr</code> file to compare against the cardiologist beat annotations.</li>
        </ul>
        <label className="field">
          <span>Sampling rate of CSV/TXT recordings (WFDB records carry their own)</span>
          <div className="input-suffix">
            <input type="number" min={50} max={5000} value={samplingRate}
              onChange={(e) => setSamplingRate(Number(e.target.value) || 360)} />
            <span>Hz</span>
          </div>
        </label>
      </details>

      <div className="segmented full tabs">
        <button className={tab === 'beats' ? 'active' : ''} onClick={() => setTab('beats')}>Test heartbeats</button>
        <button className={tab === 'records' ? 'active' : ''} onClick={() => setTab('records')}>
          Real recordings {records.length > 0 && <em>{records.length}</em>}
        </button>
      </div>

      {tab === 'beats' ? (
        <>
          <div className="sample-filters" role="tablist">
            <button className={`chip ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>
              All <em>{samples.length}</em>
            </button>
            {CLASSES.map((c) => {
              const n = samples.filter((s) => s.label === c.id).length
              if (!n) return null
              return (
                <button key={c.id} className={`chip ${filter === String(c.id) ? 'active' : ''}`}
                  style={{ '--chip': c.color }} onClick={() => setFilter(String(c.id))}>
                  <i className="dot" /> {c.short} <em>{n}</em>
                </button>
              )
            })}
          </div>

          <div className="sample-list">
            {shown.map((s) => {
              const c = CLASSES[s.label]
              return (
                <button key={s.name} className="sample-item" disabled={loading} onClick={() => onSample(s.name)}
                  style={{ '--chip': c?.color }}>
                  <i className="dot" />
                  <span className="mono">#{s.name.match(/\d+/)?.[0]}</span>
                  <span>{c?.name ?? 'Unlabelled'}</span>
                </button>
              )
            })}
            {!samples.length && <p className="muted small">No test samples found on the server.</p>}
          </div>

          <button className="btn btn-primary block" disabled={loading || !shown.length} onClick={randomSample}>
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden><path d="M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
            Analyze a random sample
          </button>
        </>
      ) : (
        <div className="record-list">
          <p className="muted small">
            Full 30-minute recordings from the MIT-BIH Arrhythmia Database, with cardiologist beat annotations.
          </p>
          {records.map((r) => (
            <button key={r.name} className="record-item" disabled={loading} onClick={() => onRecord(r.name)}>
              <span className="record-id mono">{r.name}</span>
              <span className="record-text">
                <strong>Record {r.name}</strong>
                <span>{r.note || `${Math.round(r.duration_sec / 60)} min recording`}</span>
                <em>{Math.round(r.duration_sec / 60)} min · {r.sampling_rate} Hz{r.annotated ? ' · annotated' : ''}</em>
              </span>
              <span className="record-go" aria-hidden>→</span>
            </button>
          ))}
          {!records.length && <p className="muted small">No recordings found in examples/mitbih on the server.</p>}
        </div>
      )}
    </section>
  )
}
