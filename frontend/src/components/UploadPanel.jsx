import { useRef, useState } from 'react'
import { getSampleValues } from '../api'
import { CLASSES } from '../classes'

const ACCEPT = '.csv,.txt,.npy,.hea,.dat,.atr'

function Sparkline({ values }) {
  const w = 300, h = 70
  const end = values.findLastIndex((v) => v !== 0) + 4
  const pts = values.slice(0, Math.max(end, 20))
  const d = pts.map((v, i) => `${i ? 'L' : 'M'}${((i / (pts.length - 1)) * w).toFixed(1)},${(4 + (1 - v) * (h - 8)).toFixed(1)}`).join('')
  return (
    <svg className="sparkline" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden>
      <path d={d} />
    </svg>
  )
}

const formatSize = (bytes) => (bytes < 1024 ? `${bytes} B` : bytes < 2 ** 20 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 2 ** 20).toFixed(1)} MB`)

export default function UploadPanel({ samples, loading, onFiles, onSample }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [samplingRate, setSamplingRate] = useState(360)
  const [randomClass, setRandomClass] = useState('any')
  // The input waiting to be analyzed: uploaded files or a random MIT-BIH beat
  const [selected, setSelected] = useState(null)
  const [loadingSample, setLoadingSample] = useState(false)
  const [sampleError, setSampleError] = useState('')

  const chooseFiles = (files) => {
    if (!files?.length) return
    const list = [...files]
    setSampleError('')
    setSelected({
      kind: 'files',
      files: list,
      title: list.length === 1 ? list[0].name : `${list.length} files`,
      detail: list.length === 1 ? formatSize(list[0].size) : list.map((f) => f.name).join(', '),
    })
  }

  const pool = samples.filter((s) => randomClass === 'any' || s.label === Number(randomClass))

  const loadRandom = async () => {
    if (!pool.length) return
    const pick = pool[Math.floor(Math.random() * pool.length)]
    setLoadingSample(true)
    setSampleError('')
    try {
      const values = await getSampleValues(pick.name)
      setSelected({
        kind: 'sample',
        name: pick.name,
        values,
        title: `MIT-BIH test beat #${pick.name.match(/\d+/)?.[0] ?? '?'}`,
        detail: 'True class shown after analysis',
      })
    } catch (e) {
      setSampleError(e.message)
    } finally {
      setLoadingSample(false)
    }
  }

  const analyze = () => {
    if (!selected) return
    if (selected.kind === 'files') onFiles(selected.files, samplingRate)
    else onSample(selected.name)
  }

  return (
    <section className="card upload-card">
      <div className="card-head">
        <h2>Analyze an ECG</h2>
        <p className="muted">Upload your own file or load a random MIT-BIH heartbeat.</p>
      </div>

      <div
        className={`dropzone ${dragging ? 'dragging' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); chooseFiles(e.dataTransfer.files) }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
      >
        <input ref={inputRef} type="file" accept={ACCEPT} multiple hidden
          onChange={(e) => { chooseFiles(e.target.files); e.target.value = '' }} />
        <div className="drop-icon" aria-hidden>
          <svg viewBox="0 0 48 48" width="44" height="44">
            <path d="M24 31V13m0 0-7 7m7-7 7 7" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M8 31v4a5 5 0 0 0 5 5h22a5 5 0 0 0 5-5v-4" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
          </svg>
        </div>
        <strong>Drop your ECG file here</strong>
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

      <div className="divider"><span>or</span></div>

      <div className="random-box">
        <div className="random-head">
          <strong>Random MIT-BIH beat</strong>
          <span className="muted small">A heartbeat from the held-out test set</span>
        </div>
        <div className="random-controls">
          <label className="select-wrap">
            <span className="sr-only">Class</span>
            <select value={randomClass} onChange={(e) => setRandomClass(e.target.value)} disabled={!samples.length}>
              <option value="any">Any class</option>
              {CLASSES.filter((c) => samples.some((s) => s.label === c.id)).map((c) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
          </label>
          <button className="btn btn-ghost" onClick={loadRandom} disabled={!pool.length || loadingSample}>
            <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden><path d="M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
            {loadingSample ? 'Loading…' : 'Load random beat'}
          </button>
        </div>
        {!samples.length && <p className="muted small">Example beats are not available on the server.</p>}
        {sampleError && <p className="error-text small">{sampleError}</p>}
      </div>

      <div className={`selected ${selected ? '' : 'empty'}`}>
        {selected ? (
          <>
            <div className="selected-head">
              <span className="selected-icon" aria-hidden>
                <svg viewBox="0 0 24 24" width="20" height="20">
                  {selected.kind === 'files'
                    ? <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                    : <path d="M3 12h4l3-8 4 16 3-8h4" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />}
                </svg>
              </span>
              <div className="selected-text">
                <strong title={selected.title}>{selected.title}</strong>
                <span className="muted small" title={selected.detail}>{selected.detail}</span>
              </div>
              <button className="icon-btn" onClick={() => setSelected(null)} aria-label="Clear selection" title="Clear">×</button>
            </div>
            {selected.kind === 'sample' && <Sparkline values={selected.values} />}
          </>
        ) : (
          <span className="muted small">No input selected yet: upload a file or load a random beat.</span>
        )}
      </div>

      <button className="btn btn-primary block" disabled={!selected || loading} onClick={analyze}>
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden><path d="M3 12h4l3-8 4 16 3-8h4" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg>
        {loading ? 'Analyzing…' : 'Analyze'}
      </button>
    </section>
  )
}
