import { useEffect, useRef, useState } from 'react'
import { getRecords, getSamples, predictFiles, predictRecord, predictSample, waitForBackend } from './api'
import { beatValues, CLASSES } from './classes'
import UploadPanel from './components/UploadPanel'
import BeatInspector from './components/BeatInspector'
import PerformancePage from './components/PerformancePage'
import { BeatTable, Distribution, LabelledConfusion, RecordingCard, SummaryStats } from './components/Summary'

function HeartbeatLogo() {
  return (
    <svg viewBox="0 0 64 32" width="56" height="28" aria-hidden className="logo-trace">
      <path d="M0 18h14l4-8 6 16 6-24 5 16 3-6h26" />
    </svg>
  )
}

function exportCsv(data) {
  const header = ['beat', 'prediction', 'confidence', 'needs_review', ...CLASSES.map((c) => `p_${c.name}`), 'true_label', 'annotation']
  const lines = data.results.map((r) => [
    r.index + 1, CLASSES[r.prediction].name, r.confidence.toFixed(5), r.needs_review, ...r.probabilities,
    r.true_label != null ? CLASSES[r.true_label].name : '', r.symbol ?? '',
  ].join(','))
  const blob = new Blob([[header.join(','), ...lines].join('\n')], { type: 'text/csv' })
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob),
    download: `${data.filename.replace(/\.[^.]+$/, '').replace(/\s+/g, '_')}_predictions.csv`,
  })
  a.click()
  URL.revokeObjectURL(a.href)
}

function EmptyState() {
  return (
    <section className="card empty">
      <div className="pulse-heart" aria-hidden>
        <svg viewBox="0 0 24 24" width="56" height="56"><path d="M12 21s-7.5-4.6-9.6-9.2C.9 8.4 3 4.5 6.7 4.5c2.1 0 3.6 1.2 5.3 3.1 1.7-1.9 3.2-3.1 5.3-3.1 3.7 0 5.8 3.9 4.3 7.3C19.5 16.4 12 21 12 21Z" fill="currentColor" /></svg>
      </div>
      <h2>Ready when you are</h2>
      <p className="muted">Upload an ECG file or choose an example. The model classifies every heartbeat into one of five AAMI classes.</p>
      <div className="legend-grid">
        {CLASSES.map((c) => (
          <div key={c.id} className="legend-card" style={{ '--chip': c.color }}>
            <span className="code-badge small">{c.code}</span>
            <div>
              <strong>{c.name}</strong>
              <p>{c.description}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

const MODE_LABEL = { signal: 'Continuous recording', record: 'WFDB record' }

export default function App() {
  const params = new URLSearchParams(window.location.search)
  const [page, setPage] = useState(params.get('page') === 'performance' ? 'performance' : 'analyze')
  const [info, setInfo] = useState(null)
  const [online, setOnline] = useState(null)
  const [samples, setSamples] = useState([])
  const [records, setRecords] = useState([])
  const [data, setData] = useState(null)
  const [selected, setSelected] = useState(0)
  const [cellFilter, setCellFilter] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const resultsRef = useRef(null)

  const run = async (fn) => {
    setPage('analyze')
    setLoading(true)
    setError('')
    try {
      const res = await fn()
      const firstAbnormal = res.results.findIndex((r) => r.prediction !== 0)
      setData(res)
      setCellFilter(null)
      setSelected(res.num_beats > 1 && firstAbnormal >= 0 ? firstAbnormal : 0)
      setOnline(true)
      requestAnimationFrame(() => {
        if (window.innerWidth < 1024) resultsRef.current?.scrollIntoView({ behavior: 'smooth' })
      })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let cancelled = false
    waitForBackend(() => !cancelled && setOnline('waking'))
      .then((i) => {
        if (cancelled) return
        setInfo(i)
        setOnline(true)
        getSamples().then(setSamples).catch(() => {})
        getRecords().then(setRecords).catch(() => {})
        // Deep links, e.g. /?sample=sample_003_Ventricular_label2.csv or /?record=208
        if (params.get('sample')) run(() => predictSample(params.get('sample')))
        else if (params.get('record')) run(() => predictRecord(params.get('record')))
      })
      .catch(() => !cancelled && setOnline(false))
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Arrow keys step through beats
  useEffect(() => {
    if (!data || data.num_beats < 2 || page !== 'analyze') return
    const onKey = (e) => {
      if (e.target.closest('input, textarea, select')) return
      if (e.key === 'ArrowRight') setSelected((s) => Math.min(data.num_beats - 1, s + 1))
      else if (e.key === 'ArrowLeft') setSelected((s) => Math.max(0, s - 1))
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [data, page])

  const beat = data?.results[selected]
  const multi = data && data.num_beats > 1

  const selectCell = (trueId, predId) => {
    setCellFilter([trueId, predId])
    const first = data.results.find((r) => r.true_label === trueId && r.prediction === predId)
    if (first) setSelected(first.index)
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <HeartbeatLogo />
          <div>
            <strong>CardioScan</strong>
            <span>ECG Arrhythmia Detection</span>
          </div>
        </div>
        <nav className="nav segmented">
          <button className={page === 'analyze' ? 'active' : ''} onClick={() => setPage('analyze')}>Analyze</button>
          <button className={page === 'performance' ? 'active' : ''} onClick={() => setPage('performance')}>Model performance</button>
        </nav>
        <div className={`status ${online === false ? 'off' : online === true ? 'on' : ''}`}>
          <i />
          {{ null: 'Connecting…', waking: 'Waking up server…', true: 'Model online', false: 'Backend offline' }[String(online)]}
        </div>
      </header>

      {page === 'performance' ? (
        <main className="page-single"><PerformancePage /></main>
      ) : (
        <>
          <section className="hero">
            <h1>Detect arrhythmias from <span className="grad">ECG signals</span> in seconds</h1>
            <p>
              A deep CNN + BiLSTM + attention network trained on the MIT-BIH Arrhythmia database classifies each
              heartbeat as Normal, Supraventricular, Ventricular, Fusion or Unknown.
            </p>
            {info && (
              <div className="hero-tags">
                <span>{(info.parameters / 1e6).toFixed(2)}M parameters</span>
                <span>{info.input_length}-sample beats @ {info.sampling_rate_hz} Hz</span>
                <span>5 AAMI classes</span>
              </div>
            )}
          </section>

          <main className="layout">
            <aside>
              <UploadPanel samples={samples} records={records} loading={loading}
                onFiles={(files, fs) => run(() => predictFiles(files, fs))}
                onSample={(name) => run(() => predictSample(name))}
                onRecord={(name) => run(() => predictRecord(name))} />
            </aside>

            <div className="results" ref={resultsRef}>
              {error && (
                <div className="alert" role="alert">
                  <b>Analysis failed.</b> {error}
                  <button onClick={() => setError('')} aria-label="Dismiss">×</button>
                </div>
              )}

              {!data && !loading && <EmptyState />}
              {!data && loading && (
                <section className="card empty"><div className="spinner" /><p className="muted">Analyzing heartbeats…</p></section>
              )}

              {data && (
                <div className={`result-stack ${loading ? 'dim' : ''}`} key={data.filename + data.num_beats}>
                  <div className="result-bar">
                    <div>
                      <span className="eyebrow">Results for</span>
                      <h2 className="file-name">{data.filename}</h2>
                    </div>
                    <div className="result-actions">
                      <span className="badge">{MODE_LABEL[data.mode] ?? (multi ? 'Beat batch' : 'Single beat')}</span>
                      <button className="btn btn-ghost" onClick={() => exportCsv(data)}>Export CSV</button>
                    </div>
                  </div>

                  {multi && <SummaryStats data={data} />}
                  {data.signal && <RecordingCard data={data} selected={selected} onSelect={setSelected} />}

                  {beat && (
                    <BeatInspector beat={beat} values={beatValues(data, selected)} total={data.num_beats}
                      onPrev={() => setSelected((s) => Math.max(0, s - 1))}
                      onNext={() => setSelected((s) => Math.min(data.num_beats - 1, s + 1))} />
                  )}

                  {multi && <Distribution counts={data.summary.counts} total={data.num_beats} />}
                  {multi && data.summary.confusion_matrix && <LabelledConfusion data={data} onCellClick={selectCell} />}
                  {multi && (
                    <BeatTable results={data.results} selected={selected} onSelect={setSelected}
                      cellFilter={cellFilter} onClearCell={() => setCellFilter(null)} />
                  )}
                </div>
              )}
            </div>
          </main>
        </>
      )}

      <footer className="footer">
        For research and educational use only — not a medical device. Always consult a qualified clinician.
      </footer>
    </div>
  )
}
