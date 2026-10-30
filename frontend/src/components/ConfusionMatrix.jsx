import { useState } from 'react'
import { CLASSES } from '../classes'

// Rows = true class, columns = predicted class. Cells are shaded by row-normalized share.
export default function ConfusionMatrix({ matrix, onCellClick }) {
  const [normalized, setNormalized] = useState(true)
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0))
  const rows = CLASSES.filter((c) => rowTotals[c.id] > 0 || matrix.some((r) => r[c.id] > 0))

  return (
    <div className="cm">
      <div className="cm-toolbar">
        <span className="muted small">Rows: true class · Columns: predicted class</span>
        <div className="segmented">
          <button className={normalized ? 'active' : ''} onClick={() => setNormalized(true)}>% of row</button>
          <button className={!normalized ? 'active' : ''} onClick={() => setNormalized(false)}>Counts</button>
        </div>
      </div>
      <div className="cm-scroll">
        <table className="cm-table">
          <thead>
            <tr>
              <th className="cm-corner">True ↓ / Pred →</th>
              {rows.map((c) => <th key={c.id} style={{ '--chip': c.color }}><i className="dot" />{c.short}</th>)}
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <th style={{ '--chip': r.color }}><i className="dot" />{r.short}</th>
                {rows.map((c) => {
                  const n = matrix[r.id][c.id]
                  const share = rowTotals[r.id] ? n / rowTotals[r.id] : 0
                  const diag = r.id === c.id
                  return (
                    <td key={c.id}
                      className={`cm-cell ${diag ? 'diag' : ''} ${n ? '' : 'zero'} ${onCellClick && n ? 'clickable' : ''}`}
                      style={{ '--share': Math.sqrt(share) }}
                      title={`True ${r.name}, predicted ${c.name}: ${n.toLocaleString()} (${(share * 100).toFixed(1)}%)`}
                      onClick={() => onCellClick && n && onCellClick(r.id, c.id)}>
                      {normalized ? (n ? `${(share * 100).toFixed(share < 0.01 ? 2 : 1)}%` : '·') : n.toLocaleString()}
                    </td>
                  )
                })}
                <td className="mono muted">{rowTotals[r.id].toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
