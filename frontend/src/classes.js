// Display metadata for the 5 AAMI heartbeat classes (ids match the model output)
export const CLASSES = [
  { id: 0, code: 'N', name: 'Normal', short: 'Normal', color: 'var(--c-normal)',
    description: 'Normal sinus beat, including bundle branch block and nodal escape beats.' },
  { id: 1, code: 'S', name: 'Supraventricular', short: 'Supra', color: 'var(--c-supra)',
    description: 'Premature ectopic beat originating above the ventricles (atria or AV node).' },
  { id: 2, code: 'V', name: 'Ventricular', short: 'Ventr', color: 'var(--c-ventr)',
    description: 'Premature ventricular contraction (PVC) originating in the ventricles.' },
  { id: 3, code: 'F', name: 'Fusion', short: 'Fusion', color: 'var(--c-fusion)',
    description: 'Fusion of a ventricular and a normal beat occurring simultaneously.' },
  { id: 4, code: 'Q', name: 'Unknown', short: 'Unknown', color: 'var(--c-unknown)',
    description: 'Unclassifiable beat, including paced beats.' },
]

export const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`

export const reviewReason = (r) =>
  r.confidence < 0.7
    ? `Low confidence (${pct(r.confidence)})`
    : `Top two classes are close (margin ${pct(r.margin)})`
