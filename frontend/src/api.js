const BASE = import.meta.env.VITE_API_URL ?? ''

async function request(path, options) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, options)
  } catch {
    throw new Error('Cannot reach the backend. Is the FastAPI server running on port 8001?')
  }
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join(', ') : body.detail
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return body
}

export const getInfo = () => request('/api/info')
export const getSamples = () => request('/api/samples')
export const getRecords = () => request('/api/records')
export const getPerformance = () => request('/api/performance')
export const figureUrl = (file) => `${BASE}/api/figures/${encodeURIComponent(file)}`

export const predictSample = (name) =>
  request(`/api/samples/${encodeURIComponent(name)}/predict`, { method: 'POST' })
export const predictRecord = (name) =>
  request(`/api/records/${encodeURIComponent(name)}/predict`, { method: 'POST' })

export function predictFiles(files, samplingRate) {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  form.append('sampling_rate', String(samplingRate))
  return request('/api/predict', { method: 'POST', body: form })
}
