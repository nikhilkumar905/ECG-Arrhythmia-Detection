const BASE = (import.meta.env.VITE_API_URL ?? '').trim().replace(/\/+$/, '')

const UNREACHABLE = BASE
  ? `Cannot reach the backend at ${BASE}. It may still be starting up - wait a minute and try again.`
  : 'Cannot reach the backend. Is the FastAPI server running on port 8001?'

async function request(path, options) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, options)
  } catch {
    throw new Error(UNREACHABLE)
  }
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join(', ') : body.detail
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return body
}

export const getInfo = () => request('/api/info')

// Free hosting puts the backend to sleep when idle; keep retrying while it wakes up
export async function waitForBackend(onRetry, attempts = 18, delayMs = 5000) {
  for (let i = 1; ; i++) {
    try {
      return await getInfo()
    } catch (e) {
      if (i >= attempts) throw e
      onRetry?.(i)
      await new Promise((r) => setTimeout(r, delayMs))
    }
  }
}
export const getSamples = () => request('/api/samples')
export const getPerformance = () => request('/api/performance')
export const figureUrl = (file) => `${BASE}/api/figures/${encodeURIComponent(file)}`

export const predictSample = (name) =>
  request(`/api/samples/${encodeURIComponent(name)}/predict`, { method: 'POST' })

// The 187 values of a bundled MIT-BIH test beat, for previewing it before analysis
export async function getSampleValues(name) {
  let res
  try {
    res = await fetch(`${BASE}/api/samples/${encodeURIComponent(name)}/download`)
  } catch {
    throw new Error(UNREACHABLE)
  }
  if (!res.ok) throw new Error(`Could not load the sample (${res.status})`)
  return (await res.text()).trim().split(/[,\s]+/).map(Number)
}

export function predictFiles(files, samplingRate) {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  form.append('sampling_rate', String(samplingRate))
  return request('/api/predict', { method: 'POST', body: form })
}
