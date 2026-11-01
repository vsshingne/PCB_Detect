import { createPredictionForm } from './prediction.js'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, signal: options.signal })
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The request timed out. Try again with a smaller image.')
    throw new Error('Cannot reach the backend. Check that the API is running and try again.')
  }
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = typeof payload?.detail === 'string'
      ? payload.detail
      : Array.isArray(payload?.detail)
        ? payload.detail.map((item) => item.msg).filter(Boolean).join('; ')
        : 'The request could not be completed.'
    throw new Error(detail)
  }
  if (!payload || typeof payload !== 'object') throw new Error('The backend returned an invalid prediction response.')
  return payload
}

export const api = {
  health: (signal) => request('/api/health', { signal }),
  classes: (signal) => request('/api/classes', { signal }),
  predict: (file, threshold, signal) => {
    const form = createPredictionForm(file, threshold)
    return request('/api/predict', { method: 'POST', body: form, signal })
  },
}

export const apiBaseUrl = API_BASE
