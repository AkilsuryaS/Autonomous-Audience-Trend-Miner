import type { AudiencePortfolio, TrendSnapshot } from './types'

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL

if (!configuredBaseUrl) {
  throw new Error('VITE_API_BASE_URL must be configured in frontend/.env')
}

export const API_BASE_URL = configuredBaseUrl.replace(/\/$/, '')

export const websocketRunUrl = (() => {
  const url = new URL('/ws/run', `${API_BASE_URL}/`)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
})()

async function fetchJson<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`)
    if (response.status === 404) return null
    if (!response.ok) throw new Error(`Request failed with ${response.status}`)
    return (await response.json()) as T
  } catch {
    return null
  }
}

export async function getHealth(): Promise<boolean> {
  const response = await fetchJson<{ status: string }>('/api/health')
  return response?.status === 'ok'
}

export function getLatestPortfolio(): Promise<AudiencePortfolio | null> {
  return fetchJson<AudiencePortfolio>('/api/portfolio/latest')
}

export function getLatestTrends(): Promise<TrendSnapshot | null> {
  return fetchJson<TrendSnapshot>('/api/trends/latest')
}
