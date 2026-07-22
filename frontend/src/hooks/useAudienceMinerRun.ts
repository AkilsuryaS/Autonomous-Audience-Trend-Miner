import { useCallback, useEffect, useRef, useState } from 'react'
import { websocketRunUrl } from '../api'
import type {
  AudiencePortfolio,
  MinerStatus,
  PipelineLogEntry,
  WebSocketMessage,
} from '../types'

interface UseAudienceMinerRunOptions {
  onResult?: (portfolio: AudiencePortfolio) => void
}

const timestamp = () =>
  new Intl.DateTimeFormat('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).format(new Date())

export function useAudienceMinerRun(options: UseAudienceMinerRunOptions = {}) {
  const [status, setStatus] = useState<MinerStatus>('idle')
  const [logs, setLogs] = useState<PipelineLogEntry[]>([])
  const [portfolio, setPortfolio] = useState<AudiencePortfolio | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const entryIdRef = useRef(0)
  const onResultRef = useRef(options.onResult)

  useEffect(() => {
    onResultRef.current = options.onResult
  }, [options.onResult])

  useEffect(
    () => () => {
      socketRef.current?.close()
    },
    [],
  )

  const appendLog = useCallback(
    (stage: string, detail: string, tone: PipelineLogEntry['tone'] = 'default') => {
      entryIdRef.current += 1
      setLogs((current) => [
        ...current,
        { id: entryIdRef.current, timestamp: timestamp(), stage, detail, tone },
      ])
    },
    [],
  )

  const run = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return

    setStatus('running')
    setPortfolio(null)
    setLogs([])
    entryIdRef.current = 0

    const socket = new WebSocket(websocketRunUrl)
    socketRef.current = socket
    let settled = false

    socket.onopen = () => {
      appendLog('socket', 'Signal wire opened — pipeline started')
    }

    socket.onmessage = (event) => {
      let message: WebSocketMessage
      try {
        message = JSON.parse(String(event.data)) as WebSocketMessage
      } catch {
        settled = true
        setStatus('error')
        appendLog('error', 'Received an unreadable message from the backend.', 'error')
        socket.close()
        return
      }

      if (message.type === 'progress') {
        appendLog(message.stage, message.detail)
        return
      }

      if (message.type === 'result') {
        settled = true
        setPortfolio(message.portfolio)
        setStatus('success')
        appendLog('result', 'Audience portfolio ready', 'success')
        onResultRef.current?.(message.portfolio)
        return
      }

      settled = true
      setStatus('error')
      appendLog(
        'error',
        `${message.message} You can retry when the signal is available.`,
        'error',
      )
    }

    socket.onerror = () => {
      if (settled) return
      settled = true
      setStatus('error')
      appendLog(
        'socket',
        'The backend connection failed. Check the API service and retry.',
        'error',
      )
    }

    socket.onclose = () => {
      socketRef.current = null
      if (settled) return
      settled = true
      setStatus('error')
      appendLog(
        'socket',
        'The signal wire closed before a result arrived. You can retry.',
        'error',
      )
    }
  }, [appendLog])

  return { status, logs, portfolio, run }
}
