import type { MinerStatus, PipelineLogEntry } from '../types'

interface PipelineConsoleProps {
  logs: PipelineLogEntry[]
  status: MinerStatus
}

export function PipelineConsole({ logs, status }: PipelineConsoleProps) {
  return (
    <div className="pipeline-console" aria-live="polite">
      <div className="pipeline-console__bar">
        <div className="pipeline-console__lights" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <p>AGENT WIRE / LIVE</p>
        <span className={`console-status console-status--${status}`}>
          {status.toUpperCase()}
        </span>
      </div>
      <div className="pipeline-console__body">
        {logs.length ? (
          <ol>
            {logs.map((entry) => (
              <li key={entry.id} className={`log-line log-line--${entry.tone}`}>
                <time>{entry.timestamp}</time>
                <span className="log-line__stage">[{entry.stage.toUpperCase()}]</span>
                <span>{entry.detail}</span>
              </li>
            ))}
          </ol>
        ) : (
          <p className="pipeline-console__empty">
            <span aria-hidden="true">›</span> Waiting for a run command
            <span className="console-cursor" aria-hidden="true" />
          </p>
        )}
      </div>
    </div>
  )
}
