import { useCallback, useEffect, useMemo, useState } from 'react'
import './App.css'
import { getHealth, getLatestPortfolio, getLatestTrends } from './api'
import { AudienceCard } from './components/AudienceCard'
import { PipelineConsole } from './components/PipelineConsole'
import { SignalTicker } from './components/SignalTicker'
import { useAudienceMinerRun } from './hooks/useAudienceMinerRun'
import type { AudiencePortfolio, TrendArticle } from './types'

function App() {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)
  const [restoredPortfolio, setRestoredPortfolio] =
    useState<AudiencePortfolio | null>(null)
  const [trends, setTrends] = useState<TrendArticle[]>([])

  const refreshTrends = useCallback(async () => {
    const snapshot = await getLatestTrends()
    if (snapshot) setTrends(snapshot.articles)
  }, [])

  const miner = useAudienceMinerRun({
    onResult: () => {
      void refreshTrends()
    },
  })

  useEffect(() => {
    let active = true

    const restore = async () => {
      const healthy = await getHealth()
      if (!active) return
      setBackendOnline(healthy)
      if (!healthy) return

      const [portfolio, trendSnapshot] = await Promise.all([
        getLatestPortfolio(),
        getLatestTrends(),
      ])
      if (!active) return
      if (portfolio) setRestoredPortfolio(portfolio)
      if (trendSnapshot) setTrends(trendSnapshot.articles)
    }

    void restore()
    return () => {
      active = false
    }
  }, [])

  const portfolio = miner.portfolio ?? restoredPortfolio
  const sourceArticleNames = useMemo(
    () =>
      new Set(
        portfolio?.segments.flatMap((segment) => segment.source_articles) ?? [],
      ),
    [portfolio],
  )
  const portfolioTrends = useMemo(() => {
    if (!sourceArticleNames.size) return trends.slice(0, 16)
    return trends.filter((article) => sourceArticleNames.has(article.title))
  }, [sourceArticleNames, trends])

  const runMiner = () => {
    setRestoredPortfolio(null)
    miner.run()
  }

  const runDisabled = backendOnline !== true || miner.status === 'running'

  return (
    <main className="app-shell">
      <header className="masthead">
        <div className="masthead__rule" aria-hidden="true">
          <span>INMARKET PROTOTYPE</span>
          <span>EN.WIKIPEDIA / 7-DAY SIGNAL</span>
        </div>
        <div className="masthead__content">
          <div>
            <p className="eyebrow">Autonomous audience intelligence</p>
            <h1>Audience Trend Miner</h1>
            <p className="masthead__description">
              Raw public attention, audited and shaped into commercially useful
              audience signals.
            </p>
          </div>
          <div
            className={`health-indicator ${backendOnline ? 'is-online' : ''}`}
            role="status"
            aria-live="polite"
          >
            <span className="health-indicator__dot" aria-hidden="true" />
            <span>
              {backendOnline === null
                ? 'Checking signal'
                : backendOnline
                  ? 'Backend online'
                  : 'Backend unavailable'}
            </span>
          </div>
        </div>
      </header>

      <SignalTicker articles={portfolioTrends} />

      <section className="control-panel" aria-labelledby="control-title">
        <div className="control-panel__intro">
          <div>
            <p className="section-index">01 / MINE</p>
            <h2 id="control-title">From signal to segment</h2>
            <p>
              Launch one bounded agent run. Live generation, critique, refinement,
              and packaging events appear in the wire log.
            </p>
          </div>
          <button
            className="run-button"
            type="button"
            onClick={runMiner}
            disabled={runDisabled}
          >
            <span>{miner.status === 'running' ? 'Mining signals' : 'Run the miner'}</span>
            <span className="run-button__mark" aria-hidden="true">
              {miner.status === 'running' ? '•••' : '↗'}
            </span>
          </button>
        </div>

        {backendOnline === false && (
          <p className="backend-warning" role="alert">
            Start the FastAPI service on port 8000 to enable the miner.
          </p>
        )}

        <PipelineConsole logs={miner.logs} status={miner.status} />
      </section>

      <section className="results" aria-labelledby="results-title">
        <div className="section-heading">
          <div>
            <p className="section-index">02 / PORTFOLIO</p>
            <h2 id="results-title">Emerging audiences</h2>
          </div>
          {portfolio && (
            <p className="results__count">
              {String(portfolio.segments.length).padStart(2, '0')} SIGNALS
            </p>
          )}
        </div>

        {portfolio ? (
          <div className="audience-grid">
            {portfolio.segments.map((segment, index) => (
              <AudienceCard
                key={segment.source_cluster_name}
                segment={segment}
                index={index + 1}
              />
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <span className="empty-state__pulse" aria-hidden="true" />
            <p>No audiences mined yet — run the miner to open the signal wire.</p>
          </div>
        )}
      </section>

      <footer className="footer">
        <p>GLOBAL ENGLISH-LANGUAGE READERSHIP / NOT COUNTRY-SCOPED</p>
        <p>WIKIMEDIA PAGEVIEWS → MCP → LANGCHAIN</p>
      </footer>
    </main>
  )
}

export default App
