import type { TrendArticle } from '../types'

interface SignalTickerProps {
  articles: TrendArticle[]
}

const placeholders: TrendArticle[] = [
  { title: 'AWAITING WIKIMEDIA SIGNAL', views: 0 },
  { title: 'SEVEN PROCESSED DAYS', views: 0 },
  { title: 'GLOBAL ENGLISH READERSHIP', views: 0 },
  { title: 'ONE MCP DATA CALL', views: 0 },
]

const viewFormatter = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
})

export function SignalTicker({ articles }: SignalTickerProps) {
  const items = articles.length ? articles : placeholders
  const repeatedItems = [...items, ...items]

  return (
    <section className="signal-ticker" aria-label="Latest Wikipedia traffic signals">
      <div className="signal-ticker__label">
        <span className="signal-ticker__live" aria-hidden="true" />
        RAW SIGNAL
      </div>
      <div className="signal-ticker__viewport">
        <div className="signal-ticker__track" aria-hidden="true">
          {repeatedItems.map((article, index) => (
            <span className="ticker-item" key={`${article.title}-${index}`}>
              <span>{article.title}</span>
              <strong>
                {article.views ? viewFormatter.format(article.views) : '—'}
              </strong>
            </span>
          ))}
        </div>
        <span className="sr-only">
          {items
            .map((article) =>
              article.views
                ? `${article.title}, ${article.views.toLocaleString()} views`
                : article.title,
            )
            .join('; ')}
        </span>
      </div>
    </section>
  )
}
