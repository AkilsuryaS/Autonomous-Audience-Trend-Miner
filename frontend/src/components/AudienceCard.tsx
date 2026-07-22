import type { AudienceSegment } from '../types'

interface AudienceCardProps {
  segment: AudienceSegment
  index: number
}

export function AudienceCard({ segment, index }: AudienceCardProps) {
  const buyingPower = segment.potential_buying_power
  const boundedIndex = Math.min(Math.max(segment.estimated_size_index, 0), 100)

  return (
    <article className="audience-card">
      <div className="audience-card__topline">
        <span className="audience-card__number">
          SEGMENT {String(index).padStart(2, '0')}
        </span>
        <span
          className={`buying-power buying-power--${buyingPower.level.toLowerCase()}`}
        >
          {buyingPower.level} buying power
        </span>
      </div>

      <h3>{segment.audience_name}</h3>
      <p className="audience-card__description">{segment.audience_description}</p>

      <div className="signal-index">
        <div className="signal-index__label">
          <span>Estimated size index</span>
          <strong>{segment.estimated_size_index.toFixed(1)}%</strong>
        </div>
        <div
          className="signal-index__track"
          role="progressbar"
          aria-label={`${segment.audience_name} estimated size index`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={boundedIndex}
        >
          <span style={{ width: `${boundedIndex}%` }} />
        </div>
      </div>

      <div className="commercial-fit">
        <p className="commercial-fit__label">Commercial read</p>
        <p>{buyingPower.rationale}</p>
        <ul className="brand-tags" aria-label="Relevant brand categories">
          {buyingPower.brand_categories.map((category) => (
            <li key={category}>{category}</li>
          ))}
        </ul>
      </div>

      <details className="source-disclosure">
        <summary>
          <span>Source signals</span>
          <span>{segment.source_articles.length}</span>
        </summary>
        <ul>
          {segment.source_articles.map((article) => (
            <li key={article}>{article}</li>
          ))}
        </ul>
      </details>
    </article>
  )
}
