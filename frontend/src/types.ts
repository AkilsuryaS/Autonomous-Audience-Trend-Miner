export type BuyingPowerLevel = 'High' | 'Medium' | 'Low'

export interface PlacementDecision {
  article_title: string
  cluster_name: string
  primary_relevance: string
  fit_rationale: string
  ambiguity_resolution: string
}

export interface BuyingPowerAssessment {
  level: BuyingPowerLevel
  rationale: string
  brand_categories: string[]
}

export interface AudienceSegment {
  source_cluster_name: string
  audience_name: string
  audience_description: string
  estimated_size_index: number
  potential_buying_power: BuyingPowerAssessment
  source_articles: string[]
  placement_decisions: PlacementDecision[]
}

export interface AudiencePortfolio {
  segments: AudienceSegment[]
}

export interface ProgressMessage {
  type: 'progress'
  stage: 'fetch' | 'cluster' | 'portfolio'
  detail: string
}

export interface ResultMessage {
  type: 'result'
  portfolio: AudiencePortfolio
}

export interface ErrorMessage {
  type: 'error'
  message: string
}

export type WebSocketMessage = ProgressMessage | ResultMessage | ErrorMessage

export interface TrendArticle {
  title: string
  views: number
}

export interface TrendSnapshot {
  articles: TrendArticle[]
}

export interface PipelineLogEntry {
  id: number
  timestamp: string
  stage: string
  detail: string
  tone?: 'default' | 'success' | 'error'
}

export type MinerStatus = 'idle' | 'running' | 'success' | 'error'
