export interface QueryRequest {
  question: string;
  mode: 'vector' | 'graph' | 'hybrid';
}

export interface CitationMarker {
  marker: number;
  source_id: string;
}

export interface EvidenceItem {
  type: 'chunk' | 'graph_path';
  content: string;
  source_ids: string[];
  confidence: number;
}

export interface GraphPath {
  source: string;
  relationship: string;
  target: string;
  properties: Record<string, any>;
}

export interface GeneratedAnswer {
  answer_text: string;
  citations: CitationMarker[];
  confidence: 'high' | 'medium' | 'low' | 'insufficient_evidence';
  mode_used: 'vector' | 'graph' | 'hybrid';
  evidence_trace?: string[];
  evidence?: EvidenceItem[];
  graph_paths?: GraphPath[];
  status?: string;
}

export interface GraphStats {
  node_counts: Record<string, number>;
  relationship_counts: Record<string, number>;
  total_nodes: number;
  total_relationships: number;
  total_documents: number;
}

export interface Evidence {
  source_id: string;
  document_id: string;
  title: string;
  section?: string;
  excerpt: string;
  url?: string;
}

export interface HealthStatus {
  status: string;
  neo4j: boolean;
  vector_store: boolean;
  llm: boolean;
}
