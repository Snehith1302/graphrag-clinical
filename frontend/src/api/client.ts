/**
 * Typed client API definitions for interacting with the FastAPI backend.
 */
import { 
  QueryRequest, 
  GeneratedAnswer, 
  GraphStats, 
  Evidence, 
  HealthStatus 
} from '../types';

export const API_BASE = ((import.meta as any).env.VITE_API_BASE_URL as string) || '/api';

export async function submitQuery(request: QueryRequest): Promise<GeneratedAnswer> {
  const response = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  if (!response.ok) {
    const errBody = await response.json().catch(() => ({}));
    throw new Error(errBody.detail || `API query error: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error('Health check request failed');
  }
  return response.json();
}

export async function fetchStats(): Promise<GraphStats> {
  const response = await fetch(`${API_BASE}/graph/stats`);
  if (!response.ok) {
    throw new Error('Graph statistics request failed');
  }
  return response.json();
}

export async function fetchNeighborhood(entity: string, hopDepth: number = 2): Promise<{nodes: any[], edges: any[]}> {
  const response = await fetch(`${API_BASE}/graph/neighborhood?entity=${encodeURIComponent(entity)}&hop_depth=${hopDepth}`);
  if (!response.ok) {
    throw new Error(`Graph neighborhood lookup failed for ${entity}`);
  }
  return response.json();
}

export async function fetchEvidence(sourceId: string): Promise<Evidence> {
  const response = await fetch(`${API_BASE}/evidence/${encodeURIComponent(sourceId)}`);
  if (!response.ok) {
    throw new Error(`Evidence details lookup failed for ${sourceId}`);
  }
  return response.json();
}
