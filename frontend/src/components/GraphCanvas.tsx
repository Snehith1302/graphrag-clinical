import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import { GraphPath } from '../types';
import { fetchNeighborhood } from '../api/client';
import { Info, GitFork, Plus } from 'lucide-react';

interface GraphCanvasProps {
  graphPaths: GraphPath[];
  onNodeClick: (entityName: string) => void;
  onEdgeClick: (sourceIds: string[], relType: string) => void;
}

// Colors representing distinct clinical entity types
const TYPE_COLORS: Record<string, string> = {
  Drug: '#059669',        // emerald-600
  Condition: '#dc2626',   // red-600
  Symptom: '#d97706',     // amber-600
  SideEffect: '#ea580c',  // orange-600
  Population: '#2563eb',  // blue-600
  ClinicalStudy: '#7c3aed',// violet-600
  Guideline: '#4f46e5'    // indigo-600
};

export default function GraphCanvas({ graphPaths, onNodeClick, onEdgeClick }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [loadingNeighborhood, setLoadingNeighborhood] = useState(false);

  // Helper to resolve entity type from node label
  const inferType = (name: string): string => {
    // Basic lookup or fallback
    const lower = name.toLowerCase();
    if (lower.includes("metformin") || lower.includes("insulin") || lower.includes("drug")) return "Drug";
    if (lower.includes("diabetes") || lower.includes("renal") || lower.includes("impairment")) return "Condition";
    if (lower.includes("diarrhea") || lower.includes("asthenia") || lower.includes("nausea")) return "SideEffect";
    return "Guideline";
  };

  useEffect(() => {
    if (!containerRef.current || !graphPaths || graphPaths.length === 0) {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
      return;
    }

    const elements: cytoscape.ElementDefinition[] = [];
    const addedNodes = new Set<string>();
    const addedEdges = new Set<string>();

    // 1. Process paths into nodes and edges
    graphPaths.forEach(path => {
      const srcName = path.source;
      const tgtName = path.target;
      const srcId = srcName.toLowerCase().replace(/\s+/g, '_');
      const tgtId = tgtName.toLowerCase().replace(/\s+/g, '_');

      // Add Source Node
      if (!addedNodes.has(srcId)) {
        elements.push({
          data: {
            id: srcId,
            name: srcName,
            type: inferType(srcName)
          }
        });
        addedNodes.add(srcId);
      }

      // Add Target Node
      if (!addedNodes.has(tgtId)) {
        elements.push({
          data: {
            id: tgtId,
            name: tgtName,
            type: inferType(tgtName)
          }
        });
        addedNodes.add(tgtId);
      }

      // Add Edge
      const edgeId = `${srcId}_${tgtId}_${path.relationship}`;
      if (!addedEdges.has(edgeId)) {
        elements.push({
          data: {
            id: edgeId,
            source: srcId,
            target: tgtId,
            type: path.relationship,
            confidence: path.properties?.confidence || 0.9,
            source_ids: path.properties?.source_ids || []
          }
        });
        addedEdges.add(edgeId);
      }
    });

    // 2. Initialize Cytoscape
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(name)',
            'background-color': (ele) => TYPE_COLORS[ele.data('type')] || '#475569',
            'color': '#f8fafc',
            'font-size': 11,
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 65,
            'height': 65,
            'shape': 'roundrectangle',
            'text-wrap': 'wrap',
            'text-max-width': '60px',
            'border-width': 2,
            'border-color': '#1e293b'
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#10b981' // Highlight selected node with emerald
          }
        },
        {
          selector: 'edge',
          style: {
            'label': 'data(type)',
            'font-size': 9,
            'color': '#94a3b8',
            'text-background-opacity': 0.7,
            'text-background-color': '#0f172a',
            'text-background-padding': '3px',
            'text-background-shape': 'roundrectangle',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#64748b',
            'line-color': '#475569',
            'width': 2.5,
            'curve-style': 'bezier',
            'control-point-step-size': 35
          }
        }
      ],
      layout: {
        name: 'cose',
        animate: true,
        fit: true,
        padding: 40,
        nodeOverlap: 20
      }
    });

    // 3. Setup click listeners
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      setSelectedNode(node.data('name'));
      onNodeClick(node.data('name'));
    });

    cy.on('tap', 'edge', (evt) => {
      const edge = evt.target;
      onEdgeClick(edge.data('source_ids'), edge.data('type'));
    });

    cyRef.current = cy;

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [graphPaths]);

  // Load adjacent connections (1-hop neighborhood)
  const handleShowNeighborhood = async () => {
    if (!selectedNode || !cyRef.current) return;
    
    setLoadingNeighborhood(true);
    try {
      const cy = cyRef.current;
      const data = await fetchNeighborhood(selectedNode, 1);
      
      const addedNodes = new Set(cy.nodes().map(n => n.id()));
      const addedEdges = new Set(cy.edges().map(e => e.id()));

      data.nodes.forEach(n => {
        if (!addedNodes.has(n.id)) {
          cy.add({
            group: 'nodes',
            data: {
              id: n.id,
              name: n.name,
              type: n.label
            }
          });
        }
      });

      data.edges.forEach(e => {
        if (!addedEdges.has(e.id)) {
          cy.add({
            group: 'edges',
            data: {
              id: e.id,
              source: e.source,
              target: e.target,
              type: e.type,
              confidence: e.properties?.confidence || 0.9,
              source_ids: e.properties?.source_ids || []
            }
          });
        }
      });

      // Rerun layout to reposition new nodes
      cy.layout({ name: 'cose', animate: true, fit: true }).run();
    } catch (err) {
      console.error("Neighborhood fetch failed:", err);
      alert(`Could not load neighborhood for ${selectedNode}`);
    } finally {
      setLoadingNeighborhood(false);
    }
  };

  if (!graphPaths || graphPaths.length === 0) {
    return (
      <div className="flex-1 bg-slate-950 border border-slate-800 rounded-lg flex flex-col items-center justify-center text-slate-500 text-xs text-center p-4">
        <GitFork className="w-8 h-8 text-slate-800 mb-2" />
        <span>No clinical graph paths returned for the current response.</span>
        <span className="text-[10px] text-slate-600 mt-1">(Requires GraphRAG or Hybrid retrieval match)</span>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full relative bg-slate-950 rounded-lg border border-slate-800">
      {/* Interactive Canvas */}
      <div ref={containerRef} className="flex-1 w-full min-h-[300px] h-full" />

      {/* Control bar */}
      <div className="absolute bottom-3 left-3 right-3 bg-slate-900/90 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between text-xs backdrop-blur-sm z-10">
        <div>
          {selectedNode ? (
            <span className="text-slate-200">
              Selected: <strong className="text-emerald-400">{selectedNode}</strong>
            </span>
          ) : (
            <span className="text-slate-500 flex items-center gap-1">
              <Info className="w-3.5 h-3.5" /> Click node or edge to inspect
            </span>
          )}
        </div>
        
        {selectedNode && (
          <button
            onClick={handleShowNeighborhood}
            disabled={loadingNeighborhood}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-emerald-800 text-white px-3 py-1 rounded font-semibold flex items-center gap-1 text-[11px] transition-colors"
          >
            <Plus className="w-3.5 h-3.5" /> 
            {loadingNeighborhood ? 'Loading...' : 'Show Neighbors'}
          </button>
        )}
      </div>
    </div>
  );
}
