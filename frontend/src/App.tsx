import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, 
  Activity, 
  GitFork, 
  BookOpen, 
  Layers, 
  Search, 
  Database, 
  AlertCircle, 
  Calendar, 
  ChevronDown, 
  ChevronUp, 
  FileText, 
  CheckCircle, 
  ExternalLink,
  Info
} from 'lucide-react';
import { 
  submitQuery, 
  fetchStats, 
  fetchEvidence, 
  fetchHealth 
} from './api/client';
import { 
  GeneratedAnswer, 
  GraphStats, 
  Evidence, 
  EvidenceItem 
} from './types';
import GraphCanvas from './components/GraphCanvas';

function App() {
  // Query state
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'vector' | 'graph' | 'hybrid'>('hybrid');
  const [answer, setAnswer] = useState<GeneratedAnswer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Stats & Health state
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [health, setHealth] = useState<{ neo4j: boolean; vector_store: boolean; llm: boolean } | null>(null);

  // UI state
  const [activeTab, setActiveTab] = useState<'query' | 'evidence' | 'graph'>('query');
  const [expandedEvidence, setExpandedEvidence] = useState<Record<number, boolean>>({});
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);
  const [loadingEvidence, setLoadingEvidence] = useState(false);

  // Fetch initial graph statistics and health details
  useEffect(() => {
    async function loadStatsAndHealth() {
      try {
        const statsData = await fetchStats();
        setStats(statsData);
      } catch (err) {
        console.error("Failed to load graph stats:", err);
      }
      try {
        const healthData = await fetchHealth();
        setHealth(healthData);
      } catch (err) {
        console.error("Failed to load health status:", err);
      }
    }
    loadStatsAndHealth();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setAnswer(null);

    try {
      const result = await submitQuery({ question: query, mode });
      setAnswer(result);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred during search.');
    } finally {
      setLoading(false);
    }
  };

  const handleSourceClick = async (sourceId: string) => {
    setLoadingEvidence(true);
    try {
      const details = await fetchEvidence(sourceId);
      setSelectedEvidence(details);
    } catch (err) {
      console.error("Failed to fetch evidence metadata:", err);
      alert(`Could not retrieve document metadata for ${sourceId}`);
    } finally {
      setLoadingEvidence(false);
    }
  };

  const toggleEvidenceExpand = (idx: number) => {
    setExpandedEvidence(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 font-sans text-slate-100">
      {/* Persistent Safety Disclaimer */}
      <div className="bg-amber-950/80 border-b border-amber-800/80 text-amber-200 px-4 py-3 text-xs md:text-sm flex items-center gap-3">
        <ShieldAlert className="w-5 h-5 flex-shrink-0 text-amber-400" />
        <span className="leading-normal">
          <strong>⚠️ Safety Disclaimer:</strong> This is a clinical literature research prototype only. 
          It is not a medical device, does not provide diagnostic or treatment recommendations, and must not be used 
          for clinical decision support. Consult a licensed healthcare provider for individual medical guidance.
        </span>
      </div>

      {/* Header */}
      <header className="bg-slate-900 border-b border-slate-800 px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Activity className="w-8 h-8 text-emerald-500" />
          <div>
            <h1 className="text-xl font-bold leading-none tracking-wide text-slate-50">GraphRAG-Clinical</h1>
            <span className="text-xs text-slate-400">Clinical Literature Grounded Intelligence Portal</span>
          </div>
        </div>
        
        {/* Environment integration indicators */}
        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400">
          <span className="flex items-center gap-1.5">
            <Layers className={`w-4 h-4 ${health?.vector_store ? 'text-blue-500' : 'text-slate-600'}`} /> 
            Vector Store: {health?.vector_store ? 'Online' : 'Offline'}
          </span>
          <span className="flex items-center gap-1.5">
            <GitFork className={`w-4 h-4 ${health?.neo4j ? 'text-emerald-500' : 'text-slate-600'}`} /> 
            Graph DB (Neo4j): {health?.neo4j ? 'Online' : 'Offline'}
          </span>
          <span className="flex items-center gap-1.5">
            <CheckCircle className={`w-4 h-4 ${health?.llm ? 'text-purple-500' : 'text-slate-600'}`} /> 
            LLM: {health?.llm ? 'Configured' : 'Missing'}
          </span>
        </div>
      </header>

      {/* Stats Widgets Bar */}
      {stats && (
        <section className="grid grid-cols-2 md:grid-cols-5 bg-slate-900/50 border-b border-slate-800 p-4 gap-4 text-center">
          <div className="border-r border-slate-800/60 last:border-none">
            <span className="block text-xs text-slate-400">Total Nodes</span>
            <span className="text-lg font-bold text-slate-100">{stats.total_nodes}</span>
          </div>
          <div className="border-r border-slate-800/60 last:border-none">
            <span className="block text-xs text-slate-400">Total Edges</span>
            <span className="text-lg font-bold text-slate-100">{stats.total_relationships}</span>
          </div>
          <div className="border-r border-slate-800/60 last:border-none">
            <span className="block text-xs text-slate-400">Drug Entities</span>
            <span className="text-lg font-bold text-slate-100">{stats.node_counts["Drug"] || 0}</span>
          </div>
          <div className="border-r border-slate-800/60 last:border-none">
            <span className="block text-xs text-slate-400">Clinical Conditions</span>
            <span className="text-lg font-bold text-slate-100">{stats.node_counts["Condition"] || 0}</span>
          </div>
          <div className="last:border-none">
            <span className="block text-xs text-slate-400">Indexed Sources</span>
            <span className="text-lg font-bold text-slate-100">{stats.total_documents}</span>
          </div>
        </section>
      )}

      {/* Responsive Tab navigation for Mobile/Tablet */}
      <div className="flex md:hidden bg-slate-900 border-b border-slate-800 p-1">
        {(['query', 'evidence', 'graph'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 text-center py-2 text-xs font-semibold rounded-md capitalize ${
              activeTab === tab ? 'bg-slate-800 text-white border-b-2 border-emerald-500' : 'text-slate-400'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Main Workspace Layout */}
      <main className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-6 p-6 overflow-hidden">
        {/* Left Side: Query Interface & Answers (Columns 1 to 6) */}
        <div className={`col-span-1 md:col-span-6 flex flex-col gap-6 overflow-y-auto pr-1 ${
          activeTab !== 'query' ? 'hidden md:flex' : ''
        }`}>
          {/* Query Form Box */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
            <h2 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
              <Search className="w-5 h-5 text-emerald-400" /> Literature Inquiry
            </h2>
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a medical/clinical question (e.g., 'What are the contraindications for Metformin?')"
                className="w-full h-24 bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 resize-none transition-colors"
                maxLength={1000}
                required
              />
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex bg-slate-950 border border-slate-800 rounded-lg p-1">
                  {(['vector', 'graph', 'hybrid'] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setMode(m)}
                      className={`px-3 py-1.5 text-xs font-semibold rounded-md capitalize transition-all ${
                        mode === m
                          ? 'bg-emerald-600 text-white shadow-md'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
                <button
                  type="submit"
                  disabled={loading}
                  className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-emerald-800 text-white px-6 py-2 rounded-lg font-bold text-sm transition-colors shadow-lg flex items-center gap-2"
                >
                  {loading ? 'Processing...' : 'Run Search'}
                </button>
              </div>
            </form>
          </div>

          {/* Error Message Panel */}
          {error && (
            <div className="bg-red-950/40 border border-red-800/80 rounded-xl p-4 flex gap-3 text-sm text-red-200 shadow-xl">
              <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-400" />
              <div>
                <span className="font-semibold block">Inquiry Execution Failed</span>
                <span className="block mt-0.5">{error}</span>
              </div>
            </div>
          )}

          {/* Answer Panel */}
          {answer && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex-1 flex flex-col">
              <div className="flex items-center justify-between mb-3 border-b border-slate-800/50 pb-2">
                <h3 className="text-md font-semibold text-slate-300">Grounded Synthesis</h3>
                <span className={`px-2.5 py-0.5 text-xs font-semibold rounded-full uppercase ${
                  answer.confidence === 'high' ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800' :
                  answer.confidence === 'medium' ? 'bg-blue-950/80 text-blue-300 border border-blue-800' :
                  answer.confidence === 'low' ? 'bg-amber-950/80 text-amber-300 border border-amber-800' :
                  'bg-rose-950/80 text-rose-300 border border-rose-800'
                }`}>
                  Evidence Status: {answer.confidence.replace("_", " ")}
                </span>
              </div>
              <div className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap flex-1 bg-slate-950/50 p-4 border border-slate-800/60 rounded-lg">
                {answer.answer_text}
              </div>
            </div>
          )}
        </div>

        {/* Center/Right: Evidence Cards List (Columns 7 to 9) */}
        <div className={`col-span-1 md:col-span-3 flex flex-col gap-6 overflow-y-auto pr-1 ${
          activeTab !== 'evidence' ? 'hidden md:flex' : ''
        }`}>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex-1 flex flex-col h-full">
            <h2 className="text-md font-semibold text-slate-200 mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-emerald-400" /> Evidence Library
            </h2>

            {!answer?.evidence || answer.evidence.length === 0 ? (
              <div className="flex-1 flex items-center justify-center text-slate-500 text-sm text-center p-6 border border-dashed border-slate-800 rounded-lg">
                Submit an inquiry query to fetch related literature snippets
              </div>
            ) : (
              <div className="flex flex-col gap-4 overflow-y-auto flex-1 max-h-[70vh]">
                {answer.evidence.map((item, idx) => {
                  const isExpanded = !!expandedEvidence[idx];
                  return (
                    <div 
                      key={idx} 
                      className="bg-slate-950/70 border border-slate-800 rounded-lg p-3 hover:border-slate-700 transition-colors flex flex-col gap-2"
                    >
                      <div className="flex items-center justify-between">
                        <span className="inline-block bg-slate-800 border border-slate-700 text-emerald-400 font-bold px-2 py-0.5 rounded text-xs">
                          [{idx + 1}]
                        </span>
                        <span className="text-[10px] text-slate-500 font-medium uppercase">
                          Conf: {item.confidence.toFixed(2)}
                        </span>
                      </div>

                      <p className={`text-slate-300 text-xs leading-relaxed ${isExpanded ? '' : 'line-clamp-3'}`}>
                        {item.content}
                      </p>

                      <div className="flex items-center justify-between border-t border-slate-900 mt-1 pt-2">
                        {item.source_ids.map(sid => (
                          <button
                            key={sid}
                            onClick={() => handleSourceClick(sid)}
                            className="text-[10px] text-emerald-400 hover:text-emerald-300 hover:underline flex items-center gap-1"
                          >
                            <FileText className="w-3 h-3" /> {sid}
                          </button>
                        ))}
                        <button
                          onClick={() => toggleEvidenceExpand(idx)}
                          className="text-slate-400 hover:text-slate-200 text-xs flex items-center gap-0.5"
                        >
                          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Graph Neighborhood and Traces (Columns 10 to 12) */}
        <div className={`col-span-1 md:col-span-3 flex flex-col gap-6 overflow-y-auto pr-1 ${
          activeTab !== 'graph' ? 'hidden md:flex' : ''
        }`}>
          {/* Path Trace Container */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col min-h-[180px]">
            <h3 className="text-md font-semibold text-slate-200 mb-3 flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" /> Retrieval Trace
            </h3>
            
            {answer?.evidence_trace ? (
              <div className="flex-1 flex flex-col gap-2 bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 text-xs">
                <div>
                  <span className="text-slate-400 block font-semibold">Mode Utilized:</span>
                  <span className="text-emerald-400 capitalize font-medium">{answer.mode_used}</span>
                </div>
                <div>
                  <span className="text-slate-400 block font-semibold">Metadata Excerpts:</span>
                  <span className="text-slate-200">{answer.evidence?.length || 0} chunks</span>
                </div>
                <div>
                  <span className="text-slate-400 block font-semibold">Citations:</span>
                  <span className="text-slate-200">{answer.citations.length} sources matched</span>
                </div>
                <div className="mt-1">
                  <span className="text-slate-400 block font-semibold mb-1">Execution path trace:</span>
                  <ol className="list-decimal list-inside text-slate-300 flex flex-col gap-1">
                    {answer.evidence_trace.map((step, sIdx) => (
                      <li key={sIdx}>{step}</li>
                    ))}
                  </ol>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-500 text-xs text-center border border-dashed border-slate-800 rounded-lg">
                No active execution trace
              </div>
            )}
          </div>

          {/* Cytoscape.js visualization canvas */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex-1 flex flex-col min-h-[300px]">
            <h3 className="text-md font-semibold text-slate-200 mb-3 flex items-center gap-2">
              <GitFork className="w-4 h-4 text-emerald-400" /> Knowledge Graph Canvas
            </h3>
            <GraphCanvas 
              graphPaths={answer?.graph_paths || []}
              onNodeClick={(entityName) => {
                console.log("Selected node:", entityName);
              }}
              onEdgeClick={(sourceIds) => {
                if (sourceIds && sourceIds.length > 0) {
                  handleSourceClick(sourceIds[0]);
                }
              }}
            />
          </div>
        </div>
      </main>

      {/* Citation Metadata Modal Overlay */}
      {selectedEvidence && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-lg w-full p-6 shadow-2xl flex flex-col gap-4 relative animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                <Info className="w-5 h-5 text-emerald-400" /> Document Reference
              </h3>
              <button 
                onClick={() => setSelectedEvidence(null)}
                className="text-slate-400 hover:text-slate-200 text-sm font-semibold hover:bg-slate-800 px-2 py-1 rounded"
              >
                Close
              </button>
            </div>
            
            <div className="flex flex-col gap-3 text-xs md:text-sm">
              <div>
                <span className="text-slate-400 block text-xs">Title</span>
                <span className="text-slate-100 font-semibold">{selectedEvidence.title}</span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-slate-400 block text-xs">Source/Doc ID</span>
                  <span className="text-slate-200 font-mono">{selectedEvidence.document_id}</span>
                </div>
                {selectedEvidence.section && (
                  <div>
                    <span className="text-slate-400 block text-xs">Section</span>
                    <span className="text-emerald-400 font-semibold">{selectedEvidence.section}</span>
                  </div>
                )}
              </div>
              <div>
                <span className="text-slate-400 block text-xs mb-1">Excerpt</span>
                <div className="bg-slate-950 p-3 rounded border border-slate-800 text-slate-300 leading-relaxed text-xs max-h-40 overflow-y-auto">
                  {selectedEvidence.excerpt}
                </div>
              </div>
              {selectedEvidence.url && (
                <div>
                  <a 
                    href={selectedEvidence.url} 
                    target="_blank" 
                    rel="noreferrer"
                    className="text-emerald-400 hover:text-emerald-300 hover:underline flex items-center gap-1 text-xs"
                  >
                    <ExternalLink className="w-3.5 h-3.5" /> View original publication source
                  </a>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
