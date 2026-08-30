from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Union
from pydantic import BaseModel, Field

# Document model
class Document(BaseModel):
    document_id: str
    title: str
    source_type: str = Field(description="guideline | study | label")
    publisher: str
    authors: List[str]
    year: int
    url: Optional[str] = None
    ingestion_date: datetime = Field(default_factory=datetime.utcnow)

# Chunk model
class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    start_offset: int
    end_offset: int
    section_title: Optional[str] = None
    embedding_id: Optional[str] = None

# Entity model
class Entity(BaseModel):
    entity_id: str
    normalized_name: str
    entity_type: str = Field(description="Drug|Condition|Symptom|SideEffect|Population|ClinicalStudy|Guideline")
    confidence: float
    document_id: str
    source_span: Tuple[int, int]

# Relationship model
# Note: As resolved in architecture review, relationships connect source to target and store citation provenance in source_ids array properties
class Relationship(BaseModel):
    relation_id: str
    source_entity_id: str
    relation_type: str = Field(description="TREATS|CAUSES|HAS_SYMPTOM|INTERACTS_WITH|CONTRAINDICATED_FOR|RECOMMENDS")
    target_entity_id: str
    confidence: float
    source_ids: List[str] = Field(default_factory=list)

# Graph Node API representation
class GraphNode(BaseModel):
    id: str
    label: str
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)

# Graph Edge API representation
class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

# Query Request API model
class QueryRequest(BaseModel):
    question: str = Field(..., max_length=1000)
    mode: str = Field(default="hybrid", description="vector | graph | hybrid")

# Retrieval Result Detail Item
class EvidenceItem(BaseModel):
    type: str = Field(description="chunk | graph_path")
    content: str
    source_ids: List[str]
    confidence: float

# Retrieval Result model
class RetrievalResult(BaseModel):
    evidence_items: List[EvidenceItem] = Field(default_factory=list)

# Evidence Detail (resolved citation metadata)
class Evidence(BaseModel):
    source_id: str
    document_id: str
    title: str
    section: Optional[str] = None
    excerpt: str
    url: Optional[str] = None

# Inline Citation Marker
class CitationMarker(BaseModel):
    marker: int
    source_id: str

# Generated Answer response model
class GeneratedAnswer(BaseModel):
    answer_text: str
    citations: List[CitationMarker] = Field(default_factory=list)
    confidence: str = Field(description="high | medium | low | insufficient_evidence")
    mode_used: str = Field(description="vector | graph | hybrid")
    evidence_trace: Optional[List[str]] = Field(default_factory=list, description="Step list trace of search retrieval path")
    evidence: Optional[List[EvidenceItem]] = Field(default_factory=list)
    graph_paths: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    status: Optional[str] = "ok"

# Evaluation Result model
class EvaluationResult(BaseModel):
    run_id: str
    mode: str = Field(description="vector | graph | hybrid")
    category: str = Field(description="direct | relational | two_hop | multi_hop | interaction | contraindication | citation | unanswerable")
    precision_at_k: float
    recall_at_k: float
    faithfulness_score: float
    hallucination_flag: bool
    latency_ms: float
