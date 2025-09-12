import os
import json
import pytest
from datetime import datetime
from ally.tools import TOOL_REGISTRY
from ally.schemas.research import Evidence, EvidenceGrade, Claim, ResearchSummary
from ally.utils.similarity import cosine_similarity
from ally.utils.text import deduplicate_texts, cluster_texts, extract_keywords

pytestmark = pytest.mark.mresearch


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    # Identical texts should have similarity 1.0
    text1 = "This is a test document about machine learning"
    assert cosine_similarity(text1, text1) == pytest.approx(1.0, rel=1e-3)
    
    # Similar texts should have high similarity
    text2 = "This is a test document about artificial intelligence"
    sim = cosine_similarity(text1, text2)
    assert 0.5 < sim < 1.0
    
    # Different texts should have lower similarity
    text3 = "The weather is sunny today"
    sim = cosine_similarity(text1, text3)
    assert 0.0 <= sim < 0.5
    
    # Empty texts should return 0
    assert cosine_similarity("", "test") == 0.0
    assert cosine_similarity("test", "") == 0.0


def test_text_deduplication():
    """Test text deduplication functionality."""
    texts = [
        "The company reported strong quarterly results",
        "The company announced strong quarterly earnings",  # Similar to first
        "New product launch scheduled for next month", 
        "Product launch planned for upcoming month",  # Similar to third
        "Weather forecast shows rain tomorrow"  # Different topic
    ]
    
    unique_indices = deduplicate_texts(texts, threshold=0.7)
    
    # Should identify fewer unique texts than original (due to deduplication)
    assert len(unique_indices) <= len(texts)
    assert len(unique_indices) >= 3  # Should have at least 3 clusters
    
    # First text should be kept (index 0)
    assert 0 in unique_indices


def test_text_clustering():
    """Test text clustering functionality."""
    texts = [
        "Revenue increased by 15% this quarter",
        "Quarterly revenue grew significantly", 
        "New AI product launched successfully",
        "Artificial intelligence tool released",
        "Stock price reached all-time high"
    ]
    
    from ally.utils.text import cluster_texts as cluster_fn
    clusters = cluster_fn(texts, threshold=0.6)
    
    # Should have at least 2 clusters (revenue and product topics)
    assert len(clusters) >= 2
    
    # Each text should be assigned to exactly one cluster
    all_assigned = []
    for cluster_texts in clusters.values():
        all_assigned.extend(cluster_texts)
    assert len(all_assigned) == len(texts)


def test_keyword_extraction():
    """Test keyword extraction."""
    text = "The company reported strong quarterly revenue growth driven by technology innovation and market expansion"
    
    keywords = extract_keywords(text, top_k=3)
    
    assert len(keywords) <= 3
    assert all(len(kw) > 2 for kw in keywords)  # No short words
    assert "company" in keywords or "revenue" in keywords or "technology" in keywords


def test_research_schemas():
    """Test research schema models."""
    # Test Evidence model
    evidence = Evidence(
        source="test_source",
        content="This is test evidence content",
        grade=EvidenceGrade.HIGH,
        timestamp=datetime.now(),
        metadata={"test": "value"}
    )
    
    assert evidence.source == "test_source"
    assert evidence.grade == EvidenceGrade.HIGH
    assert "test" in evidence.metadata
    
    # Test Claim model
    claim = Claim(
        statement="Test claim statement",
        confidence=0.85,
        evidence_ids=["evidence_1", "evidence_2"]
    )
    
    assert claim.confidence == 0.85
    assert len(claim.evidence_ids) == 2
    
    # Test ResearchSummary model
    summary = ResearchSummary(
        query="Test research query",
        claims=[claim],
        evidence=[evidence],
        methodology="test_method",
        timestamp=datetime.now()
    )
    
    assert summary.query == "Test research query"
    assert len(summary.claims) == 1
    assert len(summary.evidence) == 1


def load_fixture(filename: str):
    """Load test fixture data."""
    fixture_path = f"data/fixtures/research/{filename}"
    with open(fixture_path, 'r') as f:
        return json.load(f)


def test_research_analyze_tool():
    """Test research.analyze tool with fixtures."""
    # Load test data
    sources = load_fixture("evidence_sample.json")
    query_data = load_fixture("research_query.json")
    
    # Execute research analysis
    result = TOOL_REGISTRY["research.analyze"](
        query=query_data["query"],
        sources=sources,
        methodology=query_data["methodology"],
        dedup_threshold=query_data["dedup_threshold"]
    )
    
    assert result.ok
    
    # Verify result structure
    data = result.data
    assert "summary" in data
    assert "stats" in data
    
    # Verify stats
    stats = data["stats"]
    assert stats["original_sources"] == len(sources)
    assert stats["unique_evidence"] <= len(sources)  # Should be <= due to deduplication
    assert stats["claims_generated"] >= 1
    
    # Verify summary structure
    summary = data["summary"]
    assert "query" in summary
    assert "claims" in summary
    assert "evidence" in summary
    assert "methodology" in summary
    assert "timestamp" in summary
    
    # Verify claims have valid confidence scores
    for claim in summary["claims"]:
        assert 0.0 <= claim["confidence"] <= 1.0
        assert len(claim["statement"]) > 0


def test_research_analyze_empty_sources():
    """Test research analysis with empty sources."""
    result = TOOL_REGISTRY["research.analyze"](
        query="Test query",
        sources=[],
        methodology="bayesian_aggregation"
    )
    
    assert result.ok
    # Should handle empty sources gracefully
    assert result.data["stats"]["original_sources"] == 0


def test_research_synthesize_tool():
    """Test research.synthesize tool."""
    # Create mock research summaries
    summary1 = {
        "query": "Growth outlook for TechCorp",
        "claims": [
            {
                "statement": "TechCorp shows strong revenue growth",
                "confidence": 0.8,
                "evidence_ids": ["ev1", "ev2"]
            }
        ],
        "evidence": [
            {
                "source": "analyst_report",
                "content": "Strong revenue growth observed",
                "grade": "high",
                "timestamp": "2024-01-01T00:00:00",
                "metadata": {}
            }
        ],
        "methodology": "bayesian_aggregation",
        "timestamp": "2024-01-01T00:00:00"
    }
    
    summary2 = {
        "query": "TechCorp market position",
        "claims": [
            {
                "statement": "TechCorp maintains competitive advantage",
                "confidence": 0.7,
                "evidence_ids": ["ev3"]
            }
        ],
        "evidence": [
            {
                "source": "market_analysis",
                "content": "Competitive positioning remains strong",
                "grade": "medium", 
                "timestamp": "2024-01-02T00:00:00",
                "metadata": {}
            }
        ],
        "methodology": "bayesian_aggregation",
        "timestamp": "2024-01-02T00:00:00"
    }
    
    # Execute synthesis
    result = TOOL_REGISTRY["research.synthesize"](
        summaries=[summary1, summary2],
        synthesis_method="consensus_weighting"
    )
    
    assert result.ok
    
    # Verify synthesis structure
    data = result.data
    assert "synthesis" in data
    assert "stats" in data
    
    # Verify stats
    stats = data["stats"]
    assert stats["input_summaries"] == 2
    assert stats["total_evidence"] >= 2
    assert stats["synthesis_claims"] >= 1
    
    # Verify synthesis
    synthesis = data["synthesis"]
    assert "query" in synthesis
    assert "claims" in synthesis
    assert "evidence" in synthesis
    assert "methodology" in synthesis


def test_research_synthesize_empty():
    """Test research synthesis with empty input."""
    result = TOOL_REGISTRY["research.synthesize"](
        summaries=[],
        synthesis_method="consensus_weighting"
    )
    
    assert not result.ok
    assert "error" in result.data
    assert "No summaries provided" in result.data["error"]


def test_deterministic_fixtures():
    """Test that fixtures produce deterministic results."""
    sources = load_fixture("evidence_sample.json")
    query_data = load_fixture("research_query.json")
    
    # Run analysis twice
    result1 = TOOL_REGISTRY["research.analyze"](
        query=query_data["query"],
        sources=sources,
        methodology=query_data["methodology"],
        dedup_threshold=query_data["dedup_threshold"]
    )
    
    result2 = TOOL_REGISTRY["research.analyze"](
        query=query_data["query"],
        sources=sources,
        methodology=query_data["methodology"],
        dedup_threshold=query_data["dedup_threshold"]
    )
    
    # Results should be consistent (same number of claims/evidence)
    assert result1.data["stats"]["unique_evidence"] == result2.data["stats"]["unique_evidence"]
    assert result1.data["stats"]["claims_generated"] == result2.data["stats"]["claims_generated"]
    
    # Evidence content should be identical
    ev1_content = {ev["content"] for ev in result1.data["summary"]["evidence"]}
    ev2_content = {ev["content"] for ev in result2.data["summary"]["evidence"]}
    assert ev1_content == ev2_content