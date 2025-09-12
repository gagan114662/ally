import json
from datetime import datetime
from typing import List, Dict, Any
from ally.schemas.research import Evidence, EvidenceGrade, Claim, ResearchSummary
from ally.utils.text import deduplicate_texts, extract_keywords
from ally.schemas.base import ToolResult
from . import register


def bayesian_aggregate(evidence_list: List[Evidence], base_prior: float = 0.5) -> float:
    """
    Aggregate evidence using Bayesian updating.
    
    Grade weights: HIGH=0.9, MEDIUM=0.7, LOW=0.5
    """
    grade_weights = {
        EvidenceGrade.HIGH: 0.9,
        EvidenceGrade.MEDIUM: 0.7, 
        EvidenceGrade.LOW: 0.5
    }
    
    posterior = base_prior
    
    for evidence in evidence_list:
        weight = grade_weights[evidence.grade]
        # Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
        # Simplified: new_posterior = weight * old_posterior + (1-weight) * (1-old_posterior)
        posterior = weight * posterior + (1 - weight) * (1 - posterior)
    
    return min(1.0, max(0.0, posterior))


@register("research.analyze")
def research_analyze(
    query: str, 
    sources: List[Dict[str, Any]], 
    methodology: str = "bayesian_aggregation",
    dedup_threshold: float = 0.8
) -> ToolResult:
    """
    Analyze research sources to extract claims and evidence.
    
    Args:
        query: Research question
        sources: List of source documents with 'content', 'source', 'grade' fields
        methodology: Analysis methodology to use
        dedup_threshold: Similarity threshold for deduplication
    """
    try:
        timestamp = datetime.now()
        
        # Convert sources to Evidence objects
        evidence_list = []
        contents = []
        
        for i, source in enumerate(sources):
            evidence = Evidence(
                source=source.get('source', f'source_{i}'),
                content=source.get('content', ''),
                grade=EvidenceGrade(source.get('grade', 'medium')),
                timestamp=timestamp,
                metadata=source.get('metadata', {})
            )
            evidence_list.append(evidence)
            contents.append(evidence.content)
        
        # Deduplicate evidence
        unique_indices = deduplicate_texts(contents, threshold=dedup_threshold)
        unique_evidence = [evidence_list[i] for i in unique_indices]
        
        # Extract keywords from query for claim generation
        keywords = extract_keywords(query, top_k=3)
        
        # Generate claims based on evidence aggregation
        claims = []
        
        # Group evidence by keywords/topics
        keyword_evidence = {kw: [] for kw in keywords}
        for evidence in unique_evidence:
            for keyword in keywords:
                if keyword.lower() in evidence.content.lower():
                    keyword_evidence[keyword].append(evidence)
        
        # Create claims for each topic with enough evidence
        for keyword, topic_evidence in keyword_evidence.items():
            if len(topic_evidence) >= 2:  # Require at least 2 pieces of evidence
                confidence = bayesian_aggregate(topic_evidence)
                
                claim = Claim(
                    statement=f"Evidence supports {keyword} related to: {query}",
                    confidence=confidence,
                    evidence_ids=[f"{ev.source}_{hash(ev.content) % 10000}" for ev in topic_evidence]
                )
                claims.append(claim)
        
        # Create overall summary claim if we have evidence
        if unique_evidence:
            overall_confidence = bayesian_aggregate(unique_evidence)
            summary_claim = Claim(
                statement=f"Overall research finding for: {query}",
                confidence=overall_confidence,
                evidence_ids=[f"{ev.source}_{hash(ev.content) % 10000}" for ev in unique_evidence]
            )
            claims.append(summary_claim)
        
        # Create research summary
        summary = ResearchSummary(
            query=query,
            claims=claims,
            evidence=unique_evidence,
            methodology=methodology,
            timestamp=timestamp
        )
        
        return ToolResult(
            ok=True,
            data={
                "summary": summary.dict(),
                "stats": {
                    "original_sources": len(sources),
                    "unique_evidence": len(unique_evidence),
                    "claims_generated": len(claims),
                    "deduplication_ratio": len(unique_evidence) / len(sources) if sources else 0
                }
            }
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": f"Research analysis failed: {str(e)}"}
        )


@register("research.synthesize")
def research_synthesize(
    summaries: List[Dict[str, Any]], 
    synthesis_method: str = "consensus_weighting"
) -> ToolResult:
    """
    Synthesize multiple research summaries into a unified view.
    
    Args:
        summaries: List of ResearchSummary dicts
        synthesis_method: Method for synthesis
    """
    try:
        if not summaries:
            return ToolResult(ok=False, data={"error": "No summaries provided"})
        
        # Aggregate all evidence
        all_evidence = []
        all_claims = []
        
        for summary_data in summaries:
            summary = ResearchSummary(**summary_data)
            all_evidence.extend(summary.evidence)
            all_claims.extend(summary.claims)
        
        # Deduplicate evidence across summaries
        evidence_contents = [ev.content for ev in all_evidence]
        unique_indices = deduplicate_texts(evidence_contents, threshold=0.9)
        unique_evidence = [all_evidence[i] for i in unique_indices]
        
        # Create synthesis claims
        synthesis_claims = []
        
        if synthesis_method == "consensus_weighting":
            # Group similar claims
            claim_texts = [claim.statement for claim in all_claims]
            unique_claim_indices = deduplicate_texts(claim_texts, threshold=0.7)
            
            for idx in unique_claim_indices:
                representative_claim = all_claims[idx]
                
                # Find all similar claims
                similar_confidences = []
                for i, claim in enumerate(all_claims):
                    from ally.utils.similarity import cosine_similarity
                    sim = cosine_similarity(representative_claim.statement, claim.statement)
                    if sim >= 0.7:
                        similar_confidences.append(claim.confidence)
                
                # Average confidence with consensus weighting
                consensus_confidence = sum(similar_confidences) / len(similar_confidences)
                
                synthesis_claim = Claim(
                    statement=representative_claim.statement,
                    confidence=consensus_confidence,
                    evidence_ids=representative_claim.evidence_ids
                )
                synthesis_claims.append(synthesis_claim)
        
        # Create unified summary
        unified_query = " | ".join([s.get("query", "") for s in summaries])
        
        unified_summary = ResearchSummary(
            query=f"Synthesis of: {unified_query}",
            claims=synthesis_claims,
            evidence=unique_evidence,
            methodology=f"synthesis_{synthesis_method}",
            timestamp=datetime.now()
        )
        
        return ToolResult(
            ok=True,
            data={
                "synthesis": unified_summary.dict(),
                "stats": {
                    "input_summaries": len(summaries),
                    "total_evidence": len(all_evidence),
                    "unique_evidence": len(unique_evidence),
                    "synthesis_claims": len(synthesis_claims)
                }
            }
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": f"Research synthesis failed: {str(e)}"}
        )