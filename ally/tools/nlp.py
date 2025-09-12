"""
NLP Event Extraction Tool for Ally
Extract structured financial events from text sources
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any

from ..tools import register
from ..schemas.base import ToolResult, Meta
from ..schemas.nlp import NLPEventIn, NLPEventOut, NLPEvent
from ..utils.nlp_rules import (
    extract_tickers, extract_dates, classify_category,
    score_sentiment, extract_snippet, windowize
)
from ..utils.hashing import hash_inputs, hash_code
from ..utils.serialization import convert_timestamps


@register("nlp.extract_events")
def nlp_extract_events(**kwargs) -> ToolResult:
    """
    Extract structured financial events from text sources
    
    Args:
        **kwargs: Parameters matching NLPEventIn schema
        
    Returns:
        ToolResult with NLPEventOut containing extracted events
    """
    try:
        params = NLPEventIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    events = []
    
    for source in params.sources:
        # Load content based on source type
        if source.startswith("text://"):
            # Direct text content
            content = source[7:]  # Remove "text://" prefix
            source_path = source
        elif source.startswith("file://"):
            # File path
            file_path = source[7:]  # Remove "file://" prefix
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                source_path = source
            except Exception as e:
                # Skip files that can't be read
                continue
        else:
            # Assume it's a file path without prefix
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                source_path = f"file://{source}"
            except:
                # Try as direct text
                content = source
                source_path = f"text://{source[:50]}..."
        
        # Extract components
        tickers = extract_tickers(content, whitelist=params.tickers)
        dates = extract_dates(content)
        category = classify_category(content)
        sentiment, confidence = score_sentiment(content)
        
        # If no tickers found but we have content, try to extract from context
        if not tickers and len(content) > 20:
            # Look for company names and map to tickers (simplified)
            potential_tickers = []
            if 'apple' in content.lower():
                potential_tickers.append('AAPL')
            elif 'tesla' in content.lower():
                potential_tickers.append('TSLA')
            elif 'microsoft' in content.lower():
                potential_tickers.append('MSFT')
            elif 'amazon' in content.lower():
                potential_tickers.append('AMZN')
            
            # Apply whitelist filter if provided
            if params.tickers:
                tickers = [t for t in potential_tickers if t in params.tickers]
            else:
                tickers = potential_tickers
        
        # If no dates found, use a default (today)
        if not dates:
            from datetime import datetime
            dates = [datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')]
        
        # Create events for each ticker-date combination
        for ticker in tickers:
            for date in dates:
                # Find best snippet - look for ticker mention
                snippet = extract_snippet(content, ticker, max_length=120)
                
                event = NLPEvent(
                    ticker=ticker,
                    date=date,
                    category=category,
                    sentiment=sentiment,
                    confidence=confidence,
                    snippet=snippet,
                    source_path=source_path
                )
                events.append(event)
        
        # If we have dates but no tickers, still create an event with a generic ticker
        if dates and not tickers and params.tickers:
            # Use first ticker from filter if available
            for date in dates:
                event = NLPEvent(
                    ticker=params.tickers[0] if params.tickers else "UNKNOWN",
                    date=date,
                    category=category,
                    sentiment=sentiment,
                    confidence=confidence,
                    snippet=extract_snippet(content, None, max_length=120),
                    source_path=source_path
                )
                events.append(event)
    
    # Sort events by date then ticker for determinism
    events.sort(key=lambda e: (e.date, e.ticker))
    
    # Generate audit hash for reproducibility
    event_data = [e.model_dump() for e in events]
    audit_content = json.dumps({
        'inputs': params.model_dump(),
        'events': event_data
    }, sort_keys=True)
    audit_hash = hashlib.sha256(audit_content.encode()).hexdigest()
    
    # Create output
    output = NLPEventOut(
        events=events,
        window_days=params.window_days,
        audit_hash=audit_hash
    )
    
    # Create metadata
    meta = Meta(
        tool="nlp.extract_events",
        version="1.0.0",
        timestamp=None,  # Will be set by convert_timestamps
        provenance={
            "inputs_hash": hash_inputs(params.model_dump()),
            "code_hash": hash_code(nlp_extract_events),
            "event_count": len(events),
            "sources_count": len(params.sources)
        }
    )
    
    result = ToolResult(
        ok=True,
        data=output.model_dump(),
        meta=meta.model_dump()
    )
    
    # Ensure all timestamps are properly formatted
    return convert_timestamps(result)


@register("nlp.generate_sample")
def nlp_generate_sample(**kwargs) -> ToolResult:
    """
    Generate sample text for testing NLP event extraction
    
    Args:
        **kwargs: ticker and event_type parameters
        
    Returns:
        ToolResult with sample text
    """
    ticker = kwargs.get('ticker', 'AAPL')
    event_type = kwargs.get('event_type', 'earnings')
    
    samples = {
        "earnings": f"{ticker} reported strong Q3 2024 earnings on October 15, 2024, beating analyst estimates with revenue of $95B and EPS of $1.64. The company saw significant growth in services revenue.",
        "guidance": f"{ticker} raises full-year 2024 guidance on January 3, 2025, citing strong demand and improved margins. Management now expects revenue growth of 15-18% with an upbeat outlook for the coming quarters.",
        "litigation": f"SEC opens investigation into {ticker} accounting practices as of December 20, 2024. A class action lawsuit has been filed alleging misleading statements about product safety.",
        "product": f"{ticker} announces revolutionary new product launch scheduled for March 1, 2025. The innovative technology is expected to disrupt the market and drive significant revenue growth.",
        "regulatory": f"FDA grants approval for {ticker}'s new treatment on November 10, 2024. The regulatory clearance opens a $10B market opportunity for the company.",
        "macro": f"Fed rate decision impacts tech sector including {ticker} on September 18, 2024. Rising inflation and GDP concerns weigh on market sentiment."
    }
    
    sample_text = samples.get(event_type, samples["earnings"])
    
    meta = Meta(
        tool="nlp.generate_sample",
        version="1.0.0",
        timestamp=None,
        provenance={
            "ticker": ticker,
            "event_type": event_type
        }
    )
    
    result = ToolResult(
        ok=True,
        data={"text": sample_text, "source": f"text://{sample_text[:50]}..."},
        meta=meta.model_dump()
    )
    
    return convert_timestamps(result)