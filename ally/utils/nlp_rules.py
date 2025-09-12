"""
NLP rule-based utilities for deterministic event extraction
No network calls, pure rule-based processing for reproducibility
"""

import re
from datetime import datetime
from typing import List, Optional, Tuple


def extract_tickers(text: str, whitelist: Optional[List[str]] = None) -> List[str]:
    """
    Extract ticker symbols from text using regex
    
    Args:
        text: Input text to search
        whitelist: Optional list of valid tickers to filter against
        
    Returns:
        List of unique ticker symbols found
    """
    # Pattern: 1-5 uppercase letters, often in parens or after company name
    # Look for patterns like "Apple (AAPL)" or "AAPL" standalone
    pattern = r'\b([A-Z]{1,5})\b(?:\s|,|\.|\)|$)'
    
    # Also look for explicit ticker mentions in parentheses
    paren_pattern = r'\(([A-Z]{1,5})\)'
    
    tickers = set()
    
    # Find all matches
    for match in re.finditer(pattern, text):
        ticker = match.group(1)
        # Filter out common words that aren't tickers
        if ticker not in {'I', 'A', 'THE', 'AND', 'OR', 'NOT', 'FOR', 'CEO', 'CFO', 'IPO', 'SEC', 'FDA', 'FTC', 'DOJ', 'CPI', 'GDP', 'ETF'}:
            tickers.add(ticker)
    
    # Add parenthetical tickers (more likely to be real tickers)
    for match in re.finditer(paren_pattern, text):
        tickers.add(match.group(1))
    
    # Apply whitelist filter if provided
    if whitelist:
        tickers = {t for t in tickers if t in whitelist}
    
    return sorted(list(tickers))


def extract_dates(text: str) -> List[str]:
    """
    Extract dates from text and normalize to ISO-8601 with Z suffix
    
    Args:
        text: Input text to search
        
    Returns:
        List of ISO-8601 formatted dates with Z suffix (UTC)
    """
    dates = []
    
    # Pattern 1: YYYY-MM-DD
    iso_pattern = r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b'
    for match in re.finditer(iso_pattern, text):
        year, month, day = match.groups()
        try:
            dt = datetime(int(year), int(month), int(day), 12, 0, 0)  # Noon UTC
            dates.append(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
        except ValueError:
            pass
    
    # Pattern 2: Month DD, YYYY (e.g., "January 3, 2025")
    month_names = r'(January|February|March|April|May|June|July|August|September|October|November|December|' \
                  r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    text_date_pattern = rf'\b{month_names}\s+(\d{{1,2}}),?\s+(\d{{4}})\b'
    
    month_map = {
        'January': 1, 'Jan': 1, 'February': 2, 'Feb': 2, 'March': 3, 'Mar': 3,
        'April': 4, 'Apr': 4, 'May': 5, 'June': 6, 'Jun': 6,
        'July': 7, 'Jul': 7, 'August': 8, 'Aug': 8, 'September': 9, 'Sep': 9,
        'October': 10, 'Oct': 10, 'November': 11, 'Nov': 11, 'December': 12, 'Dec': 12
    }
    
    for match in re.finditer(text_date_pattern, text, re.IGNORECASE):
        month_str, day, year = match.groups()
        month = month_map.get(month_str.title(), 0)
        if month:
            try:
                dt = datetime(int(year), month, int(day), 12, 0, 0)  # Noon UTC
                dates.append(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
            except ValueError:
                pass
    
    # Return unique dates, sorted
    return sorted(list(set(dates)))


def classify_category(text: str) -> str:
    """
    Classify text into financial event category using keyword rules
    
    Args:
        text: Input text to classify
        
    Returns:
        Category string from predefined set
    """
    text_lower = text.lower()
    
    # Check categories in priority order
    if any(word in text_lower for word in ['earnings', 'revenue', 'profit', 'loss', 'quarter', 'q1', 'q2', 'q3', 'q4', 'eps']):
        return 'earnings'
    
    if any(word in text_lower for word in ['guidance', 'outlook', 'forecast', 'expects', 'projection', 'raises', 'lowers', 'reaffirms']):
        return 'guidance'
    
    if any(word in text_lower for word in ['lawsuit', 'litigation', 'sue', 'court', 'legal', 'class action', 'settlement']):
        return 'litigation'
    
    if any(word in text_lower for word in ['sec', 'fda', 'ftc', 'regulatory', 'approval', 'investigation', 'probe', 'compliance']):
        return 'regulatory'
    
    if any(word in text_lower for word in ['product', 'launch', 'release', 'announce', 'unveil', 'introduce', 'new']):
        return 'product'
    
    if any(word in text_lower for word in ['cpi', 'gdp', 'inflation', 'fed', 'rates', 'economy', 'unemployment', 'fomc']):
        return 'macro'
    
    return 'other'


def score_sentiment(text: str) -> Tuple[str, float]:
    """
    Score sentiment using lexicon-based approach with negation handling
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (sentiment label, confidence score)
    """
    text_lower = text.lower()
    
    # Positive words
    positive_words = {
        'good', 'great', 'excellent', 'positive', 'strong', 'gain', 'rise', 'up', 'increase',
        'improve', 'better', 'outperform', 'beat', 'exceed', 'success', 'profit', 'growth',
        'optimistic', 'bullish', 'upbeat', 'surge', 'rally', 'boom', 'soar', 'jump'
    }
    
    # Negative words
    negative_words = {
        'bad', 'poor', 'negative', 'weak', 'loss', 'fall', 'down', 'decrease', 'decline',
        'worse', 'underperform', 'miss', 'fail', 'concern', 'risk', 'warning', 'threat',
        'pessimistic', 'bearish', 'plunge', 'crash', 'slump', 'drop', 'tumble', 'sink'
    }
    
    # Negation words
    negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'isn\'t', 'wasn\'t', 'doesn\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 'can\'t', 'couldn\'t'}
    
    # Split into words
    words = re.findall(r'\b\w+\b', text_lower)
    
    score = 0.0
    word_count = 0
    
    for i, word in enumerate(words):
        # Check for negation in previous 2 words
        is_negated = False
        for j in range(max(0, i-2), i):
            if words[j] in negation_words:
                is_negated = True
                break
        
        if word in positive_words:
            score += -1.0 if is_negated else 1.0
            word_count += 1
        elif word in negative_words:
            score += 1.0 if is_negated else -1.0
            word_count += 1
    
    # Normalize score
    if word_count > 0:
        score = score / word_count
    
    # Map to sentiment label with thresholds
    if score > 0.15:
        sentiment = 'pos'
    elif score < -0.15:
        sentiment = 'neg'
    else:
        sentiment = 'neu'
    
    # Confidence is absolute value of score, clamped to [0,1]
    confidence = min(1.0, abs(score))
    
    return sentiment, confidence


def extract_snippet(text: str, keyword: str = None, max_length: int = 120) -> str:
    """
    Extract a snippet around a keyword or from the beginning of text
    
    Args:
        text: Source text
        keyword: Optional keyword to center snippet around
        max_length: Maximum snippet length
        
    Returns:
        Text snippet
    """
    if not text:
        return ""
    
    if keyword and keyword.lower() in text.lower():
        # Find keyword position (case-insensitive)
        lower_text = text.lower()
        pos = lower_text.find(keyword.lower())
        
        # Center snippet around keyword
        start = max(0, pos - max_length // 2)
        end = min(len(text), start + max_length)
        
        # Adjust start if we're at the end
        if end == len(text):
            start = max(0, end - max_length)
        
        snippet = text[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet[3:]
        if end < len(text):
            snippet = snippet[:-3] + "..."
            
        return snippet
    else:
        # Just take from beginning
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."


def windowize(date: str, window_days: int) -> dict:
    """
    Calculate event study window (just sizes, no price data)
    
    Args:
        date: Event date (ISO-8601)
        window_days: Number of days before/after
        
    Returns:
        Dict with pre_days and post_days
    """
    return {
        'pre_days': window_days,
        'post_days': window_days,
        'event_date': date
    }