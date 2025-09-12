import numpy as np
from typing import List, Dict, Set
from collections import Counter
import re


def tokenize(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace and punctuation."""
    return re.findall(r'\b\w+\b', text.lower())


def text_to_vector(text: str, vocab: Set[str]) -> np.ndarray:
    """Convert text to TF vector based on vocabulary."""
    tokens = tokenize(text)
    token_counts = Counter(tokens)
    vector = np.zeros(len(vocab))
    
    for i, word in enumerate(sorted(vocab)):
        vector[i] = token_counts.get(word, 0)
    
    return vector


def cosine_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two text strings."""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    # Build vocabulary from both texts
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))
    vocab = tokens1.union(tokens2)
    
    if not vocab:
        return 0.0
    
    # Convert to vectors
    vec1 = text_to_vector(text1, vocab)
    vec2 = text_to_vector(text2, vocab)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)