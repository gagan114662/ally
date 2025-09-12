from typing import List, Dict, Tuple, Set
from ally.utils.similarity import cosine_similarity


def deduplicate_texts(texts: List[str], threshold: float = 0.8) -> List[int]:
    """
    Deduplicate texts using cosine similarity.
    
    Returns:
        List of indices of unique texts (first occurrence of each cluster)
    """
    if not texts:
        return []
    
    unique_indices = []
    seen_clusters = []
    
    for i, text in enumerate(texts):
        is_duplicate = False
        
        for unique_idx in unique_indices:
            similarity = cosine_similarity(text, texts[unique_idx])
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_indices.append(i)
    
    return unique_indices


def cluster_texts(texts: List[str], threshold: float = 0.7) -> Dict[int, List[int]]:
    """
    Cluster texts by similarity.
    
    Returns:
        Dict mapping cluster_id to list of text indices in that cluster
    """
    if not texts:
        return {}
    
    clusters = {}
    cluster_id = 0
    
    for i, text in enumerate(texts):
        assigned = False
        
        # Try to assign to existing cluster
        for cid, indices in clusters.items():
            # Check similarity with cluster representative (first text)
            representative = texts[indices[0]]
            similarity = cosine_similarity(text, representative)
            
            if similarity >= threshold:
                clusters[cid].append(i)
                assigned = True
                break
        
        # Create new cluster if not assigned
        if not assigned:
            clusters[cluster_id] = [i]
            cluster_id += 1
    
    return clusters


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """
    Extract top keywords from text using simple frequency analysis.
    """
    from ally.utils.similarity import tokenize
    from collections import Counter
    
    tokens = tokenize(text)
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    counter = Counter(filtered_tokens)
    return [word for word, count in counter.most_common(top_k)]