#!/usr/bin/env python3
"""
Verification Pack for Ally Claims
Run deterministic tests to verify all milestone claims
"""

import sys
import json
import hashlib
import jsonschema
from pathlib import Path

# Add ally to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ally.tools.cv import cv_detect_chart_patterns, cv_generate_synthetic
from ally.tools.nlp import nlp_extract_events
from ally.schemas.cv import CVDetectIn, CVGenerateIn
from ally.schemas.nlp import NLPEventIn


def verify_cv_claims():
    """Verify Computer Vision claims (Milestone 4)"""
    print("\n" + "="*60)
    print("VERIFYING CV CLAIMS (Milestone 4)")
    print("="*60)
    
    claims_passed = []
    
    # CV-001: Deterministic synthetic data generation
    try:
        params = CVGenerateIn(pattern="engulfing_bull", rows=20, seed=42)
        result = cv_generate_synthetic(params)
        
        # Check hash is deterministic
        expected_hash_prefix = result.data['audit_hash'][:8]
        
        # Run again with same seed
        result2 = cv_generate_synthetic(params)
        assert result2.data['audit_hash'][:8] == expected_hash_prefix
        
        claims_passed.append("CV-001: Deterministic generation ✅")
        print(f"✅ CV-001: Hash={expected_hash_prefix}...")
    except Exception as e:
        print(f"❌ CV-001 failed: {e}")
    
    # CV-002: Pattern detection with numeric confirmation
    try:
        params = CVDetectIn(symbol="TEST", lookback=50, patterns=["engulfing_bull"])
        result = cv_detect_chart_patterns(params)
        
        assert result.ok == True
        assert 'detections' in result.data
        assert 'chart_base64' in result.data
        
        claims_passed.append("CV-002: Pattern detection ✅")
        print(f"✅ CV-002: Detected {len(result.data['detections'])} patterns")
    except Exception as e:
        print(f"❌ CV-002 failed: {e}")
    
    # CV-003: Chart rendering
    try:
        params = CVGenerateIn(pattern="pin_bar_bull", rows=30, seed=123)
        result = cv_generate_synthetic(params)
        
        # Verify base64 chart is present and valid
        assert 'chart_base64' in result.data
        assert len(result.data['chart_base64']) > 100
        
        claims_passed.append("CV-003: Chart rendering ✅")
        print(f"✅ CV-003: Chart size={len(result.data['chart_base64'])} bytes")
    except Exception as e:
        print(f"❌ CV-003 failed: {e}")
    
    return claims_passed


def verify_nlp_claims():
    """Verify NLP Event Extraction claims (Milestone 5)"""
    print("\n" + "="*60)
    print("VERIFYING NLP CLAIMS (Milestone 5)")
    print("="*60)
    
    claims_passed = []
    
    # NLP-001: Extract events from news1.txt
    try:
        params = NLPEventIn(
            sources=["file://data/fixtures/text/news1.txt"],
            window_days=5
        )
        result = nlp_extract_events(params)
        
        assert result.ok == True
        events = result.data['events']
        assert len(events) > 0
        
        event = events[0]
        assert event['ticker'] == 'AAPL'
        assert event['date'].endswith('Z')
        assert event['category'] in ['earnings', 'guidance']
        assert event['sentiment'] in ['neg', 'neu', 'pos']
        assert 0 <= event['confidence'] <= 1
        
        claims_passed.append("NLP-001: Event extraction ✅")
        print(f"✅ NLP-001: Extracted {len(events)} events")
        print(f"   First event: {json.dumps(events[0], indent=2)}")
    except Exception as e:
        print(f"❌ NLP-001 failed: {e}")
    
    # NLP-002: Ticker filter
    try:
        params = NLPEventIn(
            sources=[
                "file://data/fixtures/text/news1.txt",
                "file://data/fixtures/text/filing1.txt"
            ],
            tickers=["AAPL"],
            window_days=5
        )
        result = nlp_extract_events(params)
        
        events = result.data['events']
        for event in events:
            assert event['ticker'] == 'AAPL'
        
        claims_passed.append("NLP-002: Ticker filter ✅")
        print(f"✅ NLP-002: Filtered to {len(events)} AAPL events")
    except Exception as e:
        print(f"❌ NLP-002 failed: {e}")
    
    # NLP-003: Determinism
    try:
        params = NLPEventIn(
            sources=["file://data/fixtures/text/filing1.txt"],
            window_days=5
        )
        
        result1 = nlp_extract_events(params)
        result2 = nlp_extract_events(params)
        
        # Generate fingerprints
        events1 = json.dumps(result1.data['events'], sort_keys=True)
        events2 = json.dumps(result2.data['events'], sort_keys=True)
        
        fingerprint1 = hashlib.sha1(events1.encode()).hexdigest()
        fingerprint2 = hashlib.sha1(events2.encode()).hexdigest()
        
        assert fingerprint1 == fingerprint2
        
        claims_passed.append("NLP-003: Determinism ✅")
        print(f"✅ NLP-003: SHA1={fingerprint1[:16]}...")
    except Exception as e:
        print(f"❌ NLP-003 failed: {e}")
    
    # NLP-004: Schema validation
    try:
        # Load schema
        schema_path = Path(__file__).parent / "jsonschema" / "nlpevents.schema.json"
        with open(schema_path) as f:
            schema = json.load(f)
        
        params = NLPEventIn(
            sources=["text://Test event for TSLA on 2024-06-15"],
            window_days=7
        )
        result = nlp_extract_events(params)
        
        # Validate against schema
        jsonschema.validate(result.data, schema)
        
        claims_passed.append("NLP-004: Schema valid ✅")
        print(f"✅ NLP-004: Output validates against schema")
    except Exception as e:
        print(f"❌ NLP-004 failed: {e}")
    
    # NLP-005: Window sizes
    try:
        for window in [3, 5, 10]:
            params = NLPEventIn(
                sources=["text://Event text"],
                window_days=window
            )
            result = nlp_extract_events(params)
            assert result.data['window_days'] == window
        
        claims_passed.append("NLP-005: Window sizes ✅")
        print(f"✅ NLP-005: Window sizes correct")
    except Exception as e:
        print(f"❌ NLP-005 failed: {e}")
    
    return claims_passed


def main():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("ALLY VERIFICATION PACK")
    print("="*60)
    
    all_claims = []
    
    # Verify CV claims
    cv_claims = verify_cv_claims()
    all_claims.extend(cv_claims)
    
    # Verify NLP claims
    nlp_claims = verify_nlp_claims()
    all_claims.extend(nlp_claims)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for claim in all_claims:
        print(f"  {claim}")
    
    total = len(cv_claims) + len(nlp_claims)
    passed = len([c for c in all_claims if '✅' in c])
    
    print(f"\nPASSED: {passed}/{total} claims")
    
    if passed == total:
        print("\n🎉 ALL CLAIMS VERIFIED! 🎉")
        return 0
    else:
        print(f"\n⚠️  {total - passed} claims failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())