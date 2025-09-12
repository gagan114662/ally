# Ally Claims Document

## Milestone 4: Computer Vision (CV) ✅

### CV-001: Deterministic Synthetic Data Generation
- **Claim**: Same seed produces identical OHLCV data with embedded patterns
- **Verification**: Run `cv.generate_synthetic` twice with seed=42, compare hashes
- **Expected**: Identical SHA256 hashes

### CV-002: Pattern Detection with Numeric Confirmation
- **Claim**: Detects chart patterns using both visual and numeric rules
- **Verification**: Run `cv.detect_chart_patterns` on synthetic data
- **Expected**: Detections with numeric confirmations (ATR, price levels)

### CV-003: Off-Screen Chart Rendering
- **Claim**: Renders candlestick charts to base64 without display
- **Verification**: Check chart_base64 field length > 100 bytes
- **Expected**: Valid base64 PNG image data

## Milestone 5: NLP Event Signals ✅

### NLP-001: Event Extraction
- **Claim**: Extracts (ticker, date, category, sentiment, confidence, snippet, source) from news1.txt
- **Verification**: Run `nlp.extract_events` on data/fixtures/text/news1.txt
- **Expected**: AAPL event with date ending in Z, valid category and sentiment

### NLP-002: Ticker Filter
- **Claim**: tickers parameter filters events to only requested symbols
- **Verification**: Run with tickers=["AAPL"] on mixed content
- **Expected**: Only AAPL events returned

### NLP-003: Determinism
- **Claim**: Same inputs produce identical event extractions
- **Verification**: Run twice on same input, compare SHA1 of sorted events
- **Expected**: Identical fingerprints

### NLP-004: Schema Validation
- **Claim**: Output validates against nlpevents.schema.json
- **Verification**: Validate tool output with JSON Schema
- **Expected**: PASS with all required fields present

### NLP-005: Event Window Sizes
- **Claim**: window_days parameter correctly sets event study window
- **Verification**: Test with window_days=3,5,10
- **Expected**: Output window_days matches input

## Running Verification

```bash
# Run all claims verification
python -m ally.verify.verify_claims

# Run specific milestone tests
pytest tests/test_cv_detect.py -v
pytest tests/test_nlp_events.py -v
```

## Verification Output Format

Each claim outputs:
1. Raw ToolResult JSON (first 2 events/detections)
2. SHA1/SHA256 fingerprint for determinism
3. Schema validation result (PASS/FAIL)
4. Numeric confirmation values

## Determinism Guarantees

All tools provide:
- `audit_hash`: SHA256 of inputs + outputs
- `inputs_hash`: Hash of input parameters
- `code_hash`: Hash of function code
- Sorted outputs for consistent ordering
- Fixed random seeds for synthetic data