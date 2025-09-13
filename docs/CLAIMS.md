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

# CLI examples
ally run nlp.extract_events --json '{"sources":["file://data/fixtures/text/news1.txt"],"tickers":["AAPL"],"window_days":5}'
ally run nlp.generate_sample --json '{"ticker":"TSLA","event_type":"litigation"}'
```

## Verification Output Format

Each claim outputs:
1. Raw ToolResult JSON (first 2 events/detections)
2. SHA1/SHA256 fingerprint for determinism
3. Schema validation result (PASS/FAIL)
4. Numeric confirmation values

## QC Guarantees ✅

### QC-001: Runtime Assertions
- **Claim**: Runtime assertion helpers inject into QuantConnect algorithms
- **Verification**: Run `qc.inject_asserts` on algorithm file
- **PROOF:QC_ASSERTS**: ok
- **PROOF:QC_ASSERT_TRIPS**: 0
- **PROOF:QC_ASSERT_HELPERS**: 5

#### Available Assertion Helpers
- `assert_indicator_ready`: Validates indicator IsReady state
- `assert_warmup_complete`: Checks warmup period completion
- `assert_orders_filled`: Verifies expected order fills
- `assert_history_available`: Confirms history data access
- `assert_portfolio_value`: Guards minimum portfolio value

### M12-001: Portfolio & Multi-Asset Backtesting
- **Claim**: Portfolio allocation and attribution with deterministic proofs
- **Verification**: Run `portfolio.allocate` and `portfolio.attribution` with test fixtures
- **PROOF:PORT_WEIGHTS_SUM**: 1.0
- **PROOF:PORT_RISK_TARGET**: 1000
- **PROOF:ATTRIBUTION_OK**: True
- **PROOF:PORT_DET_HASH**: 388207293d9987a501ed1108b83de3f2f22b9e35

#### Portfolio Methods
- `vol_target`: Inverse volatility weighting scaled to target volatility
- `risk_parity`: Equal risk contribution through iterative optimization
- `hrp`: Hierarchical Risk Parity using correlation-based clustering

## Determinism Guarantees

All tools provide:
- `audit_hash`: SHA256 of inputs + outputs
- `inputs_hash`: Hash of input parameters
- `code_hash`: Hash of function code
- Sorted outputs for consistent ordering
- Fixed random seeds for synthetic data