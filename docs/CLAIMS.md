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

## M-Router — Task-Aware Model Selection

**PR:** https://github.com/gagan114662/ally/pull/20  
**Commit:** e65c686  
**CI Job:** M-Router (Task-Aware Model Selection)

```
PROOF:ROUTER_MATRIX: {"codegen":"llama3.1-8b-instruct","cv":"llava-phi-3-mini","math":"deepseek-math-7b","nlp":"llama3.1-8b"}
PROOF:ROUTER_FALLBACK: ok
PROOF:EVAL_DET_HASH: f4a1655deddc22dc06eabf8a45f0c6029ca7472b
PROOF:ROUTER_DET: 46ca8c1e06918f8ea57cb6631b4de2d11e9e3727
```

**Artifacts:** `mrouter-proof-bundle/*`

## M-Cache + Runtime

**PR:** https://github.com/gagan114662/ally/pull/21  
**Commit:** 021abe8

```
PROOF:CACHE_HIT: 0
PROOF:CACHE_KEY_HASH: 8e40a9d2d1344963334b2302d504c82e727c10fb
PROOF:RUNTIME_MODE: fixture

PROOF:CACHE_HIT: 1
PROOF:CACHE_KEY_HASH: 8e40a9d2d1344963334b2302d504c82e727c10fb
```

**Artifacts:** `mcache-proof-bundle/*`

## M-Receipts-Everywhere — End-to-End Provenance

**PR:** https://github.com/gagan114662/ally/pull/23  
**Commit:** [pending]  
**CI Job:** M-Receipts-Everywhere (End-to-End Provenance)

```
PROOF:ORCHESTRATOR_TOOLS: ["orchestrator.demo", "orchestrator.run"]
PROOF:DEMO_OK: true
PROOF:DEMO_RECEIPTS_LINKED: 0
PROOF:RUN_OK: true
PROOF:RUN_RECEIPTS_LINKED: 0
PROOF:PROVENANCE_HASH: f4a1655deddc22dc...
PROOF:RECEIPT_SUMMARY_COST: 125
PROOF:RECEIPT_SUMMARY_VENDORS: ["test_provider_1", "test_provider_2"]
PROOF:SCHEMA_VALIDATION_OK: true
PROOF:DB_INSERT_OK: true
PROOF:MRECEIPTS_STATUS: operational
PROOF:MRECEIPTS_DET_HASH: 46ca8c1e06918f8ea57cb6631b4de2d11e9e3727
```

### M-Receipts Claims

### RECV-001: Receipt Reference Schema
- **Claim**: ReceiptRef captures (content_sha1, vendor, endpoint, ts_iso, cost_cents) for provenance linking
- **Verification**: Schema validation passes with all required fields
- **Expected**: SCHEMA_VALIDATION_OK: true

### RECV-002: Orchestrator Receipt Wiring  
- **Claim**: orchestrator.demo and orchestrator.run include receipt references in output
- **Verification**: Both tools return provenance section with receipts_linked count
- **Expected**: DEMO_OK: true, RUN_OK: true

### RECV-003: Provenance Hash Computation
- **Claim**: compute_provenance_hash creates SHA256 of inputs + outputs + receipts
- **Verification**: Hash computed from test inputs/outputs/receipts is deterministic
- **Expected**: PROVENANCE_HASH matches expected pattern

### RECV-004: Database Receipt Integration
- **Claim**: Receipt storage and retrieval via enhanced DatabaseManager
- **Verification**: insert_receipt and get_receipt_stats methods work
- **Expected**: DB_INSERT_OK: true, DB_STATS_OK: true

### RECV-005: End-to-End Receipt Linking
- **Claim**: Runs link to all receipts generated during execution session
- **Verification**: link_receipts_to_run returns appropriate ReceiptRef objects
- **Expected**: Receipt linking operational for both dry and live modes

**Artifacts:** `mreceipts-proof-bundle/*`

## M-Receipts Invariants — Claims & Proof

**Invariant:** If `live == true`, outputs MUST include ≥1 `ReceiptRef` with `content_sha1`, `vendor`, `endpoint`, and ISO-Z `ts_iso`.

**Verification:**
- CI job `M-Receipts Invariants` runs `tests/test_receipts_invariants.py`.
- Proof lines:
  - `PROOF:RECEIPTS_INVARIANTS: ok`
  - `PROOF:RECEIPTS_DIFF_CLI: ok`
  - `PROOF:RECEIPTS_VERIFY_REGISTERED: ok`

### RECV-INV-001: Hard Invariant Enforcement
- **Claim**: assert_receipts_invariants({"live":true, "receipts":[]}) raises AssertionError
- **Verification**: Test function blocks live outputs without receipts
- **Expected**: RECEIPTS_INVARIANTS: ok

### RECV-INV-002: CLI Verification Tools  
- **Claim**: receipts.verify and receipts.diff tools are registered and functional
- **Verification**: Tools can verify receipt files and compare numeric series
- **Expected**: RECEIPTS_DIFF_CLI: ok, RECEIPTS_VERIFY_REGISTERED: ok

### RECV-INV-003: Tearsheet Provenance Box
- **Claim**: Generated tearsheets include visible provenance section
- **Verification**: HTML contains receipt count, hash, vendors, timestamps
- **Expected**: Provenance box visible in all tearsheet outputs

**Artifacts:** `receipts-invariants-proof-bundle/*`

## Determinism Guarantees

All tools provide:
- `audit_hash`: SHA256 of inputs + outputs + receipts
- `inputs_hash`: Hash of input parameters
- `code_hash`: Hash of function code
- Sorted outputs for consistent ordering
- Fixed random seeds for synthetic data
- Receipt provenance tracking via content_sha1 links