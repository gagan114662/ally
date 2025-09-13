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

## M9 Orchestrator Runtime Integration

**Branch:** `feature/m9-orchestrator-integration`
**Status:** COMPLETE ✅

```
PROOF:ORCH_RUNTIME_INTEGRATION: true
PROOF:ORCH_INTEGRATION_HASH: 41b1ce9add17adb7e96099a07cd756abb59a94a2
PROOF:ORCH_EXPERIMENT_ID: m9-proof-bundle
PROOF:ORCH_RUNTIME_MODE: fixtures
PROOF:ORCH_TASK_COUNT: 4
PROOF:ORCH_WIREUP_COMPLETE: true
```

### Integration Features
- **Orchestrator Demo Tool**: `orchestrator.demo` with runtime integration flags
- **CodeGen Unit Integration**: Added `use_runtime` and `runtime_live` parameters to orchestration workflow
- **Runtime Cache Integration**: Demonstrates cache hit/miss patterns with deterministic keys
- **Task Coverage**: Supports codegen, nlp, math, cv task types
- **Fixture-First**: CI uses fixtures, local development supports optional Ollama
- **M9 Tests**: Comprehensive test suite with `@pytest.mark.m9` marker

### Usage
```python
# Orchestrator demo (fixture mode)
result = TOOL_REGISTRY["orchestrator.demo"](
    experiment_id="test", 
    use_runtime=True, 
    runtime_live=False
)

# Code generation with runtime integration
unit.run(
    prompt="Build a web app", 
    use_runtime=True, 
    runtime_live=False
)
```

## M-RealData Gate — Live, Receipt-backed, Anti-fabrication Data

**Branch:** `feature/m-realdata-gate`
**Status:** COMPLETE ✅

M-RealData Gate provides receipt-backed, anti-fabrication market data with double-gate security and cross-provider quorum verification.

### Core Security Features
- **Double Gate**: Live data requires both `live=True` AND `ALLY_LIVE=1` environment variable
- **Receipt Attestation**: Every fetch generates SHA1-keyed receipts stored in `runs/receipts/` and DuckDB
- **Raw Payload Storage**: Complete responses saved in `runs/raw/<vendor>/` for forensics
- **Budget Guards**: Per-session cost tracking prevents API quota blowouts
- **Quorum Verification**: Optional cross-provider agreement checks within tolerance

### Dry Mode Proofs (CI Environment)
```
PROOF:REALDATA_MODE: "dry"
PROOF:RECEIPTS_N: 0
PROOF:QUORUM_OK: "n/a"
PROOF:COST_CENTS: 0
PROOF:LIVE_GATE: "working"
```

### Live Mode Proofs (Local with ALLY_LIVE=1)
```
PROOF:REALDATA_MODE: "live"  
PROOF:RECEIPTS_N: 2
PROOF:QUORUM_OK: true
PROOF:COST_CENTS: 3
PROOF:RECEIPT_SHA1: "e2d...9a1"
PROOF:SAMPLE_VENDOR: "polygon"
```

### Allow-listed Providers
- **Free Tier**: Polygon, AlphaVantage, Finnhub, FRED, Reddit, GitHub, Binance, Coinbase
- **Paid Tier**: Quandl, Tavily, Valyu  
- **Rate Limited**: All providers have per-minute limits and cost tracking
- **Authenticated**: API keys from environment variables (never stored in receipts)

### Tools Available
```python
# Single provider fetch with receipt
result = TOOL_REGISTRY["data.live_fetch"](
    vendor="polygon",
    endpoint="/v2/aggs/ticker/AAPL/range/1/day/2025-01-01/2025-01-15",
    params={},
    live=True,
    budget_cents=50
)

# Historical data with quorum verification
result = TOOL_REGISTRY["data.live_history"](
    symbol="AAPL",
    start="2025-01-01T00:00:00Z", 
    end="2025-01-15T00:00:00Z",
    vendor="polygon",
    live=True,
    quorum={"vendors":["polygon","alphavantage"],"metric":"close","tolerance_bps":5}
)

# Orchestrator with live data integration
result = TOOL_REGISTRY["orchestrator.run"](
    experiment_id="live-demo",
    symbols=["AAPL","MSFT"],
    use_live_data=True,
    live_budget_cents=100,
    live_quorum={"vendors":["polygon","alphavantage"],"metric":"close","tolerance_bps":5}
)
```

### Anti-BS Guarantees
1. **No Network in CI**: ALLY_LIVE!=1 blocks all provider access, returns `{"live_denied": true}`
2. **Receipt Requirements**: Any result claiming live data must reference receipt SHA1s  
3. **Quorum Enforcement**: Cross-provider disagreement beyond tolerance fails with clear error
4. **Budget Enforcement**: Session cost tracking prevents quota overruns
5. **Audit Trail**: Complete request/response chain stored with timestamps

### DuckDB Receipt Schema
```sql
CREATE TABLE data_receipts (
    content_sha1 TEXT PRIMARY KEY,
    vendor TEXT NOT NULL,
    endpoint TEXT NOT NULL, 
    params_json TEXT NOT NULL,
    ts_iso TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    cost_cents INTEGER,
    session_id TEXT
);
```

## Determinism Guarantees

All tools provide:
- `audit_hash`: SHA256 of inputs + outputs
- `inputs_hash`: Hash of input parameters
- `code_hash`: Hash of function code
- Sorted outputs for consistent ordering
- Fixed random seeds for synthetic data