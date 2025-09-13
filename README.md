# 🤖 Ally - Local-First CLI Agent System

[![CI](https://github.com/gagan114662/Ally/actions/workflows/ci.yml/badge.svg)](https://github.com/gagan114662/Ally/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Local-first CLI agent system with deterministic tools, audit trails, and modular architecture.**

Ally is a Python CLI framework for building AI agents that work with structured data, web APIs, computer vision, and quantitative analysis - all while maintaining reproducibility and audit trails.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/gagan114662/Ally.git
cd Ally
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# Run tests
pytest -q

# Try CV pattern detection
ally run cv.detect_chart_patterns --symbol AAPL --lookback 100
ally run cv.generate_synthetic --pattern engulfing_bull --rows 50

# List all available tools
ally list
```

## 🛠️ Core Tools

### Computer Vision (Milestone 4) ✅
```bash
# Detect chart patterns with visual + numeric confirmation
ally run cv.detect_chart_patterns --symbol TSLA --lookback 100

# Generate synthetic test data
ally run cv.generate_synthetic --pattern pin_bar_bull --rows 20
```

### Model Router (offline & deterministic)
```bash
# Task-aware model selection using offline fixtures
ally run router.build_matrix
# Prints PROOF lines and returns the chosen engine per task using fixtures only.
```

### Web & Data Tools
```bash
# Fetch and parse web content
ally run web.fetch --url "https://example.com" --prompt "Extract key metrics"

# Load OHLCV data
ally run data.load_ohlcv --symbol AAPL --period 1d --lookback 100
```

## 🧪 Testing & Verification

All tools include deterministic behavior with reproducible hashes:

```bash
# Run full test suite
pytest -q

# Run verification pack (if present)
python -m ally.verify.verify_claims

# Check specific tool with verification
ally run cv.detect_chart_patterns --verify --symbol TEST_DATA
```

## 📊 Example: CV Pattern Detection

```python
from ally.tools.cv import cv_detect_chart_patterns
from ally.schemas.cv import CVDetectIn

# Detect bullish engulfing pattern
result = cv_detect_chart_patterns(CVDetectIn(
    symbol="AAPL", 
    lookback=100,
    patterns=["engulfing_bull"]
))

print(f"Found {len(result.detections)} patterns")
# Outputs: base64 chart + numeric confirmations + audit hash
```

## 🔐 Security

- **No secrets in git**: All API keys via environment variables
- **Input validation**: Pydantic schemas prevent injection  
- **Audit trails**: Every operation logged with deterministic hashes
- **Network isolation**: Tests work offline with synthetic data

## 📁 Project Structure

```
ally/
├── cli/           # Typer CLI interface
├── schemas/       # Pydantic input/output models  
├── tools/         # Core tool implementations
├── utils/         # Shared utilities (audit, plotting, etc.)
└── verify/        # Verification pack for claims

tests/             # Test suite with synthetic fixtures
data/fixtures/     # Deterministic test data
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing guidelines, and code style.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

**Built for quantitative researchers, algorithmic traders, and AI engineers who need reliable, auditable tools.**