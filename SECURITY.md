# Security Policy

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing **security@ally-project.dev**.

## What We Protect Against

### Input Validation
- All tool inputs validated via Pydantic schemas
- Path traversal prevention in file operations
- Command injection prevention in subprocess calls

### Credential Management
- **No hardcoded secrets** in source code
- API keys via environment variables only
- Git pre-commit hooks to prevent secret commits

### Audit & Logging
- All tool executions logged with deterministic hashes
- No sensitive data in log outputs
- Audit trails for reproducibility and debugging

## API Key Security

### Optional Environment Variables
```bash
# Optional integrations (not required for core functionality)
export LLAMAPARSE_API_KEY="your_key_here"     # SemTools/PDF parsing
export ALPHA_VANTAGE_API_KEY="your_key_here"  # Financial data
export POLYGON_API_KEY="your_key_here"        # Market data
```

### Best Practices
1. **Never commit API keys** to version control
2. **Use environment variables** or secure vaults  
3. **Rotate keys regularly**
4. **Monitor API usage** for anomalies

### Testing Security
All tests work offline without API keys:
```bash
# Tests pass without any environment variables
pytest tests/
```

## Contact

- **Security Email**: security@ally-project.dev
- **Response Time**: 48 hours maximum