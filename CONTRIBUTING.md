# Contributing to Ally

Thank you for your interest in contributing to Ally!

## Development Setup

```bash
# Clone and setup
git clone https://github.com/gagan114662/Ally.git
cd Ally
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# Run tests
pytest -q
```

## Testing Guidelines

- Write tests for all new functionality
- Use synthetic/fixture data (no network calls)
- Include both positive and negative test cases
- Ensure deterministic behavior (same inputs → same outputs)

## Tool Development Pattern

1. **Create schema** in `ally/schemas/`
2. **Implement tool** with `@register` decorator
3. **Add comprehensive tests**
4. **Ensure audit hashing**

## Code Style

- Follow PEP 8 (enforced by ruff and black)
- Use type hints for all function signatures
- Write docstrings for all public functions/classes

## Commit Guidelines

```
feat(cv): add morning star pattern detection
fix(audit): resolve hash determinism issue  
docs(readme): update CLI examples
```

## Pull Request Process

1. Create feature branch
2. Add tests and documentation
3. Ensure all tests pass
4. Create PR with description and examples