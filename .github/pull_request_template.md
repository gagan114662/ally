## PR Verification Checklist

### Required (Must be ✅ green for merge)
- [ ] **Smoke Tests**: [smoke-test-results artifact link]
- [ ] **Static Analysis**: [static-analysis-results artifact link]
- [ ] **Unit Tests**: [unit-test-results artifact link]

### Advisory (Quality gates - can be ❌ initially)
- [ ] **Phase Contract**: [contract-test-results artifact link]
- [ ] **Integration Tests**: [integration-test-results artifact link]
- [ ] **Determinism**: [determinism-test-results artifact link]

### Changes
<!-- Describe what this PR changes -->

### Testing
<!-- How was this change tested? -->

---
*✨ Green checks = merge ready. Red advisory jobs = improvement opportunities.*