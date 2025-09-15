# 🔧 Orchestrator Debug Guide

**Surgical fixes for autonomous algo orchestrator pipeline failures**

Based on error analysis from `orchestrator_20250912_111233.log` showing:
- QuantConnect config errors in dry-run mode
- "Unknown template" variant generation failures
- Missing receipts and tracebacks for debugging

## 🚨 Root Cause Analysis

From your log analysis, the pipeline fails at multiple points:

```
ERROR Failed to initialize QuantConnect: 'QuantConnectConfig' object has no attribute 'get'
ERROR Error generating variants for <...MomentumTemplate object...>: Unknown template
ERROR Error generating variants for <...MeanReversionTemplate object...>: Unknown template
Research Loop completed — 0 ranked hypotheses
ERROR Orchestration failed: No hypotheses generated
```

**Issues:**
1. **QuantConnect bootstrap** touches config even in dry mode
2. **Variant generator** doesn't recognize template classes
3. **Missing tracebacks** make debugging impossible
4. **No receipt trail** for CI verification

## ✅ Surgical Fixes Applied

### Fix #1: QuantConnect Dry-Run Guard
```python
from ally.research.orchestrator_fixes import QuantConnectDryRunGuard

with QuantConnectDryRunGuard(dry_run=True):
    # Any QuantConnect code bypassed in dry mode
    config = get_quantconnect_config()  # Returns simulator config
```

### Fix #2: Receipt+Traceback Wrapper
```python
from ally.research.orchestrator_fixes import VariantGenerationWrapper

wrapper = VariantGenerationWrapper()
variants = wrapper.safe_expand(template, generator)
# Captures full tracebacks + emits receipts for CI verification
```

### Fix #3: Offline Fixture Validation
```python
from ally.research.orchestrator_fixes import OfflineFixtureValidator

validator = OfflineFixtureValidator()
validator.assert_fixtures_present()  # Fails fast if missing fixtures
```

### Fix #4: Template Schema Validation
```python
from ally.research.orchestrator_fixes import TemplateSchemaValidator

validator = TemplateSchemaValidator()
errors = validator.validate_template(template)  # Clear field-level errors
```

### Fix #5: Stoplight Summary Receipt
```python
from ally.research.orchestrator_fixes import OrchestrationStoplightReceipt

stoplight = OrchestrationStoplightReceipt()
stoplight.add_stage("template_discovery", success=True, count=5)
result = stoplight.emit_final_receipt()
# Emits: templates→variants→scored→survivors with full diagnostics
```

## 🚀 Quick Integration

### Drop-in Patch (Ready to Commit)
```python
from ally.research.orchestrator_fixes import create_orchestrator_patch

def fixed_orchestration_main():
    """Your existing orchestration main with surgical fixes applied"""
    logger.info("🚀 Starting Autonomous Algo Strategy Developer (FIXED)")

    # Apply all fixes
    patch = create_orchestrator_patch()
    patch.apply_all_fixes()

    try:
        # Your existing template discovery
        templates = discover_hypothesis_templates()

        # Safe template expansion with all checks
        variants = patch.safe_template_expand(templates, variant_generator)

        # Your existing scoring (now uses offline fixtures)
        scored = score_hypotheses_offline(variants)
        patch.stoplight.add_stage("hypothesis_scoring", True, len(scored))

        # Your existing selection
        survivors = select_top_hypotheses(scored, budget=3)
        patch.stoplight.add_stage("selection_ranking", True, len(survivors))

    except Exception as e:
        logger.exception("Orchestration failed")
        patch.stoplight.add_error("orchestration", str(e))

    finally:
        # Always emit diagnostics
        result = patch.finalize()
        return result
```

## 🧪 Verification Checklist

Run locally with max logging to verify fixes:

```bash
# Ensure dry mode
export ALLY_LIVE=0

# Run with debug logging
python -m ally.research.orchestrator --dry --log-level DEBUG \
  --out artifacts/autonomous_algo_results/

# Check these artifacts appear:
ls -la artifacts/autonomous_algo_results/logs/  # Full tracebacks
ls -la artifacts/receipts.jsonl              # Receipt entries for:
#   - research.template.validate
#   - research.variant.expand
#   - research.fixture.validation
#   - research.orchestration.summary
```

**Expected Results:**
- ❌ **Before**: 5 templates → 0 variants → 0 hypotheses (all red)
- ✅ **After**: 5 templates → N variants → M scored → K survivors (green pipeline)

## 📊 Receipt Verification

Each stage now emits verifiable receipts:

```jsonl
{"tool":"research.template.validate","ok":true,"template":"MomentumTemplate","error_count":0}
{"tool":"research.variant.expand","ok":true,"template":"MomentumTemplate","variants_count":3}
{"tool":"research.fixture.validation","ok":true,"fixtures_validated":4}
{"tool":"research.orchestration.summary","ok":true,"pipeline_counts":{"templates":5,"variants":15,"scored":10,"survivors":3}}
```

## 🔍 Debugging Guide

### When Templates Still Fail Schema Validation
```bash
# Check template structure
python -c "
from your_hypothesis_lib import MomentumTemplate
template = MomentumTemplate()
print(f'Name: {getattr(template, \"name\", \"MISSING\")}')
print(f'Params: {getattr(template, \"parameters\", \"MISSING\")}')
"
```

### When Variant Generation Still Fails
```bash
# Check generator compatibility
python -c "
template = YourTemplate()
generator = YourGenerator()
print(f'Generator expand method: {hasattr(generator, \"expand\")}')
print(f'Template class: {template.__class__.__name__}')
"
```

### When Fixtures Missing
```bash
# Auto-create minimal fixtures
python -c "
from ally.research.orchestrator_fixes import OfflineFixtureValidator
validator = OfflineFixtureValidator()
created = validator.create_minimal_fixtures()
print(f'Created {len(created)} fixtures: {list(created.keys())}')
"
```

## 🎯 Success Metrics

**Pipeline Health Check:**
- ✅ No QuantConnect config errors in dry mode
- ✅ Templates pass schema validation
- ✅ Variants generated > 0
- ✅ Full tracebacks in logs for any failures
- ✅ Receipt trail for CI verification
- ✅ Stoplight summary: `5→15→10→3` (example counts)

## 🔧 Running Tests

```bash
# Test the fixes
pytest tests/test_orchestrator_fixes.py -v

# Test integration
python -m ally.research.orchestrator_fixes  # Runs example
```

## 📋 Commit Checklist

- [ ] `ally/research/orchestrator_fixes.py` - All surgical fixes
- [ ] `tests/test_orchestrator_fixes.py` - Comprehensive test coverage
- [ ] Integration in your orchestrator main function
- [ ] Verification run with `ALLY_LIVE=0` shows green pipeline
- [ ] Receipt trail in artifacts for CI verification

---

**These fixes convert your "all red" research loop into a debuggable pipeline where each sub-tool failure is obvious with full stack traces and receipt hashes for CI verification.**