#!/usr/bin/env python3
"""
Phase contract tests runner
Validates that all core modules have proper contracts and imports
"""

import sys
import importlib
import traceback
from pathlib import Path


def test_core_imports():
    """Test that core modules can be imported without errors"""
    modules = [
        "ally.tools",
        "ally.schemas.base",
        "ally.utils.audit",
        "ally.utils.hashing",
        "ally.utils.serialization",
        "ally.tools.memory",
        "ally.tools.reporting",
        "ally.tools.router",
        "ally.tools.runtime",
        "ally.tools.cv",
        "ally.tools.data",
        "ally.tools.nlp",
        "ally.tools.web",
        "ally.tools.bt",
        "ally.tools.risk",
    ]

    failed = []
    passed = []

    for module in modules:
        try:
            importlib.import_module(module)
            passed.append(module)
            print(f"✅ {module}")
        except Exception as e:
            failed.append((module, str(e)))
            print(f"❌ {module}: {e}")

    print(f"\n📊 Import Summary: {len(passed)} passed, {len(failed)} failed")
    return len(failed) == 0


def test_tool_registry():
    """Test that tool registry loads and has expected tools"""
    try:
        from ally.tools import TOOL_REGISTRY

        expected_tools = [
            "memory.log_run",
            "memory.query",
            "reporting.generate_tearsheet",
            "cache.clear",
            "runtime.generate",
            "cv.detect_chart_patterns",
            "data.load_ohlcv",
        ]

        missing = []
        for tool in expected_tools:
            if tool not in TOOL_REGISTRY:
                missing.append(tool)
            else:
                print(f"✅ Tool: {tool}")

        if missing:
            print(f"❌ Missing tools: {missing}")
            return False

        print(f"✅ Tool registry loaded with {len(TOOL_REGISTRY)} tools")
        return True

    except Exception as e:
        print(f"❌ Tool registry error: {e}")
        return False


def test_schema_validation():
    """Test that core schemas can be imported and validated"""
    try:
        from ally.schemas.base import ToolResult, Meta, ToolStatus
        from ally.schemas.memory import LogRunIn, QueryIn
        from ally.schemas.report import ReportSummary

        # Test basic schema instantiation
        meta = Meta()
        result = ToolResult(ok=True, data={}, errors=[], meta=meta)

        print("✅ Schema validation passed")
        return True

    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False


def main():
    """Run all phase contract tests"""
    print("🔍 Running Phase Contract Tests")
    print("=" * 50)

    tests = [
        ("Core Imports", test_core_imports),
        ("Tool Registry", test_tool_registry),
        ("Schema Validation", test_schema_validation),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n📋 {name}")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
                print(f"✅ {name} PASSED")
            else:
                print(f"❌ {name} FAILED")
        except Exception as e:
            print(f"❌ {name} ERROR: {e}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"📊 Phase Contract Summary: {passed}/{total} passed")

    if passed == total:
        print("🎯 All phase contracts passed!")
        return 0
    else:
        print("⚠️ Some phase contracts failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())