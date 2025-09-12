"""
QuantConnect algorithm linter with AST rules and autofixes
"""

from __future__ import annotations
import ast
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ally.schemas.base import ToolResult, Meta


class QCLintRule:
    """Base class for QC linting rules"""
    
    def __init__(self, rule_id: str, description: str, autofix: bool = False):
        self.rule_id = rule_id
        self.description = description
        self.autofix = autofix
        self.violations = []
    
    def check(self, node: ast.AST, source: str, file_path: str) -> List[Dict[str, Any]]:
        """Check AST node for violations"""
        self.violations = []
        self.visit(node, source, file_path)
        return self.violations
    
    def visit(self, node: ast.AST, source: str, file_path: str) -> None:
        """Visit AST node - override in subclasses"""
        pass
    
    def fix(self, source: str, violations: List[Dict[str, Any]]) -> str:
        """Apply autofixes to source code"""
        return source


class ImportRule(QCLintRule):
    """Ensure 'from AlgorithmImports import *' is present"""
    
    def __init__(self):
        super().__init__("QC001", "Missing AlgorithmImports import", autofix=True)
    
    def visit(self, node: ast.AST, source: str, file_path: str) -> None:
        if isinstance(node, ast.Module):
            has_algorithm_imports = False
            for child in ast.walk(node):
                if (isinstance(child, ast.ImportFrom) and 
                    child.module == "AlgorithmImports" and 
                    any(alias.name == "*" for alias in child.names)):
                    has_algorithm_imports = True
                    break
            
            if not has_algorithm_imports:
                self.violations.append({
                    "line": 1,
                    "column": 0,
                    "message": "Missing 'from AlgorithmImports import *'",
                    "rule_id": self.rule_id
                })
    
    def fix(self, source: str, violations: List[Dict[str, Any]]) -> str:
        if violations:
            lines = source.split('\n')
            # Find first non-comment, non-blank line
            insert_line = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    insert_line = i
                    break
            
            # Insert import at the beginning
            lines.insert(insert_line, "from AlgorithmImports import *")
            return '\n'.join(lines)
        return source


class ClassFileMatchRule(QCLintRule):
    """Ensure class name matches file name"""
    
    def __init__(self):
        super().__init__("QC002", "Class name must match file name", autofix=False)
    
    def visit(self, node: ast.AST, source: str, file_path: str) -> None:
        file_name = Path(file_path).stem
        
        for child in ast.walk(node):
            if isinstance(child, ast.ClassDef):
                if child.name != file_name:
                    self.violations.append({
                        "line": child.lineno,
                        "column": child.col_offset,
                        "message": f"Class '{child.name}' should be named '{file_name}' to match file",
                        "rule_id": self.rule_id,
                        "expected": file_name,
                        "actual": child.name
                    })


class RequiredMethodsRule(QCLintRule):
    """Ensure required QC methods are present in Initialize()"""
    
    def __init__(self):
        super().__init__("QC003", "Missing required method calls in Initialize", autofix=False)
    
    def visit(self, node: ast.AST, source: str, file_path: str) -> None:
        required_calls = ["SetStartDate", "SetCash", "SetBrokerageModel"]
        
        for child in ast.walk(node):
            if (isinstance(child, ast.FunctionDef) and 
                child.name == "Initialize"):
                
                found_calls = set()
                for stmt in ast.walk(child):
                    if (isinstance(stmt, ast.Call) and 
                        isinstance(stmt.func, ast.Attribute) and
                        isinstance(stmt.func.value, ast.Name) and
                        stmt.func.value.id == "self"):
                        found_calls.add(stmt.func.attr)
                
                for required in required_calls:
                    if required not in found_calls:
                        self.violations.append({
                            "line": child.lineno,
                            "column": child.col_offset,
                            "message": f"Missing required call: self.{required}()",
                            "rule_id": self.rule_id,
                            "missing_call": required
                        })


class OnDataSignatureRule(QCLintRule):
    """Ensure OnData has correct signature"""
    
    def __init__(self):
        super().__init__("QC004", "OnData must have signature (self, data: Slice)", autofix=True)
    
    def visit(self, node: ast.AST, source: str, file_path: str) -> None:
        for child in ast.walk(node):
            if (isinstance(child, ast.FunctionDef) and 
                child.name == "OnData"):
                
                # Check signature
                args = child.args.args
                if (len(args) != 2 or 
                    args[0].arg != "self" or 
                    args[1].arg != "data" or
                    not (hasattr(args[1], 'annotation') and 
                         isinstance(args[1].annotation, ast.Name) and
                         args[1].annotation.id == "Slice")):
                    
                    self.violations.append({
                        "line": child.lineno,
                        "column": child.col_offset,
                        "message": "OnData signature should be: def OnData(self, data: Slice) -> None:",
                        "rule_id": self.rule_id
                    })
    
    def fix(self, source: str, violations: List[Dict[str, Any]]) -> str:
        if not violations:
            return source
            
        lines = source.split('\n')
        for violation in violations:
            line_idx = violation["line"] - 1
            line = lines[line_idx]
            # Replace the function definition line
            indent = len(line) - len(line.lstrip())
            lines[line_idx] = " " * indent + "def OnData(self, data: Slice) -> None:"
        
        return '\n'.join(lines)


class DatetimeNowRule(QCLintRule):
    """Replace datetime.now() with self.Time"""
    
    def __init__(self):
        super().__init__("QC005", "Use self.Time instead of datetime.now()", autofix=True)
    
    def visit(self, node: ast.AST, source: str, file_path: str) -> None:
        for child in ast.walk(node):
            if (isinstance(child, ast.Call) and
                isinstance(child.func, ast.Attribute) and
                child.func.attr in ["now", "utcnow"]):
                
                # Check if it's datetime.now() or datetime.utcnow()
                if (isinstance(child.func.value, ast.Name) and
                    child.func.value.id == "datetime"):
                    
                    self.violations.append({
                        "line": child.lineno,
                        "column": child.col_offset,
                        "message": f"Replace datetime.{child.func.attr}() with self.Time",
                        "rule_id": self.rule_id,
                        "method": child.func.attr
                    })
    
    def fix(self, source: str, violations: List[Dict[str, Any]]) -> str:
        if not violations:
            return source
            
        # Simple regex replacement for datetime.now() and datetime.utcnow()
        source = re.sub(r'datetime\.(now|utcnow)\(\)', 'self.Time', source)
        return source


class PandasInOnDataRule(QCLintRule):
    """Detect pandas usage in OnData method"""
    
    def __init__(self):
        super().__init__("QC006", "Avoid pandas operations in OnData for performance", autofix=False)
    
    def visit(self, node: ast.AST, source: str, file_path: str) -> None:
        in_ondata = False
        
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef) and child.name == "OnData":
                in_ondata = True
                # Look for pd. or pandas. usage within OnData
                for stmt in ast.walk(child):
                    if (isinstance(stmt, ast.Attribute) and
                        isinstance(stmt.value, ast.Name) and
                        stmt.value.id in ["pd", "pandas"]):
                        
                        self.violations.append({
                            "line": stmt.lineno,
                            "column": stmt.col_offset,
                            "message": "Avoid pandas operations in OnData - move to Initialize or scheduled methods",
                            "rule_id": self.rule_id
                        })


def qc_lint(file_path: str, autofix: bool = False) -> ToolResult:
    """
    Lint QuantConnect algorithm file
    
    Args:
        file_path: Path to Python file to lint
        autofix: Whether to apply automatic fixes
        
    Returns:
        ToolResult with lint results and optionally fixed code
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read source code
        with open(path, 'r') as f:
            source = f.read()
        
        # Parse AST
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            return ToolResult(
                ok=False,
                data={"error": f"Syntax error: {e}"},
                errors=[f"Syntax error at line {e.lineno}: {e.msg}"],
                meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.lint"})
            )
        
        # Apply lint rules
        rules = [
            ImportRule(),
            ClassFileMatchRule(),
            RequiredMethodsRule(),
            OnDataSignatureRule(),
            DatetimeNowRule(),
            PandasInOnDataRule()
        ]
        
        all_violations = []
        fixed_source = source
        
        for rule in rules:
            violations = rule.check(tree, source, str(path))
            if violations:
                all_violations.extend(violations)
                
                # Apply autofixes if enabled
                if autofix and rule.autofix:
                    fixed_source = rule.fix(fixed_source, violations)
        
        # Write fixed code back if autofix enabled and fixes were applied
        if autofix and fixed_source != source:
            with open(path, 'w') as f:
                f.write(fixed_source)
        
        # Categorize violations
        errors = [v for v in all_violations if v["rule_id"] in ["QC002", "QC003"]]
        warnings = [v for v in all_violations if v["rule_id"] not in ["QC002", "QC003"]]
        
        return ToolResult(
            ok=len(errors) == 0,
            data={
                "file_path": str(path),
                "total_violations": len(all_violations),
                "errors": errors,
                "warnings": warnings,
                "autofix_applied": autofix and fixed_source != source,
                "fixed_violations": len(all_violations) - len([v for r in rules for v in r.check(ast.parse(fixed_source), fixed_source, str(path)) if r.autofix]) if autofix else 0
            },
            errors=[f"Line {e['line']}: {e['message']}" for e in errors],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.lint"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.lint"})
        )