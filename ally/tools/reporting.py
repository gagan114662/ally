"""
Reporting tools for Ally - generate tearsheet reports from experimental runs
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..tools import register
from ..schemas.base import ToolResult
from ..schemas.report import GenerateTearsheetIn, ReportSummary
from ..utils.db import get_db_manager


@register("reporting.generate_tearsheet")
def reporting_generate_tearsheet(**kwargs) -> ToolResult:
    """
    Generate an HTML tearsheet report for an experimental run
    
    Creates a comprehensive HTML report with metrics, trades, events, and analysis
    for a specific run stored in the memory system.
    """
    try:
        # Parse inputs - handle both schema format and direct run_id
        if isinstance(kwargs.get('run_id'), str) and len(kwargs) == 1:
            # Simple case: just run_id provided
            inputs = GenerateTearsheetIn(run_id=kwargs['run_id'])
        else:
            inputs = GenerateTearsheetIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        # Get run data from database
        db = get_db_manager()
        run_data = db.get_run(inputs.run_id)
        
        if not run_data:
            return ToolResult.error([f"Run {inputs.run_id} not found in memory"])
        
        # Generate HTML report
        html_content = _generate_html_report(run_data, inputs)
        
        # Determine output path - generate both absolute and relative paths
        if inputs.output_path:
            html_abs_path = inputs.output_path
            # Try to make it relative to project root if possible
            project_root = Path(__file__).parent.parent.parent
            try:
                html_rel_path = str(Path(html_abs_path).relative_to(project_root))
            except ValueError:
                html_rel_path = html_abs_path  # Use absolute if can't relativize
        else:
            # Default to reports directory with relative path
            project_root = Path(__file__).parent.parent.parent
            reports_dir = project_root / "reports"
            reports_dir.mkdir(exist_ok=True)
            html_rel_path = f"reports/{inputs.run_id}_tearsheet.html"
            html_abs_path = str(reports_dir / f"{inputs.run_id}_tearsheet.html")
        
        # Write HTML file using absolute path
        with open(html_abs_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Get file size
        file_size = os.path.getsize(html_abs_path)
        
        # Extract summary data, handling None values
        metrics = run_data.get('metrics', {}) or {}
        events = run_data.get('events', []) or []
        trades = run_data.get('trades', []) or []
        
        # Determine sections included
        sections = ["overview", "metrics"]
        if inputs.include_trades and trades:
            sections.append("trades")
        if inputs.include_events and events:
            sections.append("events")
        if inputs.include_metrics and metrics:
            sections.append("performance")
        
        # Create report summary with relative path for portability
        summary = ReportSummary(
            run_id=inputs.run_id,
            html_path=html_rel_path,  # Store relative path for portability
            kpis=metrics,
            n_trades=len(trades),
            n_events=len(events),
            sections=sections,
            generated_at=datetime.utcnow(),
            file_size_bytes=file_size
        )
        
        return ToolResult.success(summary.model_dump())
        
    except Exception as e:
        return ToolResult.error([f"Report generation failed: {e}"])


def _generate_html_report(run_data: Dict[str, Any], config: GenerateTearsheetIn) -> str:
    """Generate HTML content for tearsheet report"""
    
    run_id = run_data['run_id']
    task = run_data.get('task', 'Unknown')
    timestamp = run_data.get('ts', datetime.utcnow().isoformat())
    metrics = run_data.get('metrics', {}) or {}
    events = run_data.get('events', []) or []
    trades = run_data.get('trades', []) or []
    notes = run_data.get('notes', '') or ''
    
    # Start HTML document
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tearsheet Report - {run_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .section {{
            background: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .table th, .table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .table th {{
            background-color: #f1f1f1;
            font-weight: bold;
        }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        h1, h2 {{ margin-top: 0; }}
        .timestamp {{
            opacity: 0.7;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Experimental Run Tearsheet</h1>
        <h2>{run_id}</h2>
        <div class="timestamp">Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
    </div>
    
    <div class="section">
        <h2>Run Overview</h2>
        <p><strong>Task:</strong> {task}</p>
        <p><strong>Execution Time:</strong> {timestamp}</p>
        <p><strong>Code Hash:</strong> {run_data.get('code_hash', 'N/A')}</p>
        <p><strong>Inputs Hash:</strong> {run_data.get('inputs_hash', 'N/A')}</p>
        {f'<p><strong>Notes:</strong> {notes}</p>' if notes else ''}
    </div>
"""
    
    # Add metrics section
    if config.include_metrics and metrics:
        html += """
    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metrics-grid">
"""
        
        for key, value in metrics.items():
            # Format value based on type and key name
            if isinstance(value, float):
                if 'return' in key.lower() or 'ratio' in key.lower():
                    formatted_value = f"{value:.3f}"
                    css_class = "positive" if value > 0 else "negative"
                else:
                    formatted_value = f"{value:.6f}"
                    css_class = ""
            else:
                formatted_value = str(value)
                css_class = ""
            
            html += f"""
            <div class="metric-card">
                <div class="metric-label">{key.replace('_', ' ').title()}</div>
                <div class="metric-value {css_class}">{formatted_value}</div>
            </div>
"""
        
        html += """
        </div>
    </div>
"""
    
    # Add trades section
    if config.include_trades and trades:
        html += """
    <div class="section">
        <h2>Trades Executed</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Quantity</th>
                    <th>Price</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for trade in trades:
            side_class = "positive" if trade.get('side') == 'buy' else "negative"
            html += f"""
                <tr>
                    <td>{trade.get('symbol', 'N/A')}</td>
                    <td class="{side_class}">{trade.get('side', 'N/A').upper()}</td>
                    <td>{trade.get('qty', 0)}</td>
                    <td>${trade.get('price', 0):,.2f}</td>
                    <td>{trade.get('ts', 'N/A')}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
"""
    
    # Add events section  
    if config.include_events and events:
        html += """
    <div class="section">
        <h2>Events Timeline</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Event Type</th>
                    <th>Payload</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for event in events:
            payload_str = json.dumps(event.get('payload', {}), indent=2) if event.get('payload') else 'N/A'
            html += f"""
                <tr>
                    <td>{event.get('type', 'N/A')}</td>
                    <td><pre style="margin:0; font-size:11px;">{payload_str}</pre></td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
"""
    
    # Close HTML document
    html += """
    <div class="section">
        <h2>Report Metadata</h2>
        <p><strong>Generated by:</strong> Ally Reporting System</p>
        <p><strong>Report Type:</strong> Experimental Run Tearsheet</p>
        <p><strong>Data Source:</strong> Ally Memory Database</p>
    </div>
</body>
</html>
"""
    
    return html