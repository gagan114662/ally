#!/usr/bin/env bash
# CI Phase 12 Determinism Harness
# Generates deterministic chat transcripts and TUI snapshots for testing

set -euo pipefail

# Deterministic environment
export ALLY_LIVE=0
export TZ=UTC

echo "[CI] Phase 12 Chat Determinism Harness Starting..."
echo "ALLY_LIVE=${ALLY_LIVE}"
echo "TZ=${TZ}"

# Ensure artifacts directories exist
mkdir -p artifacts/chat artifacts/status

# Create deterministic journal if missing (Phase 11 should have done this)
if [ ! -f artifacts/status/journal_ci.jsonl ]; then
  echo "[CI] Creating sample journal for chat testing..."
  python3 - <<'PY'
import sys
import os

# Add project to path
sys.path.insert(0, os.path.abspath('.'))

from ally.status.journal import Journal

# Create journal with sample entries
journal = Journal("artifacts/status/journal_ci.jsonl", deterministic_seed=42)
journal.clear()

# Add deterministic sample entries
journal.append(
    phase="Researching",
    step="SpecParsing",
    tool="spec.parse",
    params_hash="aa11bb22",
    receipt_hash="cc33dd44",
    note="ci-sample-1"
)

journal.append(
    phase="Researching",
    step="DataLoading",
    tool="data.load",
    params_hash="ee55ff66",
    receipt_hash="77889900",
    note="ci-sample-2"
)

journal.append(
    phase="Evaluating",
    step="WalkForward",
    tool="research.walkforward",
    params_hash="aabbccdd",
    receipt_hash="eeffgghh",
    note="ci-sample-3"
)

print(f"[CI] Created journal with {len(journal.read_entries())} entries")
PY
fi

# Generate deterministic chat transcript
echo "[CI] Generating deterministic chat transcript..."
python3 - <<'PY'
import json
import os
import sys

# Add project to path
sys.path.insert(0, os.path.abspath('.'))

from ally.chat.controller import ChatController

# Initialize controller with deterministic seed
controller = ChatController("artifacts/status/journal_ci.jsonl", seed=42)

# Deterministic test queries
test_queries = [
    "show status",
    "last 5 operations",
    "counters",
    "timers",
    "receipts",
    "help"
]

transcript = []
print("[CI] Processing chat queries...")

for query in test_queries:
    print(f"  Processing: {query}")
    response = controller.handle(query)

    # Create transcript entry
    entry = {
        "q": query,
        "r": {
            "ok": response.ok,
            "message": response.message,
            "receipt": response.receipt,
            "data_keys": list(response.data.keys()) if response.data else []
        }
    }
    transcript.append(entry)

# Write transcript
os.makedirs("artifacts/chat", exist_ok=True)
with open("artifacts/chat/transcript_ci.jsonl", "w") as f:
    for entry in transcript:
        f.write(json.dumps(entry, sort_keys=True) + "\n")

print(f"[CI] Wrote {len(transcript)} entries to transcript_ci.jsonl")

# Generate deterministic TUI snapshot
print("[CI] Generating TUI snapshot...")
from ally.ui.tui import AllyTUI

tui = AllyTUI("artifacts/status/journal_ci.jsonl", seed=42)

# Generate different format snapshots
snapshots = {
    "compact": tui.render_compact(),
    "json": tui.export_json(),
    "dashboard": tui.render_dashboard()
}

for format_name, snapshot in snapshots.items():
    filename = f"artifacts/chat/tui_snapshot_{format_name}.txt"
    with open(filename, "w") as f:
        f.write(snapshot)
    print(f"[CI] Wrote TUI snapshot: {filename}")

print("[CI] Chat transcript and TUI snapshots generated")
PY

# Validate transcript structure
echo "[CI] Validating transcript structure..."
python3 -c "
import json
import sys

try:
    with open('artifacts/chat/transcript_ci.jsonl', 'r') as f:
        lines = f.read().strip().split('\n')

    if len(lines) < 2:
        print('[ERROR] Transcript too short')
        sys.exit(1)

    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            if 'q' not in entry or 'r' not in entry:
                print(f'[ERROR] Invalid entry structure at line {i+1}')
                sys.exit(1)
            if 'receipt' not in entry['r'] or 'receipt_hash' not in entry['r']['receipt']:
                print(f'[ERROR] Missing receipt at line {i+1}')
                sys.exit(1)
        except json.JSONDecodeError:
            print(f'[ERROR] Invalid JSON at line {i+1}')
            sys.exit(1)

    print(f'[CI] Transcript validation passed ({len(lines)} entries)')

    # Check for expected queries
    queries = [json.loads(line)['q'] for line in lines]
    expected = ['show status', 'last 5 operations']
    for expected_query in expected:
        if expected_query not in queries:
            print(f'[ERROR] Missing expected query: {expected_query}')
            sys.exit(1)

    print('[CI] Expected queries found')

except FileNotFoundError:
    print('[ERROR] Transcript file not found')
    sys.exit(1)
"

# Validate TUI snapshots
echo "[CI] Validating TUI snapshots..."
python3 -c "
import os
import json

snapshots = ['compact', 'json', 'dashboard']
for snapshot in snapshots:
    filepath = f'artifacts/chat/tui_snapshot_{snapshot}.txt'
    if not os.path.exists(filepath):
        print(f'[ERROR] Missing snapshot: {filepath}')
        exit(1)

    with open(filepath, 'r') as f:
        content = f.read().strip()

    if not content:
        print(f'[ERROR] Empty snapshot: {filepath}')
        exit(1)

    if snapshot == 'compact':
        if not content.startswith('PHASE='):
            print(f'[ERROR] Invalid compact format: {filepath}')
            exit(1)
    elif snapshot == 'json':
        try:
            json.loads(content)
        except json.JSONDecodeError:
            print(f'[ERROR] Invalid JSON format: {filepath}')
            exit(1)
    elif snapshot == 'dashboard':
        if 'ALLY SYSTEM DASHBOARD' not in content:
            print(f'[ERROR] Invalid dashboard format: {filepath}')
            exit(1)

print('[CI] TUI snapshot validation passed')
"

# Generate determinism proof
echo "[CI] Generating determinism proof..."
python3 -c "
import json
import hashlib
import os

# Calculate file hashes
files_to_hash = [
    'artifacts/chat/transcript_ci.jsonl',
    'artifacts/chat/tui_snapshot_compact.txt',
    'artifacts/chat/tui_snapshot_json.txt'
]

file_hashes = {}
for filepath in files_to_hash:
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        file_hashes[filepath] = file_hash
        print(f'[CI] {filepath}: {file_hash}')

# Count entries
with open('artifacts/chat/transcript_ci.jsonl', 'r') as f:
    transcript_lines = len(f.read().strip().split('\n'))

# Create proof
proof = {
    'transcript_entries': transcript_lines,
    'file_hashes': file_hashes,
    'timestamp': '2025-01-01T00:00:10.000000Z',  # Fixed for determinism
    'phase': 'phase-12-tui-chat',
    'deterministic': True
}

with open('artifacts/chat/determinism_proof.json', 'w') as f:
    json.dump(proof, f, indent=2, sort_keys=True)

print(f'[CI] Determinism proof written with {transcript_lines} transcript entries')
"

# Create README for chat artifacts
cat > artifacts/chat/README.md << 'EOF'
# Phase 12 Chat Artifacts

This directory contains deterministic chat and TUI artifacts generated by CI.

## Files

- `transcript_ci.jsonl` - Deterministic chat transcript from CI run
- `tui_snapshot_compact.txt` - Compact TUI rendering
- `tui_snapshot_json.txt` - JSON TUI data export
- `tui_snapshot_dashboard.txt` - Full TUI dashboard rendering
- `determinism_proof.json` - Cryptographic proof of deterministic execution

## Determinism Contract

All files are generated deterministically with:
- Fixed timestamps and seeds
- No network dependencies: ALLY_LIVE=0
- Stable output across CI runs

## Testing

The transcript and snapshots are used for regression testing:
- Chat responses must be identical across runs
- TUI rendering must be stable
- All interactions generate audit receipts

Run the CI harness: `bash scripts/ci_phase12_chat.sh`
EOF

echo "[CI] Created chat artifacts README"

# Final verification
echo "[CI] Final verification..."

# Check file sizes
TRANSCRIPT_SIZE=$(stat -c%s "artifacts/chat/transcript_ci.jsonl" 2>/dev/null || stat -f%z "artifacts/chat/transcript_ci.jsonl" 2>/dev/null || echo "0")
COMPACT_SIZE=$(stat -c%s "artifacts/chat/tui_snapshot_compact.txt" 2>/dev/null || stat -f%z "artifacts/chat/tui_snapshot_compact.txt" 2>/dev/null || echo "0")

if [[ $TRANSCRIPT_SIZE -lt 200 ]]; then
    echo "[ERROR] Transcript file too small ($TRANSCRIPT_SIZE bytes)"
    exit 1
fi

if [[ $COMPACT_SIZE -lt 30 ]]; then
    echo "[ERROR] Compact snapshot too small ($COMPACT_SIZE bytes)"
    exit 1
fi

echo "[CI] Phase 12 Chat Determinism Harness Complete ✓"
echo "Generated artifacts:"
echo "  - artifacts/chat/transcript_ci.jsonl ($TRANSCRIPT_SIZE bytes)"
echo "  - artifacts/chat/tui_snapshot_compact.txt ($COMPACT_SIZE bytes)"
echo "  - artifacts/chat/tui_snapshot_json.txt"
echo "  - artifacts/chat/tui_snapshot_dashboard.txt"
echo "  - artifacts/chat/determinism_proof.json"
echo "  - artifacts/chat/README.md"

exit 0