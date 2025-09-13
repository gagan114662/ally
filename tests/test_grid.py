import pytest
from ally.tools import TOOL_REGISTRY

pytestmark = pytest.mark.mgrid

def test_grid_submit_and_status():
    strategy_configs = [
        {"lookback": 20, "rebalance": "M"},
        {"lookback": 30, "rebalance": "M"},
        {"lookback": 20, "rebalance": "M"},  # Duplicate for dedup test
    ]

    result = TOOL_REGISTRY["grid.submit_jobs"](
        strategy_configs=strategy_configs,
        batch_id="test_batch",
        dedup=True
    )

    assert result.ok
    assert result.data["n_submitted"] == 2  # After dedup
    assert result.data["n_deduped"] == 1

    # Test status
    status = TOOL_REGISTRY["grid.status"](batch_id="test_batch")
    assert status.ok
    assert "summary" in status.data