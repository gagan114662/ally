# ally/utils/killswitch.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class KillSwitchConfig:
    max_slippage_bps: int = 150
    max_latency_ms: int = 5000
    max_loss_bps: int = 300

def evaluate_killswitch(slippage_bps:int, latency_ms:int, loss_bps:int, cfg:KillSwitchConfig)->bool:
    return (
        slippage_bps > cfg.max_slippage_bps or
        latency_ms   > cfg.max_latency_ms   or
        loss_bps     > cfg.max_loss_bps
    )