# Regret-Aware Agent Scaffold

Self-improving Python agent orchestrator with:
- Typed state (`DWState`)
- Dynamic route planning through specialist nodes
- Predictive route quality scoring
- Online calibration using prediction error + historical regret
- Persistent memory across runs (`RouteMemory`)
- Adaptive policy updates from master-agent feedback

## Features
- Planner predicts route quality before execution
- Measures actual quality after run
- Penalizes overconfident predictions asymmetrically
- Uses regret (how much better a known route was) to avoid bad paths
- Learns visibly in smoke test (error shrinks, quality rises, routes shorten)

## Quick Start
```bash
python regret_aware_scaffold.py
