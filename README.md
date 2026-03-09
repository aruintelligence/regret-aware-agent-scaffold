# Regret-Aware Agent Scaffold
## Relation to Inward Physics

This scaffold is a minimal working prototype of several recurring themes in Inward Physics:

- Self-reference via execution trace observation  
- Regret as a directional force in possibility space  
- Inward calibration over outward expansion  
- Coherence & compression as attractors  
- Typed state as emergent ontology

The smoke test shows the system literally turning inward: measuring its own prediction failures, feeling historical regret, and rewiring itself to become more coherent over time.
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
