# Spinning Up in Active Inference — Development Guide

## Quick start

```bash
cd /home/mhough/dev/spinning-up-alf
source .venv/bin/activate   # uv venv with JAX CUDA, neuronav, alf, pgmax
```

## Testing notebooks

Run all 16 notebooks headlessly:
```bash
for nb in notebooks/*.ipynb; do
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=spinning-up-alf "$nb" --output-dir=/tmp/nb_test/
done
```

## Dependencies

Three external repos installed editable into `.venv/`:

| Repo | Path | Branch | Install |
|------|------|--------|---------|
| alf | `/home/mhough/dev/alf` | main | `uv pip install -e /home/mhough/dev/alf` |
| neuro-nav | `/home/mhough/dev/neuro-nav` | main | `uv pip install -e /home/mhough/dev/neuro-nav` |
| PGMax | `/home/mhough/dev/PGMax` | main | `uv pip install -e /home/mhough/dev/PGMax` |

Notebook 16 also needs concordia: `/home/mhough/dev/concordia` (sustainhub-autoresearch branch), added via `sys.path.insert`.

## neuronav API (current)

- `GridEnv(template=..., size=..., obs_type=...)` — template is first positional arg
- Goal: `list(env.objects['rewards'].keys())[0]` — save BEFORE training (rewards consumed)
- `agent.Q` shape: `(n_actions, n_states)` — NOT `(n_states, n_actions)`
- `agent.M` (TDSR): `(n_actions, n_states, n_states)`
- SARSA: set `agent.last_exp = None` before first update and at episode start
- State count: `env.grid_size ** 2` (no `env.observation_space.n`)

## alf imports (not pgmax.aif)

All AIF imports use `alf`, NOT `pgmax.aif`:
- `from alf.agent import AnalyticAgent`
- `from alf.generative_model import GenerativeModel`
- `from alf.free_energy import variational_free_energy, expected_free_energy_decomposed`
- `from alf.benchmarks.t_maze import build_t_maze_model, TMazeEnv`
- `from alf.hierarchical import HierarchicalLevel, HierarchicalAgent`

## Notebook structure

Each notebook follows: narrative opener → math foundations → hands-on code → dual perspective box (RL↔AIF) → exercises → further reading.

The `utils/plotting.py` provides shared visualization (imported via `sys.path.insert(0, '..')`).
