# Spinning Up in Active Inference

**A comprehensive educational curriculum bridging Reinforcement Learning and Active Inference**

Modeled on OpenAI's [Spinning Up in RL](https://spinningup.openai.com/) and [Neuromatch Academy](https://neuromatch.io/), this curriculum takes students from animal behavior foundations through classical RL into Active Inference and the Free Energy Principle.

## Prerequisites

- Python 3.10+
- Comfort with NumPy and basic linear algebra
- Familiarity with probability (Bayes' rule, conditional distributions)
- No prior RL or AIF experience required

### Installation

```bash
# Core dependencies
pip install -r requirements.txt

# ALF — Active inference/Learning Framework (JAX-native, standalone)
cd /path/to/alf && pip install -e .

# neuro-nav (RL environments and agents)
cd /path/to/neuro-nav && pip install -e .

# Optional: PGMax (for BP-based AIF in Modules 9, 13, 15)
cd /path/to/PGMax && pip install -e .

# Optional: Concordia (for capstone Module 16)
cd /path/to/concordia && pip install -e .
```

## Curriculum Map

### Part 1: Foundations — From Animals to Algorithms (Modules 1-3)

| Module | Title | Key Concept | Primary Imports |
|--------|-------|-------------|-----------------|
| 01 | The Behaving Animal | S-R vs cognitive maps | `neuronav` |
| 02 | Bellman and Value | MDPs, value iteration | `neuronav` |
| 03 | TD Learning | Dopamine, prediction errors | `neuronav` |

### Part 2: The RL Track — Scaling Up (Modules 4-7)

| Module | Title | Key Concept | Primary Imports |
|--------|-------|-------------|-----------------|
| 04 | Exploration & Exploitation | On/off-policy, UCB | `neuronav` |
| 05 | Model-Based RL | Dyna, planning | `neuronav` |
| 06 | Policy Gradients | REINFORCE, A2C | `neuronav` |
| 07 | Deep RL | DQN, PPO, SAC | `neuronav` |

### Part 3: The Active Inference Track (Modules 8-12)

| Module | Title | Key Concept | Primary Imports |
|--------|-------|-------------|-----------------|
| 08 | Generative Models & FEP | Variational inference | `alf` |
| 09 | First AIF Agent | A/B/C/D matrices, BP | `pgmax.aif` or `alf` |
| 10 | EFE Decomposition | Pragmatic vs epistemic | `alf` |
| 11 | Learning Generative Models | Differentiable HMM | `alf` |
| 12 | Deep Active Inference | Neural generative models | `alf` |

### Part 4: Convergence — RL Meets AIF (Modules 13-14)

| Module | Title | Key Concept | Primary Imports |
|--------|-------|-------------|-----------------|
| 13 | The Rosetta Stone | Formal correspondences | `pgmax.aif`, `alf`, `neuronav` |
| 14 | JAX Scaling | Batch agents, vmap | `alf` |

### Part 5: Capstone — Social Intelligence (Modules 15-16)

| Module | Title | Key Concept | Primary Imports |
|--------|-------|-------------|-----------------|
| 15 | Hierarchical AIF | Temporal abstraction | `pgmax.aif` |
| 16 | Multi-Agent Commons | Ostrom, social AIF | `concordia`, `alf` |

## Pedagogical Design

Each module follows a consistent structure:

1. **Narrative opener** — Historical and conceptual context
2. **Mathematical foundations** — Key equations with intuition
3. **Hands-on code** — Working examples from the codebase
4. **Dual Perspective boxes** — RL-AIF correspondences in every module
5. **Exercises** — 2-3 graded (guided to open-ended)
6. **Further reading** — 3-5 annotated references

## Key References

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*
- Parr, Pezzulo & Friston (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*
- Smith, Friston & Whyte (2022). A step-by-step tutorial on active inference
- Millidge, Tschantz & Buckley (2020). Whence the expected free energy?
- Sajid et al. (2021). Active inference: Demystified and compared

## Codebase Dependencies

This curriculum builds on three interconnected codebases:

- **ALF** — Active inference/Learning Framework: standalone JAX-native AIF (differentiable, GPU-accelerated)
- **PGMax-AIF** — Active Inference via factor graph belief propagation (BP-based)
- **neuro-nav** — RL agents and neuroscience-inspired environments
- **Concordia/SustainHub** — Multi-agent LLM social simulation with FEP

## License

Apache 2.0
