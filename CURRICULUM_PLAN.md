# Spinning Up in Active Inference: Curriculum Plan

## Context

This comprehensive educational resource — "Spinning Up in Active Inference" — is modeled on OpenAI's *Spinning Up in RL* and designed as a pre-course introduction to Neuromatch Academy's computational neuroscience curriculum. It takes students from animal behavior foundations through classical RL, into Active Inference and the Free Energy Principle, using our codebase across ALF, neuro-nav, and Concordia as the hands-on backbone.

This curriculum fills a gap: no existing resource teaches RL and AIF side-by-side with shared environments and escalating complexity, grounded in the history of behavioral science that motivated both frameworks.

## Existing Assets Inventory

**PGMax-AIF** (`pgmax/aif/`):
- `generative_model.py`, `inference.py`, `policy.py`, `agent.py` — Core AIF (Phase 1-2)
- `jax_native.py` — BatchAgent, vmap-accelerated inference
- `sequential_efe.py` — Forward-rollout EFE capturing value of information
- `learning.py` — Differentiable HMM learning via jax.lax.scan
- `free_energy.py` — VFE, EFE decomposition, generalized FE (Phase 4)
- `deep_aif.py` — Neural network generative models (Phase 4)
- `hierarchical.py` — Multi-level temporal hierarchy (Phase 4)
- `benchmarks/t_maze.py` — 8-state T-maze
- `benchmarks/neuronav_wrappers.py` — neuro-nav grid/graph env integration
- `benchmarks/pymdp_comparison.py` — pymdp vs PGMax head-to-head
- 212 tests across all modules

**neuro-nav** (external dependency):
- `agents/td_agents.py` — TDQ, SARSA, TDSR, TDAC
- `agents/mb_agents.py` — MBV (model-based value)
- `agents/dyna_agents.py` — DynaQ, DynaSR
- `agents/mc_agents.py` — Monte Carlo agents
- `agents/dist_agents.py` — Distributional RL (DistQ)
- `envs/grid_env.py`, `graph_env.py` — 19+ grid, 8+ graph environments
- 11 notebooks covering usage, deep RL, distributional RL, etc.

**Concordia/SustainHub** (external dependency):
- `active_inference.py` — Hand-rolled AIF for social simulation
- `pgmax_bridge.py` — PGMax integration bridge
- `fep_integration.py` — FEPBeliefTracker, AdaptiveGenerativeModel, FEPAgentComponent
- `experiments.py` — 8-level experiment ladder (L0 pure RL to L7 LLM-as-node)

---

## Curriculum Structure

### Part 1: Foundations — From Animals to Algorithms (Modules 1-3)

**Module 1: The Behaving Animal**
- Thorndike, Pavlov, Skinner, Tolman, Tinbergen
- S-R vs cognitive maps → RL vs AIF
- Hands-on: neuro-nav grid exploration

**Module 2: Reward, Value, and the Bellman Equation**
- MDPs, Bellman equations, dynamic programming
- Hands-on: tabular value iteration on FourRooms

**Module 3: Dopamine, Prediction Errors, and TD Learning**
- Rescorla-Wagner, Schultz, TD(0), successor representations
- Hands-on: TDQ/TDSR on TMaze, RPE plotting

### Part 2: The RL Track — Scaling Up (Modules 4-7)

**Module 4: Exploration, Exploitation, and Model-Free Methods**
- ε-greedy, UCB, SARSA vs Q-learning
- Hands-on: cliff-walking comparison

**Module 5: Model-Based RL and Planning**
- Dyna, MBV, planning as inference
- Hands-on: DynaQ sample efficiency, T-matrix visualization

**Module 6: Policy Gradients and Actor-Critic**
- REINFORCE, A2C, entropy regularization
- Hands-on: TDAC, entropy analysis

**Module 7: Deep RL and Representation Learning**
- Function approximation, DQN, PPO
- Hands-on: representation visualization

### Part 3: The Active Inference Track (Modules 8-12)

**Module 8: Generative Models and the Free Energy Principle**
- Bayesian brain, variational inference, F = KL + E[ln p]
- Hands-on: VFE computation with `free_energy.py`

**Module 9: Your First Active Inference Agent (Smith Tutorial)**
- A/B/C/D matrices, BP, T-maze from scratch
- Hands-on: full agent loop with `agent.py`

**Module 10: Expected Free Energy — Pragmatic and Epistemic Value**
- EFE decomposition, value of information, sequential EFE
- Hands-on: `sequential_efe.py`, comparison with RL exploration

**Module 11: Learning the World — Differentiable Generative Models**
- Forward algorithm, gradient-based parameter recovery
- Hands-on: `learning.py`, A/B matrix convergence

**Module 12: Deep Active Inference**
- Neural network generative models, encoder collapse
- Hands-on: `deep_aif.py`

### Part 4: Convergence — RL Meets AIF (Modules 13-14)

**Module 13: The Rosetta Stone — RL and AIF Are Closer Than You Think**
- Full correspondence table, same-environment comparison
- Hands-on: MBV vs AIF on neuronav FourRooms

**Module 14: Scaling with JAX — Batch Agents and Hardware Acceleration**
- jit, vmap, grad, BatchAgent benchmarks
- Hands-on: 1→1000 agent scaling

### Part 5: Capstone — Social Intelligence (Modules 15-16)

**Module 15: Hierarchical Active Inference and Temporal Abstraction**
- Multi-level models, context-dependent inference
- Hands-on: `hierarchical.py`

**Module 16: Multi-Agent Commons — FEP Meets Social Simulation**
- Ostrom, commons governance, experiment ladder
- Hands-on: SustainHub L0→L7, capstone exercise

---

## Implementation Notes

### Each notebook follows consistent structure:
1. **Narrative opener** — Historical/conceptual context
2. **Mathematical foundations** — Key equations with intuition
3. **Hands-on code** — Working examples using existing codebase
4. **Dual perspective boxes** — RL↔AIF correspondences
5. **Exercises** — 2-3 graded (guided → open-ended)
6. **Further reading** — 3-5 annotated references

### Build order (dependency):
1. First pass: Modules 1-3, 8-9 (foundations)
2. Second pass: Modules 4-7, 10-12 (depth)
3. Third pass: Modules 13-16 (convergence + capstone)

### Verification:
1. Each notebook runs end-to-end
2. All imports resolve
3. Existing tests still pass (212 tests)
4. Narrative coherence across all 16 modules
