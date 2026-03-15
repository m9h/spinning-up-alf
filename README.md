# Spinning Up in Active Inference

**From Thorndike's cats to the Free Energy Principle — a hands-on curriculum bridging Reinforcement Learning and Active Inference**

<p align="center">
  <img src="https://img.shields.io/badge/modules-16-blue" alt="16 modules"/>
  <img src="https://img.shields.io/badge/JAX-GPU_accelerated-green" alt="JAX GPU"/>
  <img src="https://img.shields.io/badge/license-Apache_2.0-orange" alt="License"/>
  <img src="https://img.shields.io/badge/tests-16%2F16_passing-brightgreen" alt="Tests passing"/>
</p>

---

In 1898, Thorndike's cats learned to escape puzzle boxes by trial and error. In 1948, Tolman showed that rats build cognitive maps. In 1997, Schultz revealed that dopamine neurons compute prediction errors — the same signal as the TD learning algorithm proposed a decade earlier.

This curriculum traces that intellectual lineage from animal behavior experiments through modern RL into Active Inference and the Free Energy Principle. You will implement every algorithm, see every correspondence, and understand why these two frameworks — seemingly so different — are solving the same problem.

> Modeled on OpenAI's [Spinning Up in RL](https://spinningup.openai.com/), and designed as a pre-course introduction to [Neuromatch Academy](https://neuromatch.io/)'s computational neuroscience curriculum.

## Who is this for?

- Graduate students in computational neuroscience, cognitive science, or AI
- RL practitioners curious about Active Inference
- Neuroscientists who want to understand computational models of behavior
- Anyone who finds the Free Energy Principle intriguing but impenetrable

**Prerequisites:** Python, NumPy, basic probability (Bayes' rule). No prior RL or AIF experience required.

---

## The Curriculum

### Part 1 — From Animals to Algorithms

The behavioral science that motivated both RL and Active Inference.

| | Module | What you'll learn | Key figures |
|---|---|---|---|
| [01](notebooks/01_behaving_animal.ipynb) | **The Behaving Animal** | Stimulus-response vs. cognitive maps. Your first neuro-nav agent. | Thorndike, Pavlov, Skinner, Tolman, Tinbergen |
| [02](notebooks/02_bellman_and_value.ipynb) | **Bellman and Value** | MDPs, the Bellman equation, value iteration from scratch. | Bellman |
| [03](notebooks/03_td_learning.ipynb) | **Dopamine and TD Learning** | Prediction errors, the RPE hypothesis, successor representations. | Rescorla-Wagner, Schultz, Dayan |

### Part 2 — The RL Track

Classical and modern RL, building toward the bridge to Active Inference.

| | Module | What you'll learn | Algorithms |
|---|---|---|---|
| [04](notebooks/04_exploration_exploitation.ipynb) | **Exploration & Exploitation** | Epsilon-greedy, softmax, UCB. Why AIF dissolves the dilemma. | SARSA, Q-learning, UCB |
| [05](notebooks/05_model_based_rl.ipynb) | **Model-Based RL** | Dyna, planning as inference. The T-matrix IS the B-matrix. | DynaQ, MBV |
| [06](notebooks/06_policy_gradients.ipynb) | **Policy Gradients** | REINFORCE, actor-critic, entropy regularization = free energy. | REINFORCE, A2C |
| [07](notebooks/07_deep_rl.ipynb) | **Deep RL** | Function approximation, DQN, PPO, SAC. Representations. | DQN, PPO, SAC |

### Part 3 — The Active Inference Track

Everything changes. We start from a model of the world, not a reward signal.

| | Module | What you'll learn | Key equations |
|---|---|---|---|
| [08](notebooks/08_generative_models_fep.ipynb) | **Generative Models & FEP** | Bayesian brain, variational inference, free energy. | F = KL + E[ln p], ELBO |
| [09](notebooks/09_first_aif_agent.ipynb) | **Your First AIF Agent** | A/B/C/D matrices, the T-maze, belief updating. | P(s\|o), EFE |
| [10](notebooks/10_efe_decomposition.ipynb) | **EFE Decomposition** | Pragmatic vs. epistemic value. Why curiosity is built in. | G = -pragmatic - epistemic |
| [11](notebooks/11_learning_generative_models.ipynb) | **Learning the World** | Differentiable HMMs, gradient-based parameter recovery in JAX. | Forward algorithm, jax.grad |
| [12](notebooks/12_deep_aif.ipynb) | **Deep Active Inference** | Neural network generative models, encoder collapse. | Amortized inference |

### Part 4 — The Rosetta Stone

The two frameworks converge. Under the right conditions, they produce identical behavior.

| | Module | What you'll learn | Key insight |
|---|---|---|---|
| [13](notebooks/13_rosetta_stone.ipynb) | **The Rosetta Stone** | Formal RL-AIF correspondences. Same environment, two agents, same policy. | Q(s,a) = -G(a) when A=I |
| [14](notebooks/14_jax_scaling.ipynb) | **JAX Scaling** | `jit`, `vmap`, `grad` — run 1000 agents in parallel. | Sub-linear batch scaling |

### Part 5 — Capstone

Multi-level models and multi-agent social intelligence.

| | Module | What you'll learn | Application |
|---|---|---|---|
| [15](notebooks/15_hierarchical_aif.ipynb) | **Hierarchical AIF** | Context-dependent perception, temporal abstraction, cross-level info gain. | Context-dependent T-maze |
| [16](notebooks/16_multiagent_commons.ipynb) | **Multi-Agent Commons** | Ostrom's design principles, FEP meets social simulation, experiment ladder L0-L7. | SustainHub open-source commons |

---

## The Dual Perspective

Every module includes **Dual Perspective boxes** that show the same concept through both lenses:

<table>
<tr>
<td width="50%" style="background: #E3F2FD; padding: 12px;">
<b>RL Perspective</b><br/>
The TDQ agent learns Q(s,a) — cached stimulus-response mappings. It never builds a model. This is Thorndike's Law of Effect as an algorithm.
</td>
<td width="50%" style="background: #FBE9E7; padding: 12px;">
<b>Active Inference Perspective</b><br/>
The AIF agent builds transition matrices B(s'|s,a) and plans by simulating consequences. Like Tolman's rats taking shortcuts they've never walked.
</td>
</tr>
</table>

## The Rosetta Stone (Module 13)

The central result of the curriculum:

| RL | Active Inference | When equivalent |
|---|---|---|
| V(s) | -G(pi) | Epistemic value = 0 |
| R(s) | ln C(o) | Log-preference = log-reward |
| Q(s,a) | -G(a) | Fully observable (A = I) |
| pi(a\|s) = softmax(Q/tau) | pi(a) = softmax(-gamma*G) | tau = 1/gamma |
| T(s'\|s,a) | B(s'\|s,a) | Always identical |
| Epsilon-greedy | Epistemic value | AIF's is principled, RL's is bolted on |

---

## Installation

```bash
# Clone
git clone https://github.com/m9h/spinning-up-alf.git
cd spinning-up-alf

# Create environment (requires uv: https://docs.astral.sh/uv/)
uv venv .venv --python 3.12
source .venv/bin/activate

# Core dependencies
uv pip install numpy matplotlib seaborn jupyter ipywidgets scikit-learn

# JAX with GPU support (use jax[cpu] if no CUDA)
uv pip install jax[cuda12]

# RL environments and agents
git clone https://github.com/awjuliani/neuro-nav.git ../neuro-nav
uv pip install -e ../neuro-nav

# Active Inference library
git clone https://github.com/m9h/alf.git ../alf
uv pip install -e ../alf

# Optional: PGMax (factor graphs)
git clone https://github.com/m9h/PGMax.git ../PGMax
uv pip install -e ../PGMax

# Optional: Concordia (Module 16 capstone only)
git clone https://github.com/m9h/concordia.git ../concordia

# Install Jupyter kernel
python -m ipykernel install --user --name spinning-up-alf

# Launch
jupyter notebook
```

## Key References

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction* — The RL bible
- Parr, Pezzulo & Friston (2022). *Active Inference* — The AIF textbook
- Smith, Friston & Whyte (2022). *A step-by-step tutorial on active inference* — Clearest AIF tutorial
- Millidge, Tschantz & Buckley (2020). *Whence the expected free energy?* — The RL-AIF equivalence proof
- Sajid et al. (2021). *Active inference: Demystified and compared* — AIF vs. everything else
- Schultz, Dayan & Montague (1997). *A neural substrate of prediction and reward* — Dopamine = TD error
- Tolman (1948). *Cognitive maps in rats and men* — The original model-based argument
- Ostrom (1990). *Governing the Commons* — Design principles for Module 16

## Codebase

This curriculum builds on:

- **[ALF](https://github.com/m9h/alf)** — Active inference/Learning Framework: JAX-native, differentiable, GPU-accelerated
- **[neuro-nav](https://github.com/awjuliani/neuro-nav)** — Neuroscience-inspired RL environments and agents
- **[Concordia](https://github.com/m9h/concordia)** — Multi-agent social simulation (SustainHub)

## License

Apache 2.0
