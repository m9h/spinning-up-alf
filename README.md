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

## How this fits with other resources

There are excellent RL courses already. This one exists because none of them do what we do.

| | [Spinning Up (OpenAI)](https://spinningup.openai.com/) | [Deep RL Course (HuggingFace)](https://huggingface.co/learn/deep-rl-course/) | [Neuromatch Academy](https://compneuro.neuromatch.io/) | **This curriculum** |
|---|---|---|---|---|
| **RL algorithms** | VPG, TRPO, PPO, DDPG, TD3, SAC | Q-learning, DQN, PPO, A2C | Multi-armed bandits, Q-learning, model-based (1 day) | Q-learning, SARSA, Dyna, REINFORCE, A2C, DQN, PPO, SAC |
| **Active Inference** | -- | -- | -- | Full AIF track: POMDP generative models, EFE, deep AIF, hierarchical |
| **Free Energy Principle** | -- | -- | -- | VFE, ELBO decomposition, surprise minimization, FEP as unifying principle |
| **RL-AIF bridge** | -- | -- | -- | Formal Rosetta Stone: Q = -G, softmax equivalence, same-environment proof |
| **Neuroscience grounding** | -- | -- | Bayesian decisions, HMMs, Kalman filters | Thorndike, Pavlov, Tolman, Schultz (dopamine = TD error), successor representations |
| **Animal behavior history** | -- | -- | Implicit | Explicit: ethology, reinforcement schedules, cognitive maps, Tinbergen's 4 questions |
| **Bayesian brain** | -- | -- | W3D1-D2 (inference, HMMs) | Helmholtz, variational inference, generative models, belief propagation |
| **Differentiable inference** | -- | -- | -- | jax.grad through the forward algorithm, learning A/B matrices from data |
| **GPU-accelerated AIF** | -- | -- | -- | JAX jit/vmap/grad, 1000 batch agents on GPU |
| **Multi-agent / social** | -- | Unit 7 (broken) | -- | Ostrom's commons, Concordia social simulation, experiment ladder L0-L7 |
| **Primary language** | Python/TF | Python/PyTorch | Python/NumPy | Python/JAX |
| **Environments** | MuJoCo | Atari, Unity, VizDoom, Godot | Custom tutorials | neuro-nav grid worlds (neuroscience-native) |

**The gap we fill:** No existing resource teaches RL and Active Inference side-by-side, in the same environments, with escalating complexity, grounded in the animal behavior research that motivated both frameworks. Spinning Up teaches RL but not AIF. HuggingFace teaches practical deep RL but no theory of mind. Neuromatch covers Bayesian inference and one day of RL, but never mentions the Free Energy Principle. We bridge all three.

**How we complement Neuromatch:** Neuromatch W3D1 (Bayesian Decisions) and W3D4 (Reinforcement Learning) are the two days most relevant to our curriculum. Students who complete our Modules 1-8 before Neuromatch will arrive with deep intuition for why Bayesian inference and RL are connected — making W3 substantially more meaningful. Our Module 13 (Rosetta Stone) then formalizes what Neuromatch leaves implicit.

## Who is this for?

- Graduate students in computational neuroscience, cognitive science, or AI
- RL practitioners curious about Active Inference
- Neuroscientists who want to understand computational models of behavior
- Anyone who finds the Free Energy Principle intriguing but impenetrable
- Students preparing for Neuromatch Academy who want deeper RL/Bayesian foundations

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
| [16](notebooks/16_multiagent_commons.ipynb) | **Multi-Agent Commons** | Ostrom's design principles, FEP meets social simulation, experiment ladder L0-L7. | Concordia social simulation |

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

## Companion resources

This curriculum is designed to work alongside — not replace — other excellent courses:

| Resource | What it's best for | Take it... |
|---|---|---|
| [INRIA scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) | Classical ML pipelines, model selection, evaluation | Before or alongside — builds general ML fluency |
| [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course/) | Practical deep RL training (Atari, Unity, VizDoom) | After our Part 2 — you'll understand *why* the algorithms work |
| [OpenAI Spinning Up](https://spinningup.openai.com/) | Reference implementations of PPO, SAC, TD3 | As a reference during our Modules 6-7 |
| [Neuromatch Comp Neuro](https://compneuro.neuromatch.io/) | Full computational neuroscience (3 weeks, live) | After our Modules 1-8 — deep RL/Bayesian intuition for their W3 |
| [Neuromatch Deep Learning](https://deeplearning.neuromatch.io/) | PyTorch, CNNs, VAEs, diffusion, transformers, RL | Before or alongside — their W2D4 (VAEs) connects to our Module 8, their W3D4-D5 (RL) to our Parts 1-2 |
| [Neuromatch NeuroAI](https://neuroai.neuromatch.io/) | Comparing biological and artificial networks, representations | After our full curriculum — our Rosetta Stone (Module 13) is NeuroAI applied to decision-making |
| [Lovelace](https://computationalcognitivescience.github.io/lovelace/) | Computational cognitive science, probabilistic models of cognition | Alongside our Part 3 — same Bayesian brain ideas, different formalism |

### Detailed correspondences with Neuromatch

Our curriculum was designed as preparation for Neuromatch Academy. Here is where the specific modules align:

| Our module | Neuromatch Comp Neuro | Neuromatch Deep Learning | Neuromatch NeuroAI |
|---|---|---|---|
| 01-03 (Animal behavior, Bellman, TD) | **W3D4** RL (bandits, Q-learning, model-based) | W3D4-D5 (RL fundamentals, RL for games) | W1D2 (contrastive and RL for generalization) |
| 04 (Exploration) | W3D1 (Bayesian decisions) | -- | -- |
| 05 (Model-based RL) | W3D3 (Optimal control) | -- | W2D1 (macrocircuits, modularity) |
| 06-07 (Policy gradients, Deep RL) | -- | W3D4-D5 (RL, MCTS) | -- |
| 08 (Generative models, FEP) | W3D1-D2 (Bayesian inference, HMMs, Kalman) | **W2D4** (VAEs, diffusion models) | -- |
| 09-10 (AIF agent, EFE) | -- | -- | -- |
| 11 (Learning generative models) | W3D2 (EM algorithm) | W2D4 (VAEs) | W2D3 (microlearning) |
| 13 (Rosetta Stone) | -- | -- | **W1D3** (comparing artificial and biological networks) |
| 14 (JAX scaling) | -- | -- | -- |
| 15-16 (Hierarchical, multi-agent) | -- | -- | W2D1-D2 (macrocircuits, neurosymbolic) |

The cells marked **bold** are the strongest correspondences. Our Modules 9-12 (the core AIF track) have no Neuromatch equivalent — that is the gap we fill.

### Multi-agent ecosystems: from Gym to social simulation

Module 16 connects to a broader ecosystem of multi-agent tools. Here is the landscape:

| Tool | What it does | Relation to our curriculum |
|---|---|---|
| [Gymnasium](https://gymnasium.farama.org/) (Farama Foundation) | Successor to OpenAI Gym. The standard single-agent RL API. | Our Modules 1-7 use [neuro-nav](https://github.com/awjuliani/neuro-nav) which wraps Gym. Future: migrate to Gymnasium. |
| [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | Gymnasium for multi-agent RL. Standard MARL API. | Natural API for Module 16's multi-agent experiments. |
| [JaxMARL](https://github.com/FLAIROx/JaxMARL) | GPU-accelerated MARL environments and algorithms in JAX. 12,500x speedup via vmap. | Direct complement to our Module 14 (JAX scaling). Run 1000s of multi-agent episodes in parallel. |
| [SocialJax](https://arxiv.org/abs/2503.14576) | JAX suite for sequential social dilemmas: Commons Harvest, Clean Up, Territory. | The JAX-native version of our Module 16 scenarios. 50x faster than Melting Pot. ICLR 2026. |
| [Melting Pot](https://github.com/google-deepmind/meltingpot) (DeepMind) | 50+ multi-agent substrates testing cooperation, competition, commons tragedies. | The benchmark our Module 16 (Ostrom's commons) prepares students to understand. |
| [Concordia](https://github.com/google-deepmind/concordia) (DeepMind) | LLM-powered generative agent social simulation. | Our Module 16 builds on this directly — AIF agents in Concordia's SustainHub game. |

**The progression:** Gymnasium (single agent) -> PettingZoo (multi-agent API) -> JaxMARL/SocialJax (GPU-accelerated MARL with social dilemmas) -> Melting Pot (rich evaluation) -> Concordia (LLM + AIF social simulation). Our curriculum teaches the theory (Modules 1-15) that makes these tools meaningful, and Module 16 lands in the Concordia layer where AIF meets LLM-powered social agents.

### Social policy simulation: from economics to commons governance

Module 16's multi-agent commons simulation connects to a deep tradition in computational economics. The [AI Economist](https://github.com/salesforce/ai-economist) (Salesforce/Harvard, [Science Advances 2022](https://www.science.org/doi/10.1126/sciadv.abk2607)) pioneered **two-level (bi-level) deep MARL** for policy design: inner-level agents optimize their own utility while an outer-level social planner learns optimal tax policy. This Stackelberg structure — where the planner accounts for agents adapting to policy — is exactly the tension Ostrom studied in real commons, and what our Module 16 explores with AIF agents in Concordia's SustainHub and Collective Innovation games.

Agent-based computational economics (ACE) has modeled these dynamics since the 1990s. What's new is coupling ABMs with deep RL and LLMs:

| Framework | What it does | Relevance to Module 16 |
|---|---|---|
| [AI Economist](https://github.com/salesforce/ai-economist) | Two-level deep RL for tax policy design (Gym-style) | The bi-level RL template: agents + planner, both learning |
| [BeforeIT.jl](https://github.com/bancaditalia/BeforeIT.jl) | **Bank of Italy** ABM for macro forecasting. First ABM matching DSGE accuracy. Parametrized with real data (Austria, Italy). | Real-data-grounded ABM — shows institutional adoption |
| [Mesa](https://github.com/mesa/mesa) | Python ABM framework (500+ papers, Mesa 4 in development) | General-purpose ABM; Concordia's social layer is richer |
| [ABCE](https://github.com/AB-CE/abce) | ABM for economics (trade, production, consumption) | Classical ACE toolkit |
| [EconAgent](https://aclanthology.org/2024.acl-long.829/) | LLM-powered agents for macroeconomic simulation (ACL 2024) | Same GABM paradigm as Concordia, applied to macro |
| [ABIDES-Economist](https://arxiv.org/html/2402.09563v1) | RL agents in economic simulation via Gym interface | Bridges RL and econ simulation |

Central banks and governments also publish open-source models, though these are typically DSGE (equation-based) rather than agent-based:

| Model | Institution | Notes |
|---|---|---|
| [DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl) | **New York Fed** | Julia. Now extending to heterogeneous-agent (HANK) models |
| [FRB/US](https://www.federalreserve.gov/econres/us-models-about.htm) | **Federal Reserve Board** | The Fed's main policy model |
| [OpenSourcedMacroModels](https://github.com/dkgaraujo/OpenSourcedMacroModels) | Collection | 20+ central bank models (Bank of Finland, French MOF, etc.) |
| [ABCredit.jl](https://github.com/bancaditalia/ABCredit.jl) | **Bank of Italy** | Credit/macro ABM |

**Why this matters for our curriculum:** The GABM wave — LLM agents replacing hand-coded behavioral rules — is exactly what Concordia enables. Our Module 16 shows how Active Inference provides principled decision-making (EFE decomposition into productivity vs. community learning) where pure RL agents would need ad-hoc reward shaping. The experiment ladder (L0 pure RL -> L7 LLM+AIF) demonstrates this progression. The Bank of Italy's [BeforeIT.jl](https://github.com/bancaditalia/BeforeIT.jl) shows that ABMs are being taken seriously by institutions — the question is whether AIF-grounded agents can do better than rule-based ones in these policy-relevant settings.

### Where to go next: Embodied intelligence

Our curriculum ends at Module 16 with multi-agent social simulation — but the natural next step is **embodiment**: giving your Active Inference agents bodies in physically realistic worlds.

**[Ludobots](https://github.com/jbongard/pyrosim)** (Josh Bongard, UVM) — The world's only MOOC taught from [reddit](https://www.reddit.com/r/ludobots/). Students evolve neural-network-controlled robots with arbitrary body plans in [pyrosim](https://github.com/mec-lab/pyrosim)/[pybullet](https://pybullet.org/). The progression — hill climber, neural networks, evolved morphologies — is the embodied cognition counterpart to our disembodied grid-world agents. Bongard's research on [xenobots](https://www.pnas.org/doi/10.1073/pnas.1910837117) (living robots designed by evolutionary algorithms) is where this line of work leads. Our Module 1 (animal behavior) and his evolutionary robotics share the same deep question: what is the relationship between an agent's body, its nervous system, and its behavior?

**[HBP Neurorobotics Platform](https://neurorobotics.net/)** (EU Human Brain Project / EBRAINS) — Connects spiking neural network simulators ([NEST](https://www.nest-simulator.org/)) to robot physics ([Gazebo](https://gazebosim.org/)) via a Closed-Loop Engine. Transfer Functions bridge brain simulation output to robot actuators and sensor data back to neural input. The NRP's [EPFL Neurorobotics MOOC](https://github.com/HBPNeurorobotics/EPFLx-RoboX-Neurorobotics) teaches SARSA and self-organizing maps on embodied robots. Where our curriculum uses discrete POMDPs and tabular/JAX inference, the NRP uses continuous spiking networks and physics — but the question is the same: how does an agent's model of the world drive its actions?

**The connection:** Active Inference was always a theory of *embodied* intelligence — Friston's original formulations describe organisms minimizing free energy through both perception and action on a physical world. Our Modules 8-15 teach the math; Ludobots and the NRP provide the bodies. A future Module 17 could bridge the gap: an Active Inference agent controlling an evolved morphology in pybullet, closing the loop from Thorndike's cats to simulated creatures that learn, plan, and adapt.

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
- **[Concordia](https://github.com/m9h/concordia)** — Multi-agent social simulation

## Roadmap

Pedagogical improvements planned, learning from the [scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) design:

- [ ] **Colab badges** — One-click execution for every notebook
- [ ] **Glossary** — RL-AIF terminology Rosetta Stone (the jargon barrier is real)
- [ ] **Wrap-up quizzes** — MCQ comprehension checks at module boundaries
- [ ] **Separate exercise notebooks** — So students can't peek at solutions
- [ ] **Module 0** — Pre-requisite refresher (probability, linear algebra, JAX basics)

## License

Apache 2.0
