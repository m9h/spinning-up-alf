# Course Design: Introduction to Computational Neuroscience and NeuroAI

## Overview

A HuggingFace-style introductory course in computational neuroscience and neuroAI, built on NeuroFedora's open-source ecosystem. Each module is a self-contained Jupyter notebook (~2 hours), runs in the browser (Colab/Binder), and follows the HF pattern: **hands-on first, theory second**.

This document serves as the architectural blueprint for the course. It maps each module to primary tools, educational source materials, and prerequisite knowledge.

## Design Principles

1. **Browser-first**: Every module runs in Colab or Binder — no local install required
2. **Progressive**: Each module builds on the previous
3. **Hands-on**: Code before theory (HF pattern)
4. **Tool diversity**: Students learn multiple tools, not just one
5. **Real data**: Use actual neural recordings where possible
6. **Open source**: All tools from NeuroFedora ecosystem where available
7. **Quizzes/exercises**: End of each module, three difficulty levels

## Tool Assessment

### Tier 1 — Core Course Backbone

| Package | Rating | Educational Material | Why Top Tier |
|---------|--------|---------------------|-------------|
| **Brian2** | 9/10 | Interactive browser tutorials (Binder), Neuronal Dynamics textbook exercises, INCF courses | Pythonic syntax, lowest barrier to entry, used in the leading compneuro textbook |
| **NEST + NEST Desktop** | 8/10 | NEST Desktop GUI for beginners, summer schools, EBRAINS workshops | Purpose-built for teaching; GUI-first entry point before code |
| **Neuromatch Academy** | 9/10 | Complete modular curriculum, already HF-style structure | Ready-made structure; covers modeling, ML, dynamical systems |

### Tier 2 — Strong Supplements

| Package | Rating | Educational Material | Best For |
|---------|--------|---------------------|----------|
| **Elephant + Neo** | 7/10 | ANDA Spring School, video tutorials, EBRAINS | Neural data analysis |
| **MOOSE** | 7/10 | Workshops (IIT Bombay, CHINTA 2025), Jupyter tutorials | Multiscale modeling |
| **NEURON** | 6/10 | Summer courses, "Neurons in Action" | Biophysical modeling (steeper curve) |

### Tier 3 — Good Tools, Less Pedagogy

| Package | Rating | Notes |
|---------|--------|-------|
| **Arbor** | 5/10 | Good docs but less teaching focus |
| **PyNN** | 5/10 | Important for abstraction, better as supplement |

## Course Structure

### Module 0: Setup & Python Refresher
- **Primary Tool**: NumPy, Matplotlib
- **Educational Source**: Neuromatch Academy W0
- **Topics**: Python refresher, NumPy array operations, matplotlib plotting, Jupyter basics
- **Duration**: ~1.5 hours
- **Prerequisites**: Basic programming experience
- **Learning Outcomes**:
  - Work fluently with NumPy arrays
  - Create publication-quality plots
  - Navigate Jupyter notebooks

### Module 1: What is a Neuron?
- **Primary Tool**: NEST Desktop (GUI)
- **Educational Source**: Neuronal Dynamics Ch.1, NEST Desktop tutorials
- **Topics**: Membrane potential, ion channels, integrate-and-fire neuron, GUI exploration before code
- **Duration**: ~2 hours
- **Prerequisites**: Module 0
- **Learning Outcomes**:
  - Explain the biophysical basis of neural signaling
  - Build and simulate a leaky integrate-and-fire neuron in NEST Desktop
  - Transition from GUI to code-based simulation
- **Key Resource**: [NEST Desktop](https://nest-desktop.readthedocs.io)

### Module 2: Hodgkin-Huxley & Biophysics
- **Primary Tool**: Brian2
- **Educational Source**: Neuronal Dynamics Ch.2, Open Source Brain HH Tutorial
- **Topics**: HH model, voltage-gated channels, action potential generation, parameter exploration
- **Duration**: ~2 hours
- **Prerequisites**: Module 1
- **Learning Outcomes**:
  - Implement the Hodgkin-Huxley equations in Brian2
  - Understand gating variables and their dynamics
  - Reproduce the classic action potential shape
- **Key Resource**: [Neuronal Dynamics](https://neuronaldynamics.epfl.ch)

### Module 3: Synapses & Simple Networks
- **Primary Tool**: Brian2
- **Educational Source**: Neuronal Dynamics Ch.3–4
- **Topics**: Excitatory/inhibitory synapses, AMPA/GABA, small networks, E-I balance
- **Duration**: ~2 hours
- **Prerequisites**: Module 2
- **Learning Outcomes**:
  - Model chemical and electrical synapses
  - Build small networks and observe emergent dynamics
  - Understand excitatory-inhibitory balance

### Module 4: Neural Data & Spike Trains
- **Primary Tool**: Elephant + Neo
- **Educational Source**: ANDA Spring School materials
- **Topics**: Spike train statistics, firing rates, ISI distributions, cross-correlation, real data loading
- **Duration**: ~2 hours
- **Prerequisites**: Module 0
- **Learning Outcomes**:
  - Load and visualize neural recording data with Neo
  - Compute firing rates, ISI distributions, and Fano factors
  - Perform cross-correlation analysis
- **Key Resource**: [Elephant](https://elephant.readthedocs.io)

### Module 5: Neural Coding & Decoding
- **Primary Tool**: NumPy/SciPy
- **Educational Source**: Neuromatch Academy W1 (Model Types, Fitting)
- **Topics**: Rate coding, population coding, tuning curves, linear decoding, Bayesian decoding
- **Duration**: ~2 hours
- **Prerequisites**: Modules 0, 4
- **Learning Outcomes**:
  - Fit tuning curves to neural data
  - Implement linear and Bayesian decoders
  - Understand the neural coding debate (rate vs. temporal)

### Module 6: Dynamical Systems in Neuroscience
- **Primary Tool**: Brian2, NumPy
- **Educational Source**: Neuromatch Academy W2 (Linear Systems, Dynamical Systems)
- **Topics**: Phase planes, fixed points, bifurcations, oscillations, FitzHugh-Nagumo model
- **Duration**: ~2 hours
- **Prerequisites**: Modules 2–3
- **Learning Outcomes**:
  - Analyze neural models as dynamical systems
  - Find and classify fixed points
  - Understand bifurcations and their neural significance
  - Plot and interpret phase portraits

### Module 7: Learning Rules & Plasticity
- **Primary Tool**: Brian2
- **Educational Source**: Neuronal Dynamics Ch.19
- **Topics**: Hebbian learning, STDP, BCM rule, homeostatic plasticity, learning in spiking networks
- **Duration**: ~2 hours
- **Prerequisites**: Module 3
- **Learning Outcomes**:
  - Implement Hebbian and STDP learning rules in Brian2
  - Understand the role of plasticity in network function
  - Connect learning rules to neural network training

### Module 8: Decision-Making & Reinforcement Learning
- **Primary Tool**: NeuroGym + neuro-nav
- **Educational Source**: Spinning Up in AIF Modules 1–3
- **Topics**: Reward prediction errors, temporal difference learning, Q-learning, exploration, neural evidence
- **Duration**: ~2 hours
- **Prerequisites**: Modules 0, 5
- **Learning Outcomes**:
  - Implement TD learning and Q-learning agents
  - Run agents on NeuroGym cognitive tasks
  - Connect computational RL to dopamine signaling
- **Key Resource**: [NeuroGym](https://github.com/neurogym/neurogym)

### Module 9: Bayesian Brain & Active Inference
- **Primary Tool**: alf framework (from Spinning Up in AIF)
- **Educational Source**: Spinning Up in AIF Modules 8–9
- **Topics**: Bayesian inference, generative models, free energy principle, Active Inference basics
- **Duration**: ~2 hours
- **Prerequisites**: Module 5
- **Learning Outcomes**:
  - Formulate perception as Bayesian inference
  - Build a generative model with A/B/C/D matrices
  - Understand the free energy principle at an intuitive level
  - Compare Active Inference to reinforcement learning

### Module 10: Modern NeuroAI
- **Primary Tool**: NeuroGym + PyTorch
- **Educational Source**: Spinning Up in AIF Appendix A3
- **Topics**: RNNs on cognitive tasks, modern architectures (SSMs, CTRNNs), representational analysis
- **Duration**: ~2 hours
- **Prerequisites**: Modules 6, 8
- **Learning Outcomes**:
  - Train RNNs on NeuroGym cognitive tasks
  - Understand modern architectures for neural modeling
  - Apply representational analysis to trained networks
  - Connect computational models to neural data

### Module 11: Capstone Projects
- **Primary Tool**: Mix (student choice)
- **Educational Source**: Open Source Brain models, Allen Brain Atlas
- **Topics**: Independent project combining tools and concepts from previous modules
- **Duration**: ~4 hours (open-ended)
- **Prerequisites**: All previous modules
- **Suggested Projects**:
  1. **Neuron to Network**: Build a biophysical network in Brian2 that exhibits working memory
  2. **Decode the Brain**: Apply decoding methods from Module 5 to real neural data from Allen Brain Atlas
  3. **Train & Analyze**: Train an RNN on a NeuroGym task and perform dynamical systems analysis
  4. **Active Inference Agent**: Build an AIF agent for a novel NeuroGym task
  5. **Compare Frameworks**: Implement the same cognitive model in RL and AIF, compare predictions

## External Resources

| Resource | URL | Integration |
|----------|-----|-------------|
| **Neuronal Dynamics** (Gerstner et al.) | neuronaldynamics.epfl.ch | Brian2 exercises for Modules 1–3, 6–7 |
| **Neuromatch Academy** | compneuro.neuromatch.io | Structure template, Modules 0, 5, 6 |
| **INCF Training Space** | training.incf.org | Tool-specific courses |
| **Open Source Brain** | opensourcebrain.org | Browser simulations, Module 2 HH tutorial |
| **EBRAINS** | ebrains.eu | HBP educational portal, NEST integration |
| **Allen Brain Atlas** | brain-map.org | Real neural data for Modules 4, 5, 11 |
| **Spinning Up in Active Inference** | (this repository) | Modules 8–10, alf framework |

## Module Dependency Graph

```
Module 0 (Python)
├── Module 1 (Neuron) → Module 2 (HH) → Module 3 (Networks) → Module 6 (DynSys) → Module 7 (Plasticity)
├── Module 4 (Data) → Module 5 (Coding) → Module 8 (RL/Decision) → Module 10 (NeuroAI)
│                                        └── Module 9 (Bayes/AIF) ──┘
└── Module 11 (Capstone) ← all above
```

## Implementation Notes

### Technical Requirements
- Python 3.9+
- Core: NumPy, SciPy, Matplotlib
- Neuroscience: Brian2, NEST (via NEST Desktop for Module 1), Elephant, Neo
- ML: PyTorch (Module 10 only)
- Curriculum-specific: neurogym, neuro-nav, alf

### Colab/Binder Setup
Each notebook should begin with:
```python
# Run this cell to install dependencies (Colab/Binder)
# !pip install brian2 elephant neo neurogym neuro-nav
```

### Assessment Strategy
- **Per-module quizzes**: 5 multiple-choice questions testing conceptual understanding
- **Exercises**: Three levels (guided, intermediate, open-ended) — matching Spinning Up in AIF format
- **Capstone**: Open-ended project with rubric

## Relationship to Spinning Up in Active Inference

This course serves as a **broader on-ramp** to computational neuroscience. Students who complete it will have the background to dive into the full Spinning Up in Active Inference curriculum. Specifically:

| This Course | Spinning Up in AIF |
|---|---|
| Module 8 (RL basics) | Modules 1–7 (deep RL coverage) |
| Module 9 (AIF intro) | Modules 8–12 (full AIF treatment) |
| Module 10 (NeuroAI) | Appendices A1–A4 |
| Module 6 (DynSys) | Module 7.7 (architectures) |

The two curricula are complementary: this course provides breadth across computational neuroscience; Spinning Up provides depth in the RL↔AIF bridge.
