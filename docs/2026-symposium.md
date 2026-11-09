# November 2026 symposium: project fit and follow-up plan

Reviewed November 8, 2026 (America/Los_Angeles). This is a research and attendance plan, not a submission or claim of acceptance.

The **6th Applied Active Inference Symposium is November 12–13, online and free**. The [official event page](https://activeinference.institute/projects/symposium/) supersedes earlier announcements that mentioned November 14. Use the [live program](https://2026symposium.activeinference.institute/) for changes and the [registration link](https://registersymposium2026.activeinference.institute/) to attend. The organizer's published snapshot is dated November 1: [program data and abstracts](https://github.com/ActiveInferenceInstitute/Symposium/blob/main/data/public/2026/program.json), [abstract book PDF](https://github.com/ActiveInferenceInstitute/Symposium/blob/main/data/public/2026/book.pdf), [calendar](2026-program.ics). This snapshot has 18 scheduled sessions and 57 presentation records, including 40 prerecorded presentations. It is not a guarantee that the live schedule is unchanged.

The announced presenter deadline on the official event page was October 31. Do not assume submissions are still open; a late lightning slot would need confirmation from the organizers. No registration, submission, or outreach was performed.

## Recommended live sessions

These priorities are our assessment of the fit between the program and the repos.

| Priority | Session, Pacific time | Why it fits | Concrete follow-up |
|---|---|---|---|
| 1 | Ryan Smith, Thu 06:00–07:00 | ALF's HGF, DDM, normative modeling, metacognition and cognitive tasks | Pick one empirical task; publish parameter recovery, posterior prediction, held-out evaluation and identifiability results before making clinical claims. |
| 1 | Alison Gopnik, Thu 09:00–10:00 | Modules 4, 8, 10 and 13 connect RL, curiosity and Bayesian models | Rewrite the Rosetta Stone around explicit conditions for policy equivalence; add a counterexample with informative observations. |
| 1 | Dan Corva / Kevin Backhouse, Fri 09:00–11:00 | cpomdp supplies the continuous-control counterpart to Module 17 | Build a matched ALF discrete / cpomdp continuous exercise with the same goal and sensor intervention. |
| 1 | Field Working Groups, Thu 08:00–09:00 and Fri 12:30–13:30 | Numerical comparison is the most useful contribution ALF can make now | Offer a small reference-model suite with pinned versions and explicit EFE conventions. |
| 2 | Takuya Isomura, Fri 04:00–05:00 | ALF differentiable learning and cognitive task adapters | Explore fitting generative models to behavior, with recovery and out-of-sample validation. |
| 2 | Ted J. Vrioni / Knowthyself, Thu 07:30–08:00 | A visual entry point for Modules 8–10 | Assess whether a model can be exported into a reproducible ALF notebook; verify export/API availability first. |
| 2 | Hongju Pae, Thu 10:00–11:30 | Hierarchical context in Module 15 | Compare augmented and unaugmented agents under matched seeds and likelihoods; distinguish context dynamics from policy optimization. |
| 2 | CogNarr, Thu 11:30–12:30 | Narrative belief representations and the Module 16 social capstone | Examine how narrative-derived assumptions could become inspectable generative-model specifications. |
| 2 | Mike Levin, Fri 07:00–09:00 | Embodiment and social/biological agency in Modules 16–17 | Use as conceptual context for embodied experiments, with operational behavioral measures. |
| 2 | Bradly Alicea, Fri 11:00–12:30 | Deep and hierarchical world models, Modules 11–15 | Discuss a compact, reproducible world-model learning curriculum and its present limitations. |

The [Working Groups proposal](https://activeinference.institute/working-groups/) explicitly calls for terminology correspondence and a versioned numerical comparison protocol. That makes a small, honest ALF/pymdp/RxInfer comparison more useful than another broad framework-positioning claim.

## Projects and recordings to follow

| Priority | Project / presenter | Relevance and next experiment |
|---|---|---|
| First | [cpomdp](https://github.com/inferogenesis/cpomdp), Corva / Backhouse | Continuous Gaussian inference and control. Reproduce the workshop's state-dependent sensor-noise comparison before extending Module 17. Its [documentation](https://cpomdp.inferogenesis.com/) describes a separate RxInfer reference backend. |
| First | [Generalized Notation Notation](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation), Daniel Friedman | Model specification and backend rendering. Prototype export of ALF's A/B/C/D/E matrices with shape and normalization checks; prove one T-maze round trip before proposing an ALF backend. |
| First | [pymdp](https://github.com/infer-actively/pymdp) and [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) | Reference implementations, not assumed numerical equivalents. Strengthen ALF tests from matching the winning action to comparing state beliefs, policy scores up to documented constants, and gradients where applicable. |
| First recording | Ahmad Nazzal's lexical T-maze | A close fit for Modules 9–10. Reproduce informative versus uninformative hints and an information-gain ablation using matched seeds. |
| First recording | Andrew Pashea's tutorial | Use as a conceptual review companion for Modules 8–15 and to check how discrete, continuous and hierarchical models are distinguished. |
| Second recording | Lancelot Da Costa's active world modeling | Relates to Modules 11–12. Separate fixed-model parameter learning from structure/model discovery in the curriculum roadmap. |
| Second recording | Tran Hoang Linh's dynamic action-space adaptation | Compare transition-learning strategies after a new action appears; evaluate held-out transition error and behavior. JAX shape changes and recompilation need an explicit design. |
| Second recording | Hyojun Choi's adaptive prosociality | Compare the Module 16 commons agents with simple reciprocal baselines and ablate the proposed regulation mechanism. |
| Exploratory | Nick Hay / Adam Safron's Commongrounds AI | Potential future embodied/social environment. The abstract describes plans as well as prototypes; verify available code and APIs before adding a dependency. |
| Exploratory | Patricia Craja's Percepia | Relevant to precision and metacognition. Its abstract describes synthetic LLM-persona experiments; do not treat these as human parameter-recovery or clinical validation. |

Presentation descriptions above are based on the [organizer's abstracts](https://github.com/ActiveInferenceInstitute/Symposium/blob/main/data/public/2026/program.json). Proposed experiments are our recommendations, not results or integration claims. Knowthyself, the lexical T-maze, and other talk-specific software have not been established here as installable public dependencies.

## How to present these repos

**ALF:** an inspectable JAX implementation connecting discrete generative models with learning and cognitive-model components. Lead with reproducibility and cross-framework comparison. Avoid claiming unique JAX support, unique differentiable inference, or clinical validation.

**Spinning Up:** a curriculum that teaches RL and active inference in shared tasks, then introduces world models, social settings and embodiment. The fit is strongest with the symposium's education and open-tooling themes.

A useful five-minute demonstration would show one T-maze, its matrices, a belief trace, and a policy-score comparison, followed by one differentiable learning example. Include the independent posterior-KL oracle and pymdp comparison. The EFE discrepancy is corrected; explain the remaining open-loop and additive-information assumptions instead of presenting a general RL/AIF equivalence.

Suggested working title: *From RL to active inference: reproducible tasks, explicit assumptions, and cognitive-model extensions in JAX.*

## Implementation review

The November maintenance adds a locked CPU environment; pinned ALF/neuro-nav source revisions; explicit cognitive and robotics extras; an exported pip requirements file; a portable Module 16 checkout path; a notebook validation/execution runner; and CI for core AIF lessons. Module 7's Q-array and successor-representation plotting APIs are repaired. Appendix A4 now imports the required ALF implementation instead of silently selecting a fallback with the opposite EFE sign.

Local verification on macOS ARM64 / Python 3.12 (November 8–9):

- **22/22 notebooks pass schema validation.**
- **11 affected lessons re-executed successfully against corrected ALF**: 04, 08–15, A2 and A4. Numerical regression tests for the manual rollout and plotting sign went from two failures to two passes. Module 13 now checks both a valid equivalence case and counterexamples; stale outputs were cleared from revised lessons.
- **21/22 notebooks executed successfully** after repairs: Modules 1–16 and appendices A0–A4. A0 is a references notebook; execution success is not an assessment of scientific correctness. Module 16 used the `sustainhub-autoresearch` Concordia checkout; the cognitive appendices used NeuroGym 2.3.1.
- **Module 17 remains unexecuted:** PyBullet 3.2.7 failed to compile on this Apple Silicon toolchain. The robotics extra is isolated so it cannot block the main curriculum installation. The [upstream build issue](https://github.com/bulletphysics/bullet3/issues/4712) documents related macOS failures. Run this lesson on a supported Linux/toolchain setup before presenting it as verified.
- Repaired Module 7's Q-value axes and 3D successor-representation plot; Module 16's decomposition dictionary indexing and checkout path; Appendix A1's removed `all_envs` call and Gymnasium reset/step API; and Appendix A4's inconsistent fallback.
- `uv pip check` passed. NeuroGym's constraints select NumPy 2.2.6 and Matplotlib 3.10.9 in this lock, while JAX is 0.12.0.1. This differs deliberately from ALF's current unconstrained scientific stack.
- Draft PRs enable remote CI; the counts above are local validation, not a claim that every remote check passed. No GPU test was performed. Notebook execution artifacts and JSON summaries are in the ignored `notebook-results/` directory. The [validation record](2026-validation.json) summarizes the latest result for each lesson.

See [ALF's maintenance review](../../alf/docs/2026-review-and-symposium.md) for numerical findings and test results. Installation from the pinned Git revision and dependency consistency were rechecked. The curriculum pins corrected ALF commit `c1c886b870acc2300aa22760aabaf2f8505d2d99` from [ALF draft PR #5](https://github.com/m9h/alf/pull/5). This makes the code and revised lessons install together. Editable sibling installs remain available for joint development.

Immediate priorities after this pass:

1. Extend the corrected single-modality oracle comparisons to joint multi-modal information and contingent planning.
2. Broaden the now-explicit RL/AIF comparison with matched held-out benchmarks.
3. Add a short reproducible learning experiment with multiple initializations and held-out NLL, rather than relying on one seed's parameter recovery.
4. Make a standalone social-agent exercise before requiring the larger Concordia research branch.
5. Add a cpomdp companion exercise after attending its workshop.

## Full scheduled program

All dates below are in 2026. Pacific time is **PST (UTC−8)**. Times come from the published November 1 snapshot. No timed slot is inferred for prerecorded talks.

| Pacific date / time | UTC time, same date | Presenter | Session |
|---|---|---|---|
| Nov 12, 04:00–05:00 | 12:00–13:00 | Sergio Rubin | An elementary form of agency at planetary-scale |
| Nov 12, 05:00–06:00 | 13:00–14:00 | Karl Friston | Opening Keynote and Panel |
| Nov 12, 06:00–07:00 | 14:00–15:00 | Ryan Smith | Recent advances in empirical applications of active inference in computational psychiatry |
| Nov 12, 07:00–07:30 | 15:00–15:30 | Bianca Avanzo | Root Frequency Theory: An Integrative Framework for the Continuity of Lived Experience |
| Nov 12, 07:30–08:00 | 15:30–16:00 | Ted J. Vrioni | Knowthyself — A code-free active inference workbench |
| Nov 12, 08:00–09:00 | 16:00–17:00 | Ed Ober | Working Groups Session 1 ~ Opening & Charter |
| Nov 12, 09:00–10:00 | 17:00–18:00 | Alison Gopnik | Active Inference, Bayesian Causal Learning and Empowerment in RL. |
| Nov 12, 10:00–11:30 | 18:00–19:30 | Hongju Pae | A Modular Perspective Layer for Active Inference Agents |
| Nov 12, 11:30–12:30 | 19:30–20:30 | John Boik, Glen EP Ropella | CogNarr Story Graph: A Formal Semantics for Narrative as a Belief Model |
| Nov 12, 12:30–13:30 | 20:30–21:30 | Montijn van Loo | Explicating and Comparing Prediction Architectures of Theories of Consciousness and its Application to INTREPID |
| Nov 13, 04:00–05:00 | 12:00–13:00 | Takuya Isomura | Reverse engineering generative models from zebrafish decision-making |
| Nov 13, 05:00–06:00 | 13:00–14:00 | Majid D. Beni | From Self-Evidencing to Global Governance: Active Inference and the Dynamics of Justice |
| Nov 13, 06:00–07:00 | 14:00–15:00 | Luca Possati | Active Inference and the Evolution of Mind and Society |
| Nov 13, 07:00–09:00 | 15:00–17:00 | Mike Levin | “Evolution, Emergence, Prediction.” |
| Nov 13, 09:00–11:00 | 17:00–19:00 | Dan Corva, Kevin Backhouse | Build a curious rocket: a hands-on continuous-state AIF session with cpomdp |
| Nov 13, 11:00–12:30 | 19:00–20:30 | Bradly Alicea | An in-depth discussion on “World models in natural and artificial intelligence” |
| Nov 13, 12:30–13:30 | 20:30–21:30 | Ed Ober | Working Groups Session 2 ~ Outcomes & Next Steps |
| Nov 13, 13:30–14:00 | 21:30–22:00 | Montijn van Loo | A New Politics: From Linear to Sorted Politics Through the Policy Map |

[Day 1 livestream](https://www.youtube.com/watch?v=-v4IaFY9MZI) · [Day 2 livestream](https://www.youtube.com/watch?v=ccze9ay8D0o)
