# spec.md — Bifurcation-Guided Replay for General AI Decision Policies

**Target venue:** AAAI Main Technical Track, preferably AAAI-27 if the experimental cycle is immediately actionable.  
**Working method name:** **Bifurcation-Guided Replay**, abbreviated **BGR**.  
**Original idea name:** Bifurcation Level Replay / Bifurcation Replay.  
**Recommended paper title:** **Bifurcation-Guided Replay: Learning at the Success–Failure Boundary of Decision Policies**.  
**Recommended robotics-instantiation title for subsections/appendix:** **BGR-Suffix: Recovery-Margin Expansion for Robot Policies**.  
**Core design decision:** Treat robotics as a compelling *instantiation* and evaluation domain, not as the main intellectual scope. The paper should be about a general replay principle for sequential decision policies: train on replayable states whose local perturbation curve crosses from success to failure near the current policy boundary.

---

## 0. Executive summary

This spec turns the initial Bifurcation Level Replay idea into an AAAI-ready research paper plan. The resulting paper should not be framed as “CREST + fine-tuning for robotics.” It should be framed as a general algorithmic contribution to curriculum learning, robust policy training, and level replay for sequential decision systems.

The paper’s thesis should be:

> Decision policies learn most efficiently from replayable states at the local success–failure transition of their current competence. Bifurcation-Guided Replay estimates each state’s critical perturbation radius and preferentially trains near that boundary, thereby expanding policy recovery margins more sample-efficiently than uniform replay, regret-based level replay, fixed-radius perturbation training, or failure-only replay.

The original idea is strong, but for AAAI it needs three upgrades.

First, the unit should be generalized from “level” to **replayable decision state**. A “level” can be an environment seed, a task instance, a partially completed episode, a robot suffix state, a browser/task state, or a code-agent repository state. This makes the work broader than robotics and connects it to general AI.

Second, the novelty should be stated as **state-conditioned robustness geometry** rather than “difficulty.” Existing prioritized replay and curriculum methods often score examples by return, TD error, regret, or generator novelty. BGR scores replayable states by an estimated local recovery curve and its critical radius. The object of interest is not “how hard is this level?” but “where does the policy’s behavior switch from success to failure under controlled perturbation?”

Third, the expensive part — estimating recovery curves — needs a practical active-estimation mechanism. A reviewer will immediately ask whether evaluating multiple perturbation radii per state is too expensive. The paper must answer with a sequential estimator: coarse probing, monotone curve fitting, posterior uncertainty, and active binary/staircase sampling around the boundary.

The strongest paper shape is:

1. **General method:** Bifurcation-Guided Replay over replayable decision states.
2. **Formal object:** recovery curve, critical radius, recovery AUC, boundary sharpness, and uncertainty.
3. **Efficient estimator:** active bifurcation search instead of full dense curve evaluation every time.
4. **Training rule:** sample both states and perturbation radii near estimated success–failure boundaries.
5. **Empirical proof:** at least one procedural RL domain plus one robotics/VLA domain. Robotics demonstrates realism and importance; the non-robotics domain demonstrates generality.

The target contribution statement should be:

> We introduce Bifurcation-Guided Replay, a general replay algorithm for sequential decision policies that estimates state-conditioned success–failure transition radii and trains near these boundaries. Across procedurally generated RL tasks and robot manipulation suffix recovery, BGR improves recovery AUC and critical-radius distributions under held-out perturbations while preserving nominal task success.

---

## 1. Source idea and recommended changes

### 1.1 Original idea, preserved

The uploaded idea has the following core thesis:

> The best states to replay are not merely hard states; they are states near the policy’s success/failure phase transition.

It proposes maintaining a buffer of replayable states, estimating recovery curves over perturbation radii, computing a critical perturbation radius, prioritizing states near a boundary, and training around that critical radius.

This should remain the heart of the paper.

### 1.2 Recommended renaming

Avoid **BLR** as the main acronym. It is likely to be confused with Bayesian linear regression and may sound too tied to “level replay.” Use:

- **BGR**: Bifurcation-Guided Replay.
- **BGR-Suffix**: the robotics/suffix-state instantiation.
- **Bifurcation margin**: the critical perturbation radius.
- **Recovery curve**: success probability as a function of perturbation radius.
- **Recovery AUC**: area under the recovery curve.
- **Boundary sharpness**: slope or discrete drop around the critical radius.

“Bifurcation” is still a good term, but use it operationally rather than as a formal dynamical-systems claim unless a precise mathematical link is added. The paper can say:

> We use “bifurcation” operationally to denote a qualitative transition in policy outcomes, from likely success to likely failure, as perturbation magnitude increases.

This avoids reviewer pushback that the method does not prove a true dynamical bifurcation.

### 1.3 Changes needed for AAAI novelty

The original version is already elegant. To make it truly AAAI-grade, make these changes:

#### Change A — General replayable decision state, not just robotics suffix

Define the replay unit as:

\[
x \in \mathcal{X}
\]

where \(x\) is any state-like object from which the policy can be restarted or reconditioned. Examples:

| Domain | Replayable decision state \(x\) | Perturbation family |
|---|---|---|
| Procgen | Environment seed, initial layout, checkpointed episode state | entity positions, start state, observation noise, level parameters |
| MiniGrid / XLand-MiniGrid | grid layout + instruction + agent state | object/wall changes, start pose, inventory, instruction-preserving edits |
| Robotics | task, scene, instruction, robot configuration, object poses, suffix time | object pose offsets, robot joint noise, end-effector offsets, camera perturbations |
| Driving | traffic scenario + ego state | vehicle positions, speeds, weather, occlusion |
| Web agents | browser DOM + task prefix | DOM changes, page state, distractor elements |
| Coding agents | repository + failing/passing tests + terminal history | codebase state edits, bug-location distractors, dependency noise |
| LLM reasoning | partial solution state | premise/order perturbations, distractor lemmas, intermediate-step corruptions |

The AAAI paper should instantiate only the domains that can be rigorously evaluated. The broad table should be in the method section or motivation, not overpromised as experiments.

#### Change B — Define a perturbation operator, not just a scalar \(\sigma\)

A scalar perturbation radius is only meaningful relative to a perturbation family. Define:

\[
\delta \sim \mathcal{P}_x(\sigma), \quad \tilde{x} = T(x, \delta)
\]

where \(T\) applies a domain-valid perturbation and \(\mathcal{P}_x(\sigma)\) is a perturbation distribution with scale \(\sigma\). This avoids pretending that all domains share a single metric.

For each domain, the paper must specify:

- what perturbation family is used;
- what semantic invariants it preserves;
- what validity filters remove impossible states;
- how perturbation magnitudes are normalized.

#### Change C — Feasibility witness instead of robotics-only oracle

The original idea uses “oracle-recoverable” states. Generalize this to a **feasibility witness**:

\[
F(x) \in [0,1]
\]

where \(F(x)\) estimates whether the perturbed/replayed state is still solvable under a competent reference. Depending on domain, the witness can be:

- a scripted solver;
- a human/expert trajectory;
- an offline demonstration suffix;
- a best historical policy;
- an antagonist or teacher agent;
- a model-predictive controller;
- a deterministic environment validity check;
- best-of-\(N\) rollouts from a stronger policy.

This makes the method portable and prevents BGR from wasting training on broken states.

#### Change D — Active boundary estimation

Full recovery curves over every state and every training checkpoint are expensive. Add an efficient estimator:

1. Probe clean success at \(\sigma=0\).
2. Probe a coarse grid, e.g. \(\sigma \in \{\sigma_{min}, \sigma_{mid}, \sigma_{max}\}\).
3. Fit a monotone curve using isotonic regression or a constrained logistic curve.
4. Use posterior uncertainty to query the next radius near the current critical-radius estimate.
5. Cache and refresh curves asynchronously with a staleness score.

The paper should report both:

- **dense-curve evaluation** for final metrics;
- **active-curve estimation** for training-time efficiency.

#### Change E — Make “boundaryness” more precise

A state can be near the boundary in several ways. Distinguish:

- **State-level boundaryness:** the state has a critical radius in a target band, e.g. neither trivially zero nor huge.
- **Radius-level boundaryness:** for a selected state, the sampled perturbation radius is near the estimated critical radius.
- **Learning boundaryness:** training at that state/radius improves recovery for nearby radii.

The algorithm should sample both **states** and **perturbation radii**.

#### Change F — Add a paper-level objective

Define the goal as **Recovery Margin Expansion**:

\[
\max_\pi \; \mathbb{E}_{x \sim \mathcal{D}_{test}}[\text{RAUC}_\pi(x)]
\]

where:

\[
\text{RAUC}_\pi(x) = \int_0^{\sigma_{max}} R_\pi(x, \sigma) w(\sigma) d\sigma.
\]

Also measure shifts in:

\[
r_{\pi,\alpha}(x) = \sup\{\sigma: R_\pi(x,\sigma) \ge \alpha R_\pi(x,0)\}.
\]

This makes the empirical story clear: BGR should move critical-radius distributions rightward without reducing clean success.

---

## 2. AAAI targeting notes

### 2.1 Current venue facts to build around

As of June 1, 2026, the AAAI-27 main-track call page says the detailed Main Technical Track CFP is “Coming Soon,” but the AAAI-27 date page is live. It lists:

- AAAI-27 conference: February 16–23, 2027, Montréal, Canada.
- Author registration opens: June 17, 2026.
- Paper submission opens: June 24, 2026.
- Abstract deadline: July 21, 2026.
- Full paper deadline: July 28, 2026.
- Supplementary material and code deadline: July 31, 2026.
- Phase 1 rejection notification: September 24, 2026.
- Author feedback window: October 19–25, 2026.
- Final notification: November 30, 2026.

Because AAAI-27’s detailed main-track instructions are not yet fully posted, use AAAI-26 as the nearest official template for format and review expectations. AAAI-26 allowed up to 7 pages of technical content plus additional pages solely for references and the reproducibility checklist. AAAI-26 also required double-blind anonymization, separate supplementary material/code, and a reproducibility checklist.

### 2.2 Consequences for this paper

This must be a **7-page-core paper**, not a sprawling robotics systems report. The main paper must contain everything needed to evaluate the method. Supplementary material can contain additional algorithms, dense hyperparameter tables, expanded proofs, videos, and long ablations, but reviewers are not required to read supplements.

Main paper priorities:

1. Clear general problem statement.
2. One crisp method figure.
3. One formal definition block.
4. One pseudocode box.
5. One highly controlled synthetic/procedural experiment.
6. One robotics/VLA experiment showing real relevance.
7. Strong baselines and statistical reporting.
8. A limitations paragraph that preempts cost and terminology objections.

The paper should not spend excessive space on robot hardware details unless the real-robot experiment is central and strong. For AAAI, robotics should show breadth and practical stakes, while the intellectual contribution remains general.

### 2.3 AAAI reviewer expectations translated into design requirements

AAAI reviewers will likely score or discuss:

- significance of the problem;
- novelty relative to prior work;
- technical soundness;
- empirical quality;
- clarity;
- reproducibility;
- breadth of relevance to AI.

Design implications:

| Reviewer concern | Required answer in paper |
|---|---|
| “Is this just hard-example mining?” | No. It estimates state-conditioned success–failure transition radii and trains near local robustness boundaries, not simply high loss, failure, or TD error. |
| “Is this just PLR?” | No. PLR prioritizes levels by estimated future learning potential, often TD error. BGR prioritizes local perturbation geometry and critical radius. |
| “Is this just PAIRED/ACCEL?” | No. PAIRED/ACCEL generate or mutate environments using regret/frontier criteria. BGR can operate over existing replayable states and explicitly models recovery curves. |
| “Is this just adversarial training?” | No. Fixed-radius or worst-case adversarial perturbations often overfocus on impossible/hard perturbations. BGR estimates each state’s current margin and samples near it adaptively. |
| “Is curve estimation too expensive?” | Use active boundary estimation, cache/staleness, and report wall-clock/sample overhead. |
| “Does it preserve clean performance?” | Include nominal success and clean-regression metrics in every table. |
| “Does robotics make it too narrow?” | Include procedural RL and present robotics as a showcase. |
| “Are results reproducible?” | Open code plan, fixed seeds, compute budget, hyperparameter ranges, confidence intervals, non-cherry-picked tasks. |

### 2.4 AAAI compliance checklist to satisfy while building

The eventual submission should be ready to answer yes to:

- all assumptions and restrictions are stated clearly;
- all novel claims are stated formally where possible;
- proofs or proof sketches are supplied for theoretical claims;
- datasets/benchmarks are cited and publicly available where possible;
- metrics are formally defined and motivated;
- number of runs/seeds is stated;
- variation/confidence is reported;
- significance tests are used for main claims;
- hyperparameter ranges and final values are listed;
- code required for experiments is in a supplement or promised for public release;
- compute infrastructure is documented;
- any AI assistance in manuscript preparation is documented if required by final policy;
- paper is anonymized, including videos and supplementary material.

---

## 3. Paper identity

### 3.1 Recommended title options

Best main title:

> **Bifurcation-Guided Replay: Learning at the Success–Failure Boundary of Decision Policies**

More technical option:

> **Bifurcation-Guided Replay for Expanding Policy Recovery Margins**

More AAAI-general option:

> **Learning Where Policies Break: Replay at Success–Failure Bifurcation Boundaries**

More robotics-heavy option, not recommended as main AAAI title:

> **Bifurcation-Guided Suffix Replay for Expanding Robot Policy Recovery Margins**

Use the first or second. The first is memorable and general. The second is safer if reviewers dislike “bifurcation.”

### 3.2 Main claim

The main claim should be:

> BGR improves robust generalization of sequential decision policies by adaptively identifying replayable states whose local recovery curves exhibit a success–failure transition near the current policy’s competence boundary, and by training on perturbations sampled near that transition.

Do **not** claim:

- formal dynamical-system bifurcations unless proven;
- certified robustness;
- universal optimality of boundary replay;
- full generality across all listed domains;
- robotics deployment safety.

### 3.3 Core contributions

The paper should state 3–4 contributions:

1. **Bifurcation replay formalism.** A general framework for replayable decision states with perturbation families, recovery curves, critical radii, recovery AUC, and boundary sharpness.
2. **Efficient BGR algorithm.** A practical method that estimates critical radii using active probing and prioritizes state/radius pairs based on solvability, boundaryness, sharpness, uncertainty, and diversity.
3. **Recovery-margin objective and metrics.** Evaluation based on recovery AUC and critical-radius distribution shifts, not just clean return/success.
4. **Empirical validation across domains.** Controlled procedural RL plus robot manipulation/VLA suffix recovery, showing BGR improves robustness and sample efficiency without sacrificing nominal success.

Optional fifth contribution if there is room:

5. **Robotics instantiation.** BGR-Suffix, a concrete procedure for suffix-state replay in manipulation policies with oracle/expert feasibility witnesses.

### 3.4 One-paragraph abstract draft

> Policies trained on nominal starts often fail under small deviations from the states encountered during training. Existing curriculum and level-replay methods prioritize difficult or high-regret levels, but they do not explicitly identify where a policy’s behavior transitions from likely success to likely failure. We introduce **Bifurcation-Guided Replay (BGR)**, a general replay method for sequential decision policies. For each replayable decision state, BGR estimates a recovery curve measuring success probability under perturbations of increasing scale, derives a critical radius at which performance crosses a success threshold, and samples training perturbations near this state-conditioned boundary. To make boundary estimation practical, BGR uses active probing, monotone curve fitting, uncertainty-aware refresh, and diversity-preserving replay. Across procedurally generated RL tasks and language-conditioned robot manipulation suffix recovery, BGR improves recovery AUC and shifts critical-radius distributions toward larger margins compared with uniform replay, prioritized level replay, fixed-radius perturbation training, and failure-only replay, while preserving clean success. These results suggest that replaying the edge of policy competence is an effective and general principle for expanding robust decision-state recovery.

### 3.5 “AAAI first page” story

The first page must communicate:

1. **Problem:** Policies pass benchmark starts but have narrow basins of recovery; this hurts generalization and reliability.
2. **Gap:** Existing curricula use difficulty/regret/error, but not explicit local success–failure geometry.
3. **Idea:** Estimate where each replayable state changes from solvable to failing as perturbation scale increases.
4. **Method:** Replay near that state-conditioned transition.
5. **Result:** Wider recovery margins faster than uniform/hard/failure replay, in both procedural RL and robotics.

Suggested opening hook:

> A policy can solve a task and still be brittle: a minor shift in object pose, grid layout, browser state, or trajectory prefix may move it outside its recovery basin. Standard success metrics hide this geometry. We ask a different question: for each replayable state, how far can the current policy be perturbed before success becomes failure?

---

## 4. Formal problem specification

### 4.1 Sequential decision setting

Let a decision problem be a Markov decision process or partially observable decision process:

\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \Omega, O)
\]

or, more generally, an interactive environment with observations, actions, transitions, and a terminal success predicate.

A policy \(\pi_\theta(a_t \mid h_t)\) maps history \(h_t\) or observation \(o_t\) to actions.

Define terminal success:

\[
Y(\tau) \in \{0,1\}
\]

for rollout \(\tau\). For reward-return domains, success can be thresholded return, normalized return, or task completion.

### 4.2 Replayable decision state

A **replayable decision state** is any object \(x \in \mathcal{X}\) from which evaluation/training can restart or resume.

Examples:

- environment seed and initial state;
- checkpointed simulator state;
- task plus current observation and instruction;
- robot suffix state from a successful demonstration;
- browser state plus task prefix;
- repository state plus terminal/test history.

Each \(x\) has metadata:

\[
x = (z, h, m)
\]

where:

- \(z\) is environment/task identity;
- \(h\) is the decision context or history;
- \(m\) is metadata such as timestep, source trajectory, task family, or validity constraints.

### 4.3 Perturbation family

For each replayable state, define a domain-valid perturbation operator:

\[
\tilde{x} = T(x, \delta), \quad \delta \sim \mathcal{P}_x(\sigma)
\]

where \(\sigma \ge 0\) controls perturbation magnitude.

Requirements for \(T\):

1. **Semantic preservation:** The perturbed state should preserve the underlying task unless intentionally measuring semantic robustness.
2. **Validity:** The perturbed state should remain physically/logically valid.
3. **Measurability:** \(\sigma\) should have an interpretable normalized scale.
4. **Replayability:** The environment can reset/resume from \(\tilde{x}\).

For robotics:

- object pose offset in centimeters;
- end-effector pose offset;
- joint-angle noise;
- camera pose/lighting perturbation;
- distractor-object perturbation.

For MiniGrid:

- start pose offset;
- object position offset preserving reachability;
- distractor placement;
- wall gap shift;
- instruction-preserving edits.

For Procgen:

- entity spawn position changes;
- layout parameter perturbation;
- observation noise/crop;
- initial velocity/state perturbation if supported.

### 4.4 Recovery curve

For policy \(\pi\), replay state \(x\), and perturbation scale \(\sigma\), define:

\[
R_\pi(x, \sigma) = \mathbb{P}_{\delta \sim \mathcal{P}_x(\sigma), \tau \sim \pi(\cdot \mid T(x,\delta))}[Y(\tau)=1].
\]

This is the **recovery curve**. It measures the probability that the policy successfully recovers/completes the task after perturbation scale \(\sigma\).

Expected shape:

- easy state: high \(R\) for large \(\sigma\);
- impossible state: low \(R\) even at \(\sigma=0\);
- boundary state: high \(R\) at small \(\sigma\), sharp drop near a critical radius.

### 4.5 Critical radius / bifurcation margin

For threshold \(\alpha \in (0,1)\), define the state-conditioned critical radius:

\[
r_{\pi,\alpha}(x) = \sup \{ \sigma : R_\pi(x,\sigma) \ge \alpha R_\pi(x,0) \}.
\]

If clean success is close to 1, this is approximately the largest perturbation radius at which success remains at least \(\alpha\).

Alternative threshold for domains with noisy clean success:

\[
r_{\pi,\alpha}^{abs}(x) = \sup \{ \sigma : R_\pi(x,\sigma) \ge \alpha \}.
\]

Recommended default:

- report \(r_{80}\) and \(r_{50}\);
- train with \(\alpha=0.8\) or \(0.7\), tuned on validation;
- use \(r_{50}\) for sharper “transition” visualization.

### 4.6 Recovery AUC

Define:

\[
\text{RAUC}_\pi(x) = \frac{1}{\sigma_{max}} \int_0^{\sigma_{max}} R_\pi(x,\sigma) d\sigma.
\]

In practice, approximate with trapezoidal integration over a dense evaluation grid.

This is the primary robustness metric because it uses the entire curve and is less brittle than a single critical radius.

### 4.7 Boundary sharpness

Define boundary sharpness around \(r\) as:

\[
B_\pi(x) = -\left.\frac{dR_\pi(x,\sigma)}{d\sigma}\right|_{\sigma=r_{\pi,\alpha}(x)}
\]

or a finite difference:

\[
B_\pi(x) = \frac{R_\pi(x, r-\Delta) - R_\pi(x, r+\Delta)}{2\Delta}.
\]

Interpretation:

- high sharpness means a clear success–failure transition;
- low sharpness means gradual degradation or noisy estimates;
- high sharpness states may be more informative for margin expansion;
- extremely sharp states may also indicate hidden discontinuities or invalid perturbations, so pair sharpness with feasibility checks.

### 4.8 Feasibility witness

Define:

\[
F(x, \sigma) = \mathbb{P}_{\delta \sim \mathcal{P}_x(\sigma), \tau \sim \pi^*(\cdot \mid T(x,\delta))}[Y(\tau)=1]
\]

where \(\pi^*\) is a teacher/oracle/reference. In practice, \(F\) may be estimated by:

- scripted solver success;
- expert demonstration availability;
- best historical policy success;
- environment validity checks;
- best-of-\(N\) rollouts;
- human teleoperation in a small robotics subset.

BGR should prefer state/radius pairs where \(F\) is high and \(R_\pi\) is near threshold. This means the state is learnable: current policy fails near the boundary, but a competent policy can still succeed.

### 4.9 BGR training objective

The ideal objective is to improve recovery margins over a target distribution \(\mathcal{D}\):

\[
\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}}[\text{RAUC}_{\pi_\theta}(x)] - \lambda \, \mathcal{L}_{clean}(\theta)
\]

where \(\mathcal{L}_{clean}\) penalizes degradation in nominal performance.

Because RAUC is expensive to optimize directly, BGR approximates it through adaptive replay:

\[
(x, \sigma) \sim q_\theta(x,\sigma) \propto U_\theta(x,\sigma)
\]

where \(U\) is a utility score high near feasible success–failure boundaries.

---

## 5. Method: Bifurcation-Guided Replay

### 5.1 High-level algorithm

BGR maintains a buffer of replayable decision states. For each state, it estimates the recovery curve under a current or recent policy. It derives a critical radius and uncertainty. During training, it samples state/radius pairs near the estimated boundary, applies perturbations at those radii, and updates the policy using the domain’s learning objective.

Core loop:

1. Collect or generate replayable states.
2. Filter for feasibility and clean solvability.
3. Estimate recovery curves or critical radii.
4. Score states by bifurcation utility.
5. Sample a state according to utility.
6. Sample a perturbation radius near its critical radius.
7. Train on the resulting perturbed rollout or teacher recovery.
8. Periodically refresh curve estimates as the policy changes.

### 5.2 Buffer item schema

Each buffer item should store:

```python
LevelRecord:
    id: str
    domain: str
    task_id: str
    state_ref: Any                 # seed, simulator checkpoint, serialized state, suffix state
    source: str                    # demo, rollout, generator, replay, heldout, etc.
    source_return: float
    clean_success_hat: float
    feasibility_hat: float
    perturbation_family: str
    sigma_grid: list[float]
    trials: dict[float, int]
    successes: dict[float, int]
    recovery_curve_hat: list[float]
    recovery_curve_ci: list[tuple[float, float]]
    r_alpha_hat: float
    r_alpha_ci: tuple[float, float]
    sharpness_hat: float
    uncertainty_hat: float
    diversity_embedding: np.ndarray
    last_evaluated_step: int
    priority: float
    replay_count: int
```

For robotics, `state_ref` can be a simulator state snapshot plus task instruction and optional expert suffix trajectory.

For Procgen/MiniGrid, `state_ref` can be a seed plus saved environment state or generation parameters.

### 5.3 Recovery-curve estimation

#### 5.3.1 Dense estimator for evaluation

For final reporting, use a fixed grid:

\[
\Sigma_{eval} = \{0, \sigma_1, \ldots, \sigma_K\}
\]

Run \(n\) rollouts per state/radius. Estimate:

\[
\hat{R}(x,\sigma_k)=\frac{s_k}{n_k}.
\]

Use Wilson or bootstrap confidence intervals.

Recommended dense grids:

- MiniGrid: 7–11 perturbation magnitudes normalized to environment edit scale.
- Procgen: 5–9 magnitudes depending on reset/edit support.
- LIBERO robotics: 7 radii such as 0, 0.5 cm, 1 cm, 2 cm, 3 cm, 5 cm, 7.5 cm, normalized by object/task family.

#### 5.3.2 Training-time active estimator

For training, use an active estimator to reduce overhead.

Initialize with:

\[
\Sigma_{init} = \{0, \sigma_{min}, \sigma_{mid}, \sigma_{max}\}.
\]

After each batch of rollouts:

1. Fit a monotone decreasing curve.
2. Estimate \(r_\alpha\).
3. Compute uncertainty over \(r_\alpha\).
4. Query the next \(\sigma\) where information gain is highest, typically near \(\hat{r}_\alpha\).

Two implementation options:

**Option 1: Isotonic regression + bootstrap**

- Use empirical \(\hat{R}\) values.
- Enforce monotonicity using PAVA.
- Bootstrap Bernoulli outcomes to get uncertainty over \(r_\alpha\).
- Robust to weird curve shapes.

**Option 2: Logistic curve + Beta-binomial posterior**

Fit:

\[
R(\sigma) = c + \frac{a-c}{1 + \exp((\sigma-b)/t)}
\]

where:

- \(a\) approximates clean success;
- \(c\) approximates floor success;
- \(b\) is transition midpoint;
- \(t\) is temperature/slope.

Use beta-binomial likelihood or maximum likelihood with bootstrapped CIs. This gives smooth estimates of \(r\) and sharpness.

Recommended for the paper:

- Use isotonic regression as the default because it is simple and assumption-light.
- Use logistic fits only for visualization and sharpness estimates.
- Include an appendix ablation comparing both.

### 5.4 Priority score

BGR priority should be a product or log-linear combination of interpretable factors.

First define gates:

\[
G(x) = \mathbf{1}[\hat{R}(x,0) \ge \rho_{clean}] \cdot \mathbf{1}[\hat{F}(x,\hat{r}_\alpha) \ge \rho_{feas}] \cdot \mathbf{1}[N_x \ge N_{min}]
\]

where:

- \(\rho_{clean}\): minimum clean success, e.g. 0.5 or 0.7;
- \(\rho_{feas}\): minimum feasibility, e.g. 0.7;
- \(N_x\): number of observations used for curve estimate.

Then define soft components:

#### Boundary utility

Prefer states whose critical radius is in a trainable target band:

\[
U_{boundary}(x) = \exp\left(-\frac{(\hat{r}_\alpha(x)-r_{target})^2}{2\tau_r^2}\right)
\]

where \(r_{target}\) can be set to the current lower quantile of radii or a domain-specific target.

Alternative if target band is not known:

\[
U_{boundary}(x)= \sigma_g\left(\frac{\hat{r}_\alpha(x)-r_{min}}{\tau}\right) \cdot \sigma_g\left(\frac{r_{max}-\hat{r}_\alpha(x)}{\tau}\right)
\]

where \(\sigma_g\) is a sigmoid. This downweights states that are too brittle or already too robust.

#### Sharpness utility

\[
U_{sharp}(x)=\text{clip}\left(\frac{\hat{B}(x)}{B_{median}+\epsilon}, u_{min}, u_{max}\right)
\]

This prefers clear boundaries but caps extremes.

#### Uncertainty utility

\[
U_{unc}(x)=\left(\hat{\mathrm{Var}}[r_\alpha(x)] + \epsilon\right)^\beta
\]

This supports exploration and curve refinement.

#### Learning-progress utility

Track whether past BGR updates improved the state’s curve:

\[
U_{lp}(x)=\max(0, \text{RAUC}_{t}(x)-\text{RAUC}_{t-k}(x)) + \epsilon
\]

This can be an optional ablation. It may leak evaluation cost if overused.

#### Diversity utility

Let \(e(x)\) be an embedding of task/state metadata. Penalize overrepresented clusters:

\[
U_{div}(x) = \frac{1}{\sqrt{1 + C(cluster(x))}}
\]

or use determinantal point process sampling / k-means stratification in appendix.

#### Staleness utility

If a record has not been evaluated for many training steps:

\[
U_{stale}(x)=1+\eta \cdot \min\left(1, \frac{t - t_{last}(x)}{T_{refresh}}\right)
\]

This prevents stale underestimates from dominating.

#### Cost utility

If replaying a state is expensive:

\[
U_{cost}(x)=\frac{1}{\sqrt{c(x)+\epsilon}}
\]

Useful for robotics where resets can vary in cost.

#### Final priority

Recommended final score:

\[
P(x) = G(x) \cdot U_{boundary}(x) \cdot U_{sharp}(x)^\lambda \cdot U_{unc}(x)^\beta \cdot U_{div}(x)^\gamma \cdot U_{stale}(x)^\eta \cdot U_{cost}(x)^\kappa.
\]

In implementation, normalize priorities over the buffer with a temperature:

\[
q(x)=\frac{P(x)^{1/T}}{\sum_{x'} P(x')^{1/T}}.
\]

Add \(\epsilon\)-uniform mixing:

\[
q_{mix}(x) = (1-\epsilon)q(x)+\epsilon/|\mathcal{B}|.
\]

This avoids replay collapse.

### 5.5 Perturbation-radius sampling

After selecting state \(x\), sample \(\sigma\) from a boundary-centered mixture:

\[
q(\sigma \mid x) = p_b \, \mathcal{N}_{trunc}(\hat{r}_\alpha(x), s_r^2) + p_e \, q_{easy}(\sigma) + p_h \, q_{hard}(\sigma) + p_0 \, \delta_0.
\]

Recommended defaults:

- \(p_b=0.6\): boundary perturbations;
- \(p_e=0.15\): slightly easier than boundary;
- \(p_h=0.15\): slightly harder than boundary;
- \(p_0=0.10\): clean replay to preserve nominal skill.

For easier/harder:

\[
\sigma_{easy} \sim \mathcal{N}_{trunc}(0.7\hat{r}, s^2), \quad \sigma_{hard} \sim \mathcal{N}_{trunc}(1.3\hat{r}, s^2).
\]

This mixture is important. Pure boundary-only replay can overfit to a narrow band and may not preserve clean performance.

### 5.6 Training update options

BGR is a sampling and curriculum mechanism, not a single learning algorithm. The paper should define BGR as wrapping a base learner.

#### RL version

For on-policy RL such as PPO:

1. sample \(x,\sigma\);
2. reset environment to \(T(x,\delta)\);
3. rollout policy;
4. update PPO using collected trajectories.

Use same total environment steps across baselines.

#### Off-policy RL version

For off-policy algorithms:

1. sample state/radius;
2. collect transitions;
3. add to replay buffer with optional importance weights;
4. update from buffer.

Need to distinguish BGR’s state sampler from standard transition replay.

#### Imitation/recovery version

For robotics/VLA:

1. sample suffix state \(x\) and perturbation radius \(\sigma\);
2. apply perturbation;
3. generate or use an expert/teacher recovery trajectory \(\tau^*\);
4. fine-tune with behavioral cloning or DAgger-style updates;
5. optionally mix clean demonstrations.

Loss:

\[
\mathcal{L} = \mathcal{L}_{BC}^{boundary} + \lambda_{clean}\mathcal{L}_{BC}^{clean} + \lambda_{KL}D_{KL}(\pi_\theta || \pi_{base}) + \lambda_{RL}\mathcal{L}_{RL}.
\]

If no teacher is available, use sparse success RL or best-of-N successful recoveries.

#### Hybrid version

For robot manipulation in simulation:

- BC on oracle recovery trajectories;
- sparse-success RL from perturbed suffix states;
- clean demo regularization;
- optional Q-filter or advantage filter to avoid imitating poor teacher actions.

### 5.7 Pseudocode

```text
Algorithm 1: Bifurcation-Guided Replay (BGR)
Inputs:
  base policy pi_theta
  replayable-state source S
  perturbation family P_x(sigma)
  success predicate Y
  feasibility witness F
  alpha threshold
  base learner Update
  buffer B

Initialize B with replayable states x from S
For each x in B:
    Estimate clean success R_hat(x, 0)
    Estimate feasibility F_hat(x)
    If passes minimal gates:
        Probe initial radii Sigma_init
        Fit monotone recovery curve
        Estimate r_alpha(x), sharpness B(x), uncertainty U(x)

For training iteration t = 1 ... T:
    For each data collection episode j = 1 ... M:
        Sample x ~ q_B(x) proportional to BGRPriority(x)
        Sample sigma ~ q(sigma | x), centered near r_alpha(x)
        Sample perturbation delta ~ P_x(sigma)
        Reset/resume at x_tilde = T(x, delta)
        Collect rollout or teacher recovery data D_j
    theta <- Update(theta, D_1 ... D_M)

    Periodically:
        Add newly discovered successful states to B
        Refresh stale curve estimates using active radius probes
        Recompute priorities

Return pi_theta
```

### 5.8 Computational complexity

Let:

- \(N\): buffer size;
- \(K\): number of perturbation radii in dense evaluation;
- \(m\): rollouts per radius;
- \(A\): active probes per refresh;
- \(C\): cost per rollout.

Dense full-curve estimation costs:

\[
O(NKmC).
\]

Active refresh costs:

\[
O(N_{refresh}AmC), \quad A \ll K.
\]

Training rollout cost is matched to baselines by equal environment-step budgets. The extra overhead is curve probing. The paper should report overhead as:

- additional rollouts spent on probing;
- wall-clock ratio vs uniform replay;
- performance per environment step;
- performance per wall-clock hour if feasible.

### 5.9 Nonstationarity handling

Policy changes make old recovery curves stale. Use:

- `last_evaluated_step` metadata;
- staleness boosting;
- periodic refresh of top-priority states;
- random refresh of a small buffer subset;
- decay of confidence over training steps.

Example confidence decay:

\[
\mathrm{Var}_t(r) = \mathrm{Var}_{last}(r) + \zeta(t-t_{last}).
\]

Refresh policy:

- every \(T_{refresh}\) learner updates, probe top \(M_{top}\) priority states;
- also probe \(M_{rand}\) random states;
- update curves with new Bernoulli outcomes;
- refit monotone curves.

---

## 6. Robotics instantiation: BGR-Suffix

### 6.1 Role in the paper

Robotics should be the high-impact demonstration that BGR is useful in a realistic setting where recovery matters. It should not be the only setting unless experiments are extremely strong.

Frame robotics as:

> A natural instance of replayable decision states is a mid-trajectory suffix state in a robot manipulation task. Successful demonstrations define feasible states, and perturbations reveal where the policy’s recovery basin ends.

### 6.2 Replayable state definition for robotics

For manipulation/VLA:

\[
x = (I, o_t, s_t, g, \tau_{t:T}^*)
\]

where:

- \(I\): language instruction;
- \(o_t\): image observation(s);
- \(s_t\): robot proprioceptive state;
- \(g\): task goal metadata if available;
- \(\tau_{t:T}^*\): optional expert suffix trajectory.

Sources:

- successful demonstration trajectories;
- successful policy rollouts;
- simulator-generated reset states;
- held-out task validation states.

### 6.3 Perturbation families

Use multiple perturbation families, but start with one cleanly defined family for main results.

Recommended main family:

**Object pose perturbation**

- Translate target object by \(\sigma\) centimeters in XY plane.
- Preserve reachability and avoid collisions using validity filters.
- Normalize \(\sigma\) by task workspace range.

Secondary families:

**End-effector offset**

- Perturb gripper pose relative to demonstration suffix.
- Useful for recovery from drift.

**Joint perturbation**

- Add noise to robot joint configuration.
- Useful for control-state recovery.

**Distractor perturbation**

- Move irrelevant objects or add distractors.
- Tests semantic/visual robustness.

**Camera/observation perturbation**

- Lighting/camera shifts.
- Use carefully: this may test perception rather than recovery.

### 6.4 Feasibility witness in robotics

Potential witnesses:

- original expert suffix if perturbation is small and can be transformed;
- scripted controller for simple tasks;
- simulator oracle planner;
- teleoperated recovery for small subset;
- best-of-N rollouts from a stronger or ensemble policy;
- validity check for reachability/collision plus clean expert success.

Recommended for first paper:

- use successful demonstrations as suffix-state source;
- use a scripted or demonstration-adjusted oracle where available;
- for tasks without oracle, use feasibility by reset validity plus best historical policy success;
- be explicit about this limitation.

### 6.5 Training objective for VLA/robotics

For OpenVLA/Octo/diffusion policy:

\[
\mathcal{L}_{robot} = \mathcal{L}_{boundary\_BC} + \lambda_{clean}\mathcal{L}_{clean\_BC} + \lambda_{KL}\mathcal{L}_{KL/base}.
\]

If online RL is available:

\[
\mathcal{L}_{robot} = \mathcal{L}_{BC} + \lambda_{RL}\mathcal{L}_{PPO/SAC} + \lambda_{clean}\mathcal{L}_{clean}.
\]

To keep the paper feasible, prefer BC-style recovery fine-tuning if reliable teacher suffixes exist. Add sparse RL only if already implemented.

### 6.6 Robotics experiments

Recommended benchmark:

- LIBERO, because it is language-conditioned and widely used for robot manipulation.
- Use OpenVLA-OFT or an accessible VLA baseline if compute permits.
- Also include a smaller diffusion policy or behavior-cloning transformer to avoid overfitting the paper to one huge model.

Experiment design:

- train/fine-tune on subset of tasks/suffix states;
- evaluate clean task success;
- evaluate recovery curves on held-out suffix states;
- evaluate perturbation-family transfer, e.g. train on end-effector offsets, test on object pose offsets;
- evaluate held-out task families.

Metrics:

- clean success rate;
- RAUC over perturbation radii;
- \(r_{80}\), \(r_{50}\) distribution;
- success at fixed perturbation radii;
- recovery sample efficiency;
- degradation from base policy;
- training compute overhead.

### 6.7 Real-robot optional experiment

A real-robot experiment is valuable but not required for AAAI if the general algorithmic evaluation is strong.

Minimal real-robot protocol:

1. Select one simple task: pick-place, drawer open, button press, or object insertion.
2. Collect nominal successful executions.
3. Reset to mid-trajectory states.
4. Apply calibrated small perturbations.
5. Compare recovery ordering between simulation and hardware.
6. Report whether BGR-trained policy improves physical recovery at 2–3 perturbation magnitudes.

Do not make the real-robot result the main claim unless enough trials are collected.

---

## 7. Non-robotics instantiations

### 7.1 Why non-robotics is necessary for AAAI

AAAI is broad AI. If the paper is only robotics/VLA, reviewers may push it toward CoRL/ICRA. To make it AAAI-relevant, include at least one domain where:

- the method is clearly about decision policies, not robot reset engineering;
- experiments are cheap enough for many seeds and ablations;
- PLR/PAIRED/ACCEL baselines are meaningful;
- the recovery-curve idea can be visualized cleanly.

### 7.2 Recommended primary non-robotics domain: MiniGrid / XLand-MiniGrid

MiniGrid is ideal for mechanistic validation:

- easy to reset states;
- easy to define perturbations;
- easy to visualize states and curves;
- cheap enough for ablations and statistics;
- known to support curriculum and level-replay experiments.

Potential setup:

- Tasks: DoorKey, KeyCorridor, MultiRoom, ObstructedMaze, BabyAI-style instruction tasks.
- Replayable state: environment seed + checkpointed partial trajectory state.
- Perturbation: start pose, object location, door/key position, distractor object placement, gap shift.
- Feasibility witness: shortest-path planner or reachability solver.
- Policy: PPO or recurrent PPO.
- Baselines: uniform replay, PLR, fixed-radius perturbation, failure-only replay, easy-to-hard curriculum.

Key result:

> BGR increases recovery AUC and held-out perturbation success with fewer interactions than baselines.

### 7.3 Recommended secondary non-robotics domain: Procgen

Procgen adds credibility because PLR is strongly associated with Procgen-like settings.

Caution: It may be harder to apply mid-episode perturbations unless environment states can be reset or mutated. Use Procgen if implementation support is available.

Possible setup:

- use levels/seeds as replayable states;
- perturb initial conditions or observation features;
- compute success/return degradation curves;
- compare against PLR under same PPO base learner.

If state-level perturbation is too hard, use level-generator parameter perturbations and call them “level perturbations” rather than suffix states.

### 7.4 Optional domain: web/coding agents

These domains could make the paper exciting, but they are risky for an AAAI deadline because evaluation is slower and perturbation validity is harder.

Use only as a small proof-of-concept or future-work section unless already implemented.

Potential web-agent instance:

- replayable state: browser state + task prefix;
- perturbation: distractor elements, form field order, DOM layout changes;
- success predicate: task completion;
- feasibility: original task remains solvable.

Potential coding-agent instance:

- replayable state: repository + failing tests + partial history;
- perturbation: distractor files, error message variants, dependency noise;
- success: tests pass;
- feasibility: reference patch exists.

These are conceptually nice but not necessary.

---

## 8. Experimental plan

### 8.1 Research questions

The paper should explicitly answer:

**RQ1.** Does BGR improve recovery robustness compared with uniform replay and difficulty/error-based replay?  
**RQ2.** Does BGR improve sample efficiency, i.e. recovery AUC per environment step or per recovery trajectory?  
**RQ3.** Does BGR preserve nominal clean-task performance?  
**RQ4.** Does BGR generalize to held-out perturbation families, states, tasks, or levels?  
**RQ5.** Is explicit bifurcation estimation necessary, or do simpler failure-only/hard-example methods perform similarly?  
**RQ6.** Is active boundary estimation accurate and cheaper than dense curve estimation?  
**RQ7.** In robotics, does BGR improve recovery margins for language-conditioned manipulation policies?

### 8.2 Experimental tiers

#### Tier 0: Diagnostic toy environment

Purpose: create undeniable visual intuition.

Environment:

- 2D navigation with obstacles, goal, and resettable mid-trajectory states; or
- MiniGrid custom layout with known shortest-path oracle.

Show:

- recovery curves for easy, impossible, and boundary states;
- BGR samples near the boundary;
- critical-radius distribution shifts right after training.

This can be one figure and one paragraph in the main paper, with details in supplement.

#### Tier 1: Procedural RL benchmark

Purpose: prove BGR is a general AI/RL algorithm.

Recommended environments:

- MiniGrid: DoorKey, KeyCorridor, MultiRoom, ObstructedMaze;
- optionally XLand-MiniGrid for more task diversity;
- optionally Procgen if reset/perturbation implementation is clean.

Base learner:

- PPO or recurrent PPO.

Replayable states:

- environment seeds and/or mid-episode checkpoint states from successful or partially successful trajectories.

Perturbations:

- start pose perturbations;
- object position perturbations with reachability filter;
- layout parameter perturbations;
- observation noise as secondary, not main.

Baselines:

1. PPO + uniform level/state replay.
2. PPO + PLR-style priority by TD error / learning potential.
3. PPO + failure-only replay.
4. PPO + fixed-radius perturbation training.
5. PPO + random-radius perturbation training.
6. PPO + BGR without uncertainty.
7. PPO + BGR without diversity.
8. PPO + BGR dense curves, if affordable, as an upper bound.

Metrics:

- clean success/return;
- held-out seed success;
- RAUC over held-out states;
- \(r_{80}\) and \(r_{50}\) distributions;
- success at standardized perturbation radii;
- area under learning curve;
- number of environment interactions to reach target RAUC;
- statistical confidence across seeds.

#### Tier 2: Robotics/VLA benchmark

Purpose: demonstrate usefulness in a realistic high-stakes setting.

Recommended benchmark:

- LIBERO with language-conditioned manipulation tasks.

Policies:

- OpenVLA-OFT or OpenVLA fine-tuning if compute permits;
- smaller diffusion policy / BC-transformer baseline for faster ablations;
- optionally Octo if implementation available.

Replayable states:

- successful demonstration suffixes;
- successful policy rollout suffixes;
- held-out suffix states for evaluation only.

Perturbations:

- object XY offset as primary;
- end-effector offset as secondary;
- object-pose vs end-effector transfer as generalization.

Baselines:

1. clean fine-tuning only;
2. uniform suffix replay;
3. random perturbation augmentation;
4. fixed-radius perturbation training;
5. failure-only replay;
6. oracle-recoverable failure replay without bifurcation priority;
7. BGR-Suffix.

Metrics:

- clean task success;
- recovery success at each radius;
- recovery AUC;
- \(r_{80}\) distribution;
- held-out task RAUC;
- perturbation-family transfer;
- number of teacher recovery trajectories used;
- wall-clock/compute overhead.

#### Tier 3: Optional real robot

Use only if feasible. It can be included as:

- one small table;
- one figure of sim-to-real recovery ranking;
- supplementary video.

Claim modestly:

> A small hardware study suggests that recovery-margin improvements transfer to physical perturbations.

Do not overstate.

### 8.3 Baseline definitions

Baselines must be fair and strong.

#### Uniform Replay

Sample replay states uniformly. Sample perturbation radii uniformly or use same perturbation distribution without boundary priority.

Purpose: isolate prioritization.

#### PLR-style priority

Prioritize states using TD error or value/advantage-based score. For robotics BC, adapt PLR to imitation loss or prediction error.

Purpose: compare against learning-potential replay.

#### Failure-only Replay

Prioritize states where the policy failed. This tests whether BGR is just failure mining.

Important: failure-only can get stuck on impossible states; include feasibility-filtered failure replay as a stronger baseline.

#### Fixed-radius perturbation training

Choose a fixed \(\sigma\) per domain. Train on perturbations at that radius.

Purpose: test adaptive radius selection.

#### Uniform random perturbation

Sample \(\sigma \sim \mathrm{Uniform}(0,\sigma_{max})\).

Purpose: test whether simple randomization is enough.

#### Curriculum over radius

Start with small \(\sigma\), increase over time.

Purpose: compare against manual/easy-to-hard curriculum.

#### ACCEL/PAIRED when applicable

Use if environment generation/mutation is available. If not, include a conceptual comparison and do not force an unfair baseline.

#### BGR ablations

- no sharpness term;
- no uncertainty term;
- no diversity term;
- no feasibility gate;
- dense curve vs active estimator;
- boundary radius vs random radius for same selected states;
- state priority only vs radius priority only.

### 8.4 Metrics in detail

#### Clean success

\[
S_{clean}=\mathbb{E}_{x \sim \mathcal{D}}[R_\pi(x,0)].
\]

Report clean performance for all methods. Any robustness improvement with clean collapse is suspect.

#### Recovery AUC

Primary robustness metric:

\[
\text{RAUC}=\mathbb{E}_{x \sim \mathcal{D}_{eval}}\left[\frac{1}{\sigma_{max}}\int_0^{\sigma_{max}}R_\pi(x,\sigma)d\sigma\right].
\]

Report with confidence intervals.

#### Critical radius

Report mean, median, and distribution of \(r_{80}\) and \(r_{50}\). Use histograms or violin plots.

#### Boundary-estimation error

For active estimator:

\[
\mathrm{Err}_r = |\hat{r}_{active} - \hat{r}_{dense}|.
\]

Also report active probes per state.

#### Sample efficiency

Area under RAUC learning curve:

\[
\text{AULC}_{RAUC}=\int_0^T \text{RAUC}_t dt.
\]

Or steps to reach a target RAUC.

#### Generalization

Test on:

- held-out states from seen tasks;
- held-out levels/seeds;
- held-out task families;
- held-out perturbation families;
- larger perturbation ranges than training.

#### Calibration

Check whether estimated \(R\) curves predict held-out success. Use Brier score or expected calibration error across radius bins if space permits.

### 8.5 Statistical reporting

Minimum standards:

- at least 5 random seeds for cheap RL domains;
- at least 3 seeds for expensive robotics simulation;
- bootstrap confidence intervals over evaluation states and seeds;
- paired tests where possible, e.g. Wilcoxon signed-rank over seeds/tasks;
- report exact environment-step budgets;
- report hyperparameter search ranges and selection criterion.

Avoid reporting only best seed or only final mean.

### 8.6 Figure plan

The paper can have excellent figures. Recommended main figures:

#### Figure 1 — Concept figure

Three recovery curves:

- easy/robust state: high curve;
- too-hard/impossible state: low curve;
- boundary state: sharp transition.

Show BGR sampling near the transition.

#### Figure 2 — Algorithm schematic

Pipeline:

state buffer → perturbation probes → recovery curve → critical radius/priority → boundary replay → policy update → wider margins.

#### Figure 3 — Critical-radius distribution shift

Histogram/violin of \(r_{80}\) before and after training for BGR vs uniform/fixed perturbation.

This is the money plot.

#### Figure 4 — Recovery AUC over training

Learning curve comparing BGR and baselines.

#### Figure 5 — Robotics recovery curves

LIBERO/VLA recovery success vs perturbation radius, with BGR above baselines.

#### Figure 6 — Clean success preservation

Bar/table showing clean task success stays stable.

Optional appendix figures:

- active estimator error vs probes;
- ablation priority components;
- qualitative examples of boundary states;
- failure taxonomy.

### 8.7 Main result table template

| Domain | Method | Clean success ↑ | RAUC ↑ | Median \(r_{80}\) ↑ | Held-out RAUC ↑ | Steps to target ↓ |
|---|---:|---:|---:|---:|---:|---:|
| MiniGrid | Uniform replay | | | | | |
| MiniGrid | PLR-style | | | | | |
| MiniGrid | Fixed \(\sigma\) | | | | | |
| MiniGrid | Failure-only | | | | | |
| MiniGrid | BGR | | | | | |
| LIBERO | Clean FT | | | | | |
| LIBERO | Uniform suffix | | | | | |
| LIBERO | Random perturb | | | | | |
| LIBERO | BGR-Suffix | | | | | |

Use bold only for statistically significant best values, and use underlining for non-significant ties.

---

## 9. Theory and analysis plan

The paper does not need heavy theory, but AAAI reviewers like formal clarity. Include one proposition or analysis subsection if possible.

### 9.1 Proposition 1: Boundary probes maximize information about a logistic recovery threshold

Assume recovery outcomes are Bernoulli with:

\[
R(\sigma)=\frac{1}{1+\exp((\sigma-r)/t)}.
\]

The Fisher information about \(r\) from an observation at \(\sigma\) is proportional to:

\[
R(\sigma)(1-R(\sigma))/t^2.
\]

This is maximized at \(R(\sigma)=0.5\), i.e. near the transition midpoint.

Use this to justify active probing near the boundary. If training uses \(r_{80}\), the information argument is not exact, but still supports probing around steep regions.

### 9.2 Proposition 2: Active bifurcation search reduces probe complexity

Assume \(R(\sigma)\) is monotone decreasing and the critical radius lies on a discrete grid of \(K\) radii. Binary search or active thresholding identifies the bracket for \(r_\alpha\) in \(O(\log K)\) radius probes under noiseless observations, compared with \(O(K)\) dense grid evaluation.

In noisy settings, repeated probes scale with confidence requirements. State this as a sketch rather than a strong theorem unless fully derived.

### 9.3 Proposition 3: Boundary replay improves local RAUC under a kernel influence model

Assume a training update at radius \(\sigma\) increases recovery at nearby radii according to a local kernel:

\[
\Delta R(\sigma') \propto k(|\sigma-\sigma'|) \cdot g(R(\sigma))
\]

where \(g(R)\) is high when outcomes are neither saturated success nor saturated failure. Then sampling near radii where \(R\) is near threshold yields greater expected RAUC improvement than sampling uniformly, assuming equal cost.

This provides a formal version of the intuition: training on already solved states has low gradient; training on impossible states has low useful signal; boundary states have maximum signal.

### 9.4 What not to prove

Avoid claiming:

- global optimality of BGR;
- convergence guarantees for arbitrary deep RL;
- certified robustness;
- true dynamical bifurcation unless formalized.

The theory should be modest and explanatory.

---

## 10. Related-work positioning

### 10.1 Core comparison table

| Area | What it does | Why it is related | BGR distinction |
|---|---|---|---|
| Prioritized Level Replay | Replays levels estimated to have high future learning potential, often using TD error | Closest level-replay baseline | BGR scores local perturbation recovery geometry and critical radii rather than TD error/return alone |
| PAIRED | Generates environments using regret between protagonist and antagonist | Learner-adaptive curriculum | BGR need not generate environments and directly estimates success–failure transition curves |
| ACCEL | Mutates high-regret levels to evolve curricula | Frontier of capability idea | BGR’s frontier is a perturbation-radius boundary around replayable states, not only environment complexity/regret |
| Domain randomization | Trains on broad randomized environment parameters | Robustness via diversity | BGR adaptively concentrates samples near each state’s current margin instead of sampling blindly |
| Adversarial training / robust RL | Trains against fixed or worst-case perturbations | Robustness and margin expansion | BGR uses state-conditioned critical radii and feasibility gates to avoid impossible/wasteful perturbations |
| Active learning / margin sampling | Samples uncertain boundary examples | Boundary intuition | BGR extends boundary sampling to sequential decision states and recovery curves, not static supervised labels |
| Self-paced curriculum RL | Adapts task difficulty to learner | Curriculum | BGR defines difficulty by perturbation recovery threshold and explicitly trains at success–failure transitions |
| Recovery learning / suffix replay | Trains policies from off-nominal states | Robotics instantiation | BGR provides a general priority/perturbation rule based on recovery-curve bifurcation |

### 10.2 PLR positioning

PLR is the nearest conceptual neighbor. It asks: which levels have high expected learning potential when replayed? BGR asks: how far can the current policy be perturbed from this replayable state before its success probability collapses?

Key language:

> PLR estimates a scalar replay priority from learning signals collected during ordinary rollouts. BGR instead constructs a local response curve around a replayable state and uses the curve’s critical radius as the replay object. This lets BGR choose not only which state to replay, but also which perturbation scale to train on.

Baseline implication:

- Use PLR-style TD-error priority in MiniGrid/Procgen.
- Use imitation-loss or failure-rate priority as PLR analog in robotics if TD error is unavailable.

### 10.3 PAIRED/ACCEL positioning

PAIRED and ACCEL are about environment design/generation. They try to create challenging but solvable environments. BGR can operate on generated environments, but does not require a generator. It can be layered on top of PLR, PAIRED, or ACCEL.

Key language:

> Regret-based environment design searches for tasks at the frontier of agent capability. BGR searches for perturbation radii at the frontier of recovery for each replayable state. The former changes the task distribution; the latter maps and expands local recovery basins.

Possible combo experiment:

- Use ACCEL-generated levels, then BGR perturbation replay within those levels.
- This is optional, not required.

### 10.4 Robust/adversarial RL positioning

Robust RL often uses adversarial perturbations or regularization against observation/state noise. BGR differs by focusing on learnable transition zones rather than fixed or worst-case perturbations.

Key language:

> Worst-case perturbation training can overemphasize states outside the current policy’s learnable region. BGR instead chooses perturbation scales close to the current success–failure transition, making robustness training adaptive to each state and policy checkpoint.

### 10.5 Active learning positioning

Active learning often samples uncertain examples near decision boundaries. BGR is inspired by boundary sampling but differs because:

- labels are rollout outcomes, not static class labels;
- the boundary is a curve over perturbation radii;
- actions alter future states;
- feasibility and clean-performance preservation matter;
- training data may be generated online.

### 10.6 CREST / recovery-margin positioning

If CREST is cited in the final paper, position it as an evaluation-style predecessor if accurate:

> CREST-style recovery curves reveal brittleness around successful trajectories. BGR turns this diagnostic object into a training curriculum: estimate the curve online, prioritize states near the critical radius, and fine-tune to expand recovery margins.

Do not assume all reviewers know CREST. Define the recovery curve self-containedly.

---

## 11. Paper outline and page budget

AAAI main body is tight. Suggested 7-page allocation:

### Page 1: Introduction

- Motivation: policies brittle beyond nominal starts.
- Gap: difficulty/regret/error does not capture local recovery boundaries.
- Idea: estimate success–failure transition radii.
- Contributions.
- Small concept figure.

### Page 2: Related work + setup

- Brief related work: PLR, UED/PAIRED/ACCEL, robust RL/adversarial training, active learning/curriculum, recovery learning.
- Formal problem definitions begin.

### Page 3: Method definitions

- Replayable decision state.
- Perturbation family.
- Recovery curve.
- Critical radius.
- RAUC.
- Feasibility witness.
- Boundary utility.

### Page 4: Algorithm

- BGR pseudocode.
- Active boundary estimation.
- Priority score.
- Radius sampling.
- Complexity note.

### Page 5: Procedural RL experiments

- Environment setup.
- Baselines.
- RAUC learning curve.
- Critical-radius distribution shift.
- Ablation summary.

### Page 6: Robotics/VLA experiments

- LIBERO/suffix setup.
- Policies and perturbations.
- Recovery curves.
- Clean success preservation.
- Held-out perturbation/task transfer.

### Page 7: Analysis, limitations, conclusion

- Active estimator accuracy/cost table.
- Failure modes or qualitative examples.
- Limitations: perturbation family choice, reset access, no certification.
- Conclusion.

### Supplementary material

- Full pseudocode variants.
- Proof sketches.
- Full hyperparameters.
- Full per-task results.
- Additional ablations.
- Robotics videos.
- Feasibility witness details.
- Anonymous code/data package.

---

## 12. Implementation specification

### 12.1 Repository structure

```text
bgr/
  bgr/
    buffers.py
    perturbations.py
    curve_estimators.py
    priorities.py
    samplers.py
    learners/
      ppo.py
      bc.py
      vla_finetune.py
    envs/
      minigrid_wrappers.py
      procgen_wrappers.py
      libero_wrappers.py
    eval/
      recovery_curves.py
      metrics.py
      statistics.py
    utils/
      seeding.py
      logging.py
      plotting.py
  configs/
    minigrid_bgr.yaml
    minigrid_baselines.yaml
    libero_bgr.yaml
    libero_baselines.yaml
  scripts/
    train_minigrid.py
    eval_minigrid_curves.py
    train_libero.py
    eval_libero_curves.py
    plot_main_figures.py
  tests/
    test_curve_estimators.py
    test_priority_scores.py
    test_perturbation_validity.py
    test_replay_buffer.py
  README.md
  LICENSE
```

### 12.2 Core interfaces

#### ReplayableState

```python
class ReplayableState(Protocol):
    id: str
    task_id: str
    domain: str
    metadata: dict

    def reset(self, env) -> None:
        ...
```

#### PerturbationFamily

```python
class PerturbationFamily(Protocol):
    name: str
    sigma_min: float
    sigma_max: float

    def sample(self, state: ReplayableState, sigma: float, rng) -> Perturbation:
        ...

    def apply(self, env, state: ReplayableState, perturbation: Perturbation) -> bool:
        """Returns valid=True if perturbation preserves constraints."""
        ...
```

#### CurveEstimator

```python
class CurveEstimator:
    def update(self, sigma: float, successes: int, trials: int) -> None:
        ...

    def fit(self) -> RecoveryCurve:
        ...

    def estimate_radius(self, alpha: float) -> RadiusEstimate:
        ...

    def next_probe(self, alpha: float, strategy: str = "info_gain") -> float:
        ...
```

#### PriorityScorer

```python
class PriorityScorer:
    def score(self, record: LevelRecord, global_stats: dict) -> float:
        ...
```

### 12.3 Logging requirements

Every training run should log:

- config hash;
- git commit hash;
- seed;
- environment version;
- policy architecture;
- base learner hyperparameters;
- perturbation family and radii;
- curve-estimation probes;
- per-state critical radius estimates;
- priority components, not only final priority;
- replay counts per state/task;
- clean success over time;
- RAUC over time;
- wall-clock time;
- GPU/CPU information.

Use Weights & Biases, TensorBoard, or simple JSONL. For anonymity, online logs should not identify authors/institutions.

### 12.4 Unit tests

Minimum tests:

1. Monotone curve estimator returns non-increasing curve.
2. Critical radius computed correctly on known synthetic curves.
3. Active estimator converges on synthetic logistic data.
4. Priority score downweights clean-failing states.
5. Priority score downweights infeasible states.
6. Perturbation validity filter rejects impossible MiniGrid/robot states.
7. Uniform mixing prevents zero-probability starvation.
8. RAUC calculation matches trapezoidal integration.

### 12.5 Compute plan

For MiniGrid:

- 5–10 seeds;
- 4–8 tasks;
- dense evaluation every fixed interval;
- active curve training probes separate from evaluation.

For robotics:

- 3 seeds if expensive;
- 10–20 tasks if possible;
- held-out suffix set;
- limited dense evaluation grid;
- smaller policy ablations before VLA.

Report final compute exactly. Do not hide curve-estimation cost.

---

## 13. Risk register and mitigations

### Risk 1: “Bifurcation” sounds mathematically overclaimed

Mitigation:

- Define it operationally.
- Use “success–failure transition” throughout.
- Include “critical radius” and “recovery margin” as precise terms.
- Avoid dynamical-systems claims unless proven.

### Risk 2: Curve estimation is too expensive

Mitigation:

- Active estimator.
- Cache and refresh.
- Report overhead.
- Show active estimator matches dense estimates with far fewer probes.

### Risk 3: Method is just hard-example mining

Mitigation:

- Include failure-only and hardest-state baselines.
- Show that BGR avoids impossible states and outperforms failure replay.
- Show state/radius boundary ablations.

### Risk 4: Method is just PLR with perturbations

Mitigation:

- Compare to PLR-style priority.
- Show BGR chooses different states/radii.
- Include analysis of priority correlation with TD error/regret.
- Demonstrate radius sampling is essential.

### Risk 5: Robotics-only framing weakens AAAI fit

Mitigation:

- Main method is domain-general.
- Include MiniGrid/Procgen.
- Put robotics as “BGR-Suffix instantiation.”

### Risk 6: Perturbation families are arbitrary

Mitigation:

- State semantic invariants.
- Use multiple perturbation families.
- Test held-out perturbation transfer.
- Report sensitivity to \(\sigma_{max}\).

### Risk 7: Clean success degrades

Mitigation:

- Clean replay mixture.
- KL/base regularization for VLA.
- Report clean success in every table.
- Early stop on validation clean success if needed.

### Risk 8: Results depend on oracle/teacher quality

Mitigation:

- Explicitly define feasibility witness.
- Include non-robotics domains with exact planners.
- Ablate witness strength.
- Report witness failure cases.

### Risk 9: BGR overfits to buffer states

Mitigation:

- Evaluate held-out states/tasks/radii.
- Diversity term.
- Buffer refresh with new successful states.
- Test held-out perturbation families.

### Risk 10: VLA fine-tuning compute too high

Mitigation:

- First validate with smaller policy.
- Use LoRA/efficient fine-tuning.
- Use fewer tasks but more rigorous curves.
- Keep OpenVLA result as strong final experiment if compute allows.

---

## 14. Milestones

### Milestone 1 — Formalization and toy proof

Deliverables:

- synthetic toy environment;
- recovery-curve estimator;
- critical radius and RAUC metrics;
- first BGR sampler;
- Figure 1 and Figure 3 prototypes.

Success criterion:

- BGR shifts critical-radius distribution right compared with uniform replay in toy/MiniGrid.

### Milestone 2 — MiniGrid full study

Deliverables:

- PPO baseline;
- uniform replay;
- PLR-style replay;
- failure-only replay;
- fixed/random perturbation replay;
- BGR with active estimator;
- dense held-out evaluation.

Success criterion:

- BGR beats strongest baseline on RAUC and median \(r_{80}\) with no clean success loss.

### Milestone 3 — Active-estimator validation

Deliverables:

- active vs dense estimation comparison;
- probe cost analysis;
- estimator error plots.

Success criterion:

- active estimator reaches acceptable radius error with substantially fewer probes than dense grid.

### Milestone 4 — Robotics simulation

Deliverables:

- LIBERO suffix-state extraction;
- perturbation/validity filters;
- baseline policy fine-tuning;
- BGR-Suffix fine-tuning;
- recovery-curve evaluation.

Success criterion:

- BGR-Suffix improves RAUC and \(r_{80}\) on held-out suffix states and does not reduce clean task success.

### Milestone 5 — Paper integration

Deliverables:

- main result figures;
- ablation table;
- related work comparison;
- AAAI-format draft;
- supplementary reproducibility package.

Success criterion:

- paper can be understood without supplement and all main claims have direct evidence.

---

## 15. Acceptance bar and claim calibration

### 15.1 Strong acceptance-level evidence

The paper is strong if it shows:

- BGR beats uniform, PLR-style, fixed-radius, and failure-only replay in MiniGrid/Procgen.
- BGR improves RAUC and critical-radius distributions, not just average return.
- BGR preserves clean success.
- Active estimation is efficient and accurate enough.
- Robotics/VLA shows the same qualitative margin expansion.
- Ablations confirm both state selection and radius selection matter.

### 15.2 Medium evidence

The paper is still viable if:

- MiniGrid results are strong;
- robotics results are moderate but clean;
- active estimator is clearly useful;
- PLR-style comparison is fair;
- claims are conservative.

Frame as a general replay principle with a promising robotics instantiation.

### 15.3 Weak evidence and fallback

If robotics results are weak:

- make robotics a qualitative/diagnostic section;
- emphasize procedural RL and algorithmic contribution;
- include failure analysis;
- avoid VLA-specific claims.

If PLR beats BGR on clean return but not RAUC:

- emphasize that BGR optimizes robustness/recovery margins, not raw return;
- show combined PLR+BGR if possible.

If BGR improves RAUC but costs too much:

- report performance per probing budget;
- show active estimator reduces overhead;
- position as useful when recovery robustness matters more than raw training speed.

---

## 16. Draft contribution paragraph

Use something like this in the paper:

> This work contributes a replay principle for robust sequential decision making. Rather than ranking replay states solely by error, regret, or failure, we estimate how success changes under controlled perturbations around each replayable state. The resulting recovery curve exposes a policy-specific critical radius: the perturbation scale at which behavior transitions from likely success to likely failure. BGR prioritizes feasible states with informative transition regions and samples perturbations near their estimated critical radii. This converts recovery-margin measurement into a training curriculum and provides a domain-general mechanism for expanding local policy robustness.

---

## 17. Draft intro skeleton

Paragraph 1:

- Policies can solve benchmark starts but fail after small deviations.
- This matters for agents, robots, games, web tasks, and embodied interaction.
- Nominal success hides local recovery geometry.

Paragraph 2:

- Existing curricula and replay methods use difficulty, TD error, regret, or environment generation.
- These are useful but do not answer: how far can this policy be perturbed from this state before it fails?

Paragraph 3:

- Introduce recovery curves and critical radius.
- The most useful training states are not merely hard; they lie near the success–failure boundary and remain feasible.

Paragraph 4:

- Present BGR.
- Estimate curves, actively probe boundaries, prioritize feasible boundary states, train near critical radii.

Paragraph 5:

- Empirical summary.
- Procedural RL + robotics/VLA.
- BGR improves RAUC, critical radius, and sample efficiency while preserving clean success.

Contribution bullets:

1. Formalization.
2. Algorithm.
3. Efficient estimator.
4. Experiments.

---

## 18. Draft method text snippets

### 18.1 Recovery curve definition text

> For a replayable decision state \(x\), we define a perturbation family \(\mathcal{P}_x(\sigma)\) indexed by magnitude \(\sigma\). Applying a perturbation \(\delta \sim \mathcal{P}_x(\sigma)\) yields a valid state \(T(x,\delta)\). The recovery curve of policy \(\pi\) at \(x\) is the probability of task success after resuming from \(T(x,\delta)\). This curve summarizes local robustness around \(x\): easy states remain successful at large \(\sigma\), impossible states fail even at \(\sigma=0\), and boundary states exhibit a sharp success–failure transition.

### 18.2 BGR priority text

> BGR prioritizes states that are cleanly solvable, feasible under a reference witness, close to the current recovery boundary, informative to refine, and diverse relative to recently replayed states. These components avoid two degenerate curricula: replaying states already solved at all perturbation scales and replaying impossible perturbations that provide little useful learning signal.

### 18.3 Robotics text

> Robot suffix states provide a natural testbed for BGR because they expose a mismatch between benchmark success and recovery robustness. A policy may complete tasks from nominal starts but fail when an object is displaced by a few centimeters at a mid-trajectory state. BGR-Suffix estimates this local recovery margin for demonstration-derived suffix states and prioritizes recoverable boundary perturbations for fine-tuning.

---

## 19. Suggested appendix contents

### Appendix A: Full algorithm details

- dense estimator;
- active estimator;
- priority formulas;
- hyperparameters.

### Appendix B: Theory sketches

- Fisher information for logistic recovery curves;
- active threshold search;
- kernel influence model.

### Appendix C: Environment details

- MiniGrid tasks;
- perturbation constraints;
- feasibility witness;
- state reset implementation.

### Appendix D: Robotics details

- LIBERO task list;
- suffix extraction;
- perturbation definitions;
- policy architectures;
- fine-tuning hyperparameters.

### Appendix E: Full results

- all tasks;
- all seeds;
- confidence intervals;
- significance tests;
- failure cases.

### Appendix F: Reproducibility checklist

- hardware;
- software versions;
- random seeds;
- code package structure;
- data availability;
- hyperparameter search ranges.

### Appendix G: Videos and qualitative examples

- boundary perturbation examples;
- success/failure rollouts;
- robotics recovery examples.

---

## 20. References and source map for the paper

This is not a final bibliography, but these works should be considered when drafting the related work.

### Core replay/curriculum/UED

- Minqi Jiang, Edward Grefenstette, Tim Rocktäschel. **Prioritized Level Replay.** ICML 2021. https://arxiv.org/abs/2010.03934
- Michael Dennis et al. **Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design / PAIRED.** NeurIPS 2020. https://arxiv.org/abs/2012.02096
- Jack Parker-Holder et al. **Evolving Curricula with Regret-Based Environment Design / ACCEL.** ICML 2022. https://proceedings.mlr.press/v162/parker-holder22a.html
- Sanmit Narvekar et al. **Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey.** JMLR 2020. https://arxiv.org/abs/2003.04960

### Robustness, margins, perturbation training

- Huan Zhang et al. **Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations.** NeurIPS 2020. https://arxiv.org/abs/2003.08938
- Josh Tobin et al. **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.** IROS 2017. https://arxiv.org/abs/1703.06907
- Gavin Ding et al. **MMA Training: Direct Input Space Margin Maximization through Adversarial Training.** ICLR 2020. https://arxiv.org/abs/1812.02637
- Yuancheng Xu et al. **Exploring and Exploiting Decision Boundary Dynamics for Adversarial Robustness.** CVPR 2023. https://arxiv.org/abs/2302.03015

### Benchmarks

- Karl Cobbe et al. **Leveraging Procedural Generation to Benchmark Reinforcement Learning / Procgen.** 2020. https://cdn.openai.com/procgen.pdf
- Maxime Chevalier-Boisvert et al. **Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks.** NeurIPS Datasets and Benchmarks 2023. https://openreview.net/forum?id=PFfmfspm28
- Bo Liu et al. **LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.** NeurIPS 2023. https://arxiv.org/abs/2306.03310

### VLA / generalist robot policies

- Moo Jin Kim et al. **OpenVLA: An Open-Source Vision-Language-Action Model.** CoRL 2024 / PMLR 2025. https://arxiv.org/abs/2406.09246
- Octo Model Team et al. **Octo: An Open-Source Generalist Robot Policy.** RSS 2024. https://arxiv.org/abs/2405.12213
- Kevin Black et al. **π0: A Vision-Language-Action Flow Model for General Robot Control.** 2024. https://arxiv.org/abs/2410.24164

### AAAI process sources used for this spec

- AAAI-27 conference page and timetable. https://aaai.org/conference/aaai/aaai-27/
- AAAI-27 Main Technical Track page, currently “Coming Soon” as of this spec. https://aaai.org/conference/aaai/aaai-27/call/main-technical-track/
- AAAI-26 Submission Instructions. https://aaai.org/conference/aaai/aaai-26/submission-instructions/
- AAAI-26 Main Technical Track CFP. https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/
- AAAI-26 Review Process. https://aaai.org/conference/aaai/aaai-26/review-process/
- AAAI-26 Reproducibility Checklist. https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/
- AAAI Publication Policies & Guidelines. https://aaai.org/aaai-publications/aaai-publication-policies-guidelines/

---

## 21. Final “build this paper” checklist

Before submission, the project should have:

- [ ] a crisp title using BGR, not BLR;
- [ ] a general formalism for replayable decision states;
- [ ] a precise perturbation-family definition for every domain;
- [ ] recovery curve, RAUC, and critical-radius metrics implemented;
- [ ] active boundary estimator implemented and validated;
- [ ] BGR priority score with gates, uncertainty, sharpness, and diversity;
- [ ] radius sampling near critical radii;
- [ ] at least one cheap procedural domain with full ablations;
- [ ] at least one robotics/VLA or manipulation domain;
- [ ] uniform, PLR-style, fixed-radius, random-radius, and failure-only baselines;
- [ ] clean-success preservation results;
- [ ] held-out state/task/perturbation generalization;
- [ ] confidence intervals and significance tests;
- [ ] compute overhead reporting;
- [ ] hyperparameter ranges and final values;
- [ ] anonymous code package or release plan;
- [ ] supplementary proofs/appendix;
- [ ] failure analysis;
- [ ] a limitations section that admits perturbation-family dependence and lack of certification.

---

## 22. Most important strategic advice

The paper should not sell “we found harder states.” It should sell a new object:

> the **policy-conditioned recovery bifurcation curve** around a replayable decision state.

That object gives the paper its novelty. Once the object is accepted, the algorithm follows naturally: estimate the curve, find the critical radius, sample states and perturbations near that boundary, and train to move the boundary outward.

The robotics story is powerful because suffix states make recovery margins concrete. But AAAI will be more receptive if the paper shows that BGR is a general principle for sequential decision policies and uses robotics as the most visually compelling demonstration.

