# Review Weakness Response

This is an internal triage note for the weak-reject review. It tracks what is
now addressed in the manuscript versus what still requires new evidence.

## Current Acceptance Read

The paper is more defensible as an honest mechanism study, but it is not yet a
90%+ AAAI main-track accept. The main blocker is still evidence, not packaging:
there is no promoted positive independent benchmark and no learned-policy
OpenVLA/LIBERO improvement over both matched random and the official checkpoint.

## Reviewer Weaknesses

| Review concern | Current paper defense | Remaining gap | Priority |
| --- | --- | --- | --- |
| Robotics promise not delivered | Title, abstract, introduction, evidence table, OpenVLA table, and conclusion now frame OpenVLA/LIBERO as an audit rather than a robotics claim. The preregistered image-augmentation audit is included as negative. | Still no stable learned-policy win. Do not spend more compute on the same recipe family. | High |
| Tiny effect sizes hidden by p-values | Protocol section says effect sizes are primary and sign tests are consistency checks. Main text reports absolute RAUC/AULC/clean/r80 differences before p-values. | Synthetic and suffix effects remain small. They should stay scoped support. | High |
| Benchmarks constructed for BGR | Abstract, synthetic section, grid section, evidence contract, and limitations explicitly say synthetic/grid-margin establish mechanism rather than broad dominance. | Need a positive pre-existing benchmark before making stronger main-track claims. | High |
| Fragile/post-hoc variants | Suffix text explains the first boundary-heavy run undercovered and promotes only coverage-aware BGR with caveats. Grid learning-rate caveat remains in text. | Coverage-aware suffix still reads like a rescue variant. Keep it positioned as manipulation-style support only. | Medium |
| Incremental novelty | Related work and grid ablation emphasize the state-priority-only control: keeping BGR state priority but drawing radii uniformly loses the grid effect. | Novelty still depends on this ablation and the mechanism benchmark. A pre-existing win would make it stronger. | High |
| Proposition too strong | Proposition is renamed local boundary intuition and explicitly says it is not a convergence, global robustness, or margin-expansion theorem. | No theory upgrade planned unless there is time for a real guarantee. | Medium |
| Metrics favor BGR | Protocol states RAUC/AULC are author-defined summaries and reports median r80 disagreements directly. Suffix median-r80 caveat remains in abstract/table text/limitations. | Independent metrics still matter. Any promoted benchmark must avoid saturated or contradictory r80. | High |
| Feasibility witness is a hidden requirement | Problem setting has a dedicated feasibility-witness paragraph saying it is a real assumption, not free supervision. | Need deeper witness sensitivity only if we try to broaden claims. | Medium |
| Results dump/no intuition | Paper now has boundary-intuition and grid learning-curve figures, plus an evidence-contract table. | If page budget allows, one more compact plot for recovery curves across methods could help readability. | Medium |

## Immediate Paper Policy

- Do not promote FourRooms, Acrobot, Pendulum, CliffWalking, Taxi, MountainCar,
  CartPole, FrozenLake, or OpenVLA adaptation as positive evidence under
  current results.
- Keep FourRooms internal: the preregistered 4-seed screen has BGR and
  BGR-Coverage below uniform, fixed-radius, failure-only, TD-loss, and the
  state-priority/uniform-radius ablation, with saturated or contradictory
  median-r80 evidence.
- Keep Acrobot internal: the 4-seed diagnostic is non-saturated, but default
  BGR trails uniform and BGR-Coverage has only a +0.0016 RAUC edge with a
  2/1/1 paired split.
- Keep Pendulum internal: the 4-seed diagnostic has near-zero endpoints,
  saturated median r80, and failure-only replay above BGR.
- Keep CliffWalking in limitations only: default protocol saturates, and the
  harder undertrained protocol has BGR below uniform and all strong baselines.
- Do not add another authored toy benchmark unless it replaces, not expands,
  the current mechanism evidence.
- The next acceptance-moving result must be either a fixed-protocol positive
  pre-existing benchmark or a genuinely different preregistered learned-policy
  intervention that can beat both official and matched random.
