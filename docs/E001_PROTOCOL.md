# E001 — Minimal Foraging Protocol

## Hypothesis

A policy constrained by the measured fruit-fly connectome topology can learn minimal foraging with better sample efficiency, robustness, or held-out generalization than matched rewired/random graph controls and conventional MLP/GRU policies.

## Environment

A bounded 2D arena contains a single fly and sparse food patches. Energy decays every step. Food restores energy and yields reward. The episode ends on starvation, collection of all food, or the step limit.

## Observation

E001 deliberately starts with compact engineered sensory features rather than pixels:

1. energy
2. normalized distance to nearest food
3. sine of relative food bearing
4. cosine of relative food bearing
5. normalized x position
6. normalized y position

This isolates sequential control from visual representation learning. Pixel/retina experiments are later phases.

## Actions

- forward
- turn left
- turn right
- idle

## Policies

- BioPolicy: measured directed connectome topology
- RewiredPolicy: degree-preserving rewired topology
- RandomGraphPolicy: node/edge/weight-matched sparse topology
- GRUPolicy
- MLPPolicy

## Training

Initial algorithm: PPO. All policies use identical environment budgets, evaluation seeds, and reward definitions. Compute/parameter budgets should be reported, not silently assumed equivalent.

## Evaluation

Minimum final experiment:

- 10 training seeds
- 100 held-out episodes per seed
- reward vs environment steps
- learning-curve AUC
- food collected
- survival time
- success rate
- unseen map evaluation
- observation-noise sweep
- edge/node ablation sweep

## Required controls

A biological-topology claim is not supported unless BioPolicy is compared with at least a degree-preserving rewired graph. Random sparse topology alone is insufficient because degree distribution and hubs can explain performance.

## E001 pass criterion

Proceed to E002 when the training/evaluation pipeline is reproducible and at least one policy reliably learns the task. A claim that biological topology is advantageous requires a statistically repeatable improvement across multiple seeds and matched controls.

## Non-claims

This experiment does not recreate a fruit fly's mind, consciousness, natural sensory processing, or biological learning rule. The connectome is used as an architectural constraint inside an engineered learning system.
