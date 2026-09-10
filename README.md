# FlyForage

A connectome-based fruit fly learning to explore, forage, and survive in simulated environments.

## Research question

Can an evolution-shaped fruit-fly connectome provide a more sample-efficient, robust, or generalizable control policy than conventional neural architectures in open-ended foraging tasks?

## E001 — Minimal Foraging

The first experiment is intentionally narrow:

- one simulated fly
- one bounded arena
- sparse food patches
- energy decreases over time
- food restores energy and yields reward
- the policy receives compact sensory observations
- the policy controls movement

The experiment compares the biological connectome topology against matched controls and standard neural policies.

### Baselines

- BioPolicy — MaleCNS-derived fixed topology
- RewiredPolicy — degree-preserving rewiring
- RandomGraphPolicy — matched sparse random graph
- GRUPolicy — recurrent conventional baseline
- MLPPolicy — feed-forward conventional baseline

### Primary metrics

- food collected per episode
- survival time
- reward vs environment steps
- learning-curve AUC
- steps to performance threshold
- held-out map generalization
- sensory-noise robustness
- node/synapse ablation robustness

## Roadmap

- E001: single-food minimal arena
- E002: multiple food patches
- E003: obstacles
- E004: food vs harmful stimuli
- E005: sensory noise
- E006: unseen map distributions
- E007: neuron/synapse ablation
- E008: exploration vs exploitation
- E009: memory-dependent foraging
- E010: predator and survival

## Scientific discipline

FlyForage is an experimental control-learning project, not a claim to reproduce a fruit fly mind or consciousness. Measured biological connectivity, engineered interfaces, learned parameters, and simulation assumptions must be documented separately.

## Status

Repository initialization in progress.
