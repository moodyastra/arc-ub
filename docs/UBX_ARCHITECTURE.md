# UB-X architecture and acceptance contract

## Runtime data flow

1. `Observation` retains the exact 64x64 indexed-color grid and previous grid.
2. `MultiviewPerception` adds delta masks, 2x2 and 4x4 patch histograms, connected components, relations, geometry, trajectories, and an independent edge/UI color census.
3. `HypothesisBank` updates at most eight executable mechanic/goal hypotheses from verified transitions.
4. `TransitionGraph` stores exact states, transpositions, dead actions, return paths, and explored frontiers. It is authoritative over neural predictions.
5. `WorldModelSearch` scores directional and click candidates for progress, information gain, action cost, and risk. The first three actions are observed independently. Later plans contain three to eight actions when uncertainty permits.
6. The worker validates each real transition and interrupts a macro at its first meaningful divergence.

`GraphBaselineEngine` runs without PyTorch. `OfflineUBXEngine` adds a trained sparse model when `--model-path` is supplied. `codex_teacher` remains a development-only data source.

## Sparse model

The default model has a width of 512 and 12 blocks. Blocks 0 and 1 use dense gated feed-forward layers. The other ten contain eight routed experts plus one shared expert, with top-2 routed activation. Five of every six attention blocks are local; the sixth is global. Two-dimensional relative row/column biases preserve geometry. The default configuration is approximately 188M total parameters and 76M active parameters per pass.

Heads predict legal/action policy, a 64x64 click map, completion value, progress/reset/terminal/resource events, epistemic and transition uncertainty, object roles, goals, a latent visual delta, and eight future actions. `UBXModel.parameter_report()` reports total, estimated active, INT8, and INT4 sizes; export refuses artifacts at or above 1GB.

## Training boundary

Procedural data varies color, position, obstacles, control mode, UI, timers, delayed effects, hazards, and composed mechanics. `family_split()` holds out complete mechanic families rather than random seeds. Public game IDs and source constants are absent from model inputs.

Training is staged:

1. masked/delta-oriented representation learning;
2. cold-start policy/value imitation from exact simulator actions;
3. dense on-policy distribution distillation;
4. eight-trajectory group-relative policy optimization with separate value regression;
5. INT8 TorchScript packaging and size validation.

## Required release gates

- Final artifact below 1GB and no runtime downloads.
- At least 80% of held-out mechanic families improve before final tuning.
- No change may trade more than two percentage points of held-out completion for public-game gains.
- High-confidence transition predictions must reach 95% empirical agreement.
- A verified macro must abort before a second unexpected action.
- Competition mode uses one Arcade instance and one `make()` call per environment, never imports the viewer, and cannot instantiate Codex or Sol planners.

The current repository supplies these enforcement points and measurements. Passing the learned-performance gates depends on producing and evaluating trained checkpoints; the deterministic bootstrap is not represented as a trained model.
