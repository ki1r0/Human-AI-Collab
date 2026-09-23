# Research Findings

## Research Question

Can a current action/world-action model identify a visible physical feasibility change, rather than merely replaying the demonstrated assembly order?

## Current Understanding

The repository already contains the required gearbox identities and several assembly sockets, but the five requested causal probes need an explicit paired mechanism contract. The MVP therefore separates three layers: (1) source CAD appearance references, (2) a shared parameter source for render/collision mutation geometry, and (3) an evaluator-only symbolic oracle used to validate the intended relation before Isaac calibration.

LingBot-VA's official base checkpoint can be deployed as an interface smoke test. It emits native action/video chunks, but it has not been adapted to this custom Franka/gearbox embodiment; a zero-shot run must not be reported as a physics-understanding result.

## Key Results

- The five-task pack has 10 evaluator domains (five HARD and five COMMUTABLE). The
  offline oracle and manifest separation tests pass; all 10 generated USD fixtures
  reference source assets and pass the source-geometry audit.
- LingBot-VA base is locally deployed in the isolated `lingbot-va` environment and
  emits a native Franka-layout chunk on an HCF-01/HARD observation. The reproducible
  fast smoke output is
  `pilot_12pair/outputs/lingbot_va_hcf01_hard_fast_lowres/`:
  `actions_0.pt` has shape `(1,30,4,20,1)`, is finite, and is non-degenerate;
  `demo.mp4` is readable.
- The official default inference configuration (224x320, text length 512, 5 video
  and 10 action steps) loaded successfully but was CPU-offload bound and did not
  produce output within 20 minutes on this host. The fast smoke changes only
  explicitly recorded runtime parameters and is not a scientific task score.

## Patterns and Insights

- Each pair keeps the goal and shown A→B order fixed while changing one visible mechanism.
- A/B-only terminal procedures make a reverse-order failure diagnostic; recovery policies need a separate track.
- Source STL dimensions are measured in millimetres and converted once. The mutation recipes require the same parameter source for visual opening and collision proxy.

## Lessons and Constraints

- Do not use the existing VLM video smoke output as robot action data.
- The EAI media mount is not present in this shell (`/media/sunsiliang/CoAI` is currently empty), so this pass cannot truthfully render or evaluate the 30 human video sets.
- The default host interpreter is Python 3.13 without torch; LingBot-VA is isolated in a Python 3.10 environment.
- The base checkpoint has no gearbox/Isaac action adapter; its native output is a
  deployment baseline, not evidence that it selected a physically valid assembly
  branch.

## Open Questions

- Which five USD mutation scenes pass Isaac swept-volume and contact calibration at the required robustness margin?
- Can a small converted teleoperation/simulation dataset map the five tasks into LingBot-VA's 30-channel action layout without leaking the variant label?
- Does base LingBot-VA produce a non-degenerate action chunk on a calibrated task
  observation before post-training? (The synthetic-image smoke answers only the
  interface part; real USD camera frames remain open.)

## Optimization Trajectory

The current proxy is structural: task-pack invariants first, native model deployment second, and calibrated simulator success third.
