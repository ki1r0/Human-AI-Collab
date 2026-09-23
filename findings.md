# Research Findings

## Research Question

Can a current action/world-action model identify a visible physical feasibility change, rather than merely replaying the demonstrated assembly order?

## Current Understanding

The repository already contains the required gearbox identities and several assembly sockets, but the five requested causal probes need an explicit paired mechanism contract. The MVP therefore separates three layers: (1) source CAD appearance references, (2) a shared parameter source for render/collision mutation geometry, and (3) an evaluator-only symbolic oracle used to validate the intended relation before Isaac calibration.

LingBot-VA's official base checkpoint can be deployed as an interface smoke test. It emits native action/video chunks, but it has not been adapted to this custom Franka/gearbox embodiment; a zero-shot run must not be reported as a physics-understanding result.

## Key Results

No model result yet. The five-task contract and offline oracle are being implemented first.

## Patterns and Insights

- Each pair keeps the goal and shown A→B order fixed while changing one visible mechanism.
- A/B-only terminal procedures make a reverse-order failure diagnostic; recovery policies need a separate track.
- Source STL dimensions are measured in millimetres and converted once. The mutation recipes require the same parameter source for visual opening and collision proxy.

## Lessons and Constraints

- Do not use the existing VLM video smoke output as robot action data.
- The EAI media mount is not present in this shell (`/media/sunsiliang/CoAI` is currently empty), so this pass cannot truthfully render or evaluate the 30 human video sets.
- The default host interpreter is Python 3.13 without torch; LingBot-VA is isolated in a Python 3.10 environment.

## Open Questions

- Which five USD mutation scenes pass Isaac swept-volume and contact calibration at the required robustness margin?
- Can a small converted teleoperation/simulation dataset map the five tasks into LingBot-VA's 30-channel action layout without leaking the variant label?
- Does base LingBot-VA produce a non-degenerate action chunk on a task observation before post-training?

## Optimization Trajectory

The current proxy is structural: task-pack invariants first, native model deployment second, and calibrated simulator success third.
