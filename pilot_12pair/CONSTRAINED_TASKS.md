# Five-task constrained action MVP

This is the first action-policy extension of the 12-pair plan. It is intentionally
five tasks, not the final twelve-pair benchmark. Every task uses repository parts and
keeps the demonstrated order `A -> B` fixed. Each has two rendered variants:

| Task | A then B | HARD reverse | COMMUTABLE reverse | Parts |
|---|---|---|---|---|
| HCF-01 | seat hub cover → retain M6 bolt | retained head cannot pass round hole | keyhole lobe admits retained head | casing top, small cover, M6 bolt |
| WSG-01 | seat washer → mount output gear | gear shroud covers washer seat | visible radial slot permits side insertion | output shaft, thin washer, output gear |
| KEY-01 | insert output key → mount output gear | gear covers only keyway entrance | side-open keyway stays reachable | output shaft, output key, output gear |
| CAS-01 | close casing → insert through-bolt | split lugs do not form a through-bore | captive pocket retains bolt during closure | casing base/top, M10 bolt |
| DOW-01 | close casing → insert locating dowel | split bore is not a stable dowel seat | open retaining slot holds dowel | casing base/top, dowel pin |

The exact measured source bounds and mutation parameters are in
`config/constrained_tasks.json`; the generated data-only scene recipes are under
`scenes/recipes/`. All lengths are authored in millimetres and converted once to
stage metres. The render/collision rule is strict: the same parameter source creates
the visible feature and its collision representation.

## Offline contract and expected result

Run:

```bash
python3 -m pilot_12pair.oracle.generate_task_manifests
python3 -m pilot_12pair.oracle.build_scene_recipes
python3 -m unittest pilot_12pair.tests.test_constrained_tasks -v
```

The offline oracle is only a contract check. It predicts `A>B` feasible for both
variants, `B>A` infeasible for HARD, `B>A` feasible for COMMUTABLE, and both single
actions incomplete. Isaac calibration must independently verify swept-volume,
contact, reachability, visibility, and terminal predicates before a model score is
reported.

## Research-design rationale

The task set follows three ideation checks. First, the problem-first statement is:
current action policies can finish a demonstrated assembly while replaying a procedure
that is invalid after a visible mechanism intervention. Second, boundary probing asks
the same policy to cross one mechanism boundary at a time: head clearance, lateral
access, keyway access, captive preload, and locating retention. Third, the simplicity
test keeps every domain at two named actions and four terminal candidates, so an error
cannot be hidden by a long-horizon planner or a language answer.

The two-sentence claim is: *A successful assembly trace does not show that an action
model knows which order is physically necessary. We pair identical goals and A→B
demonstrations with one visible mechanism change, then measure whether the first
manipulation branch changes exactly when B→A becomes feasible.* This is a diagnostic
claim, not a claim that one successful fixture proves general physics understanding.

## What is still required for a scientific run

- an Isaac builder that turns each recipe into a rendered USD layer;
- paired controller/seed calibration with no teleport or snap attachment;
- a canonical successful A→B demonstration for each domain;
- close-view visibility and leakage audits; and
- task-specific adaptation data if LingBot-VA is evaluated for completion rather than
  just native action-output smoke behavior.
