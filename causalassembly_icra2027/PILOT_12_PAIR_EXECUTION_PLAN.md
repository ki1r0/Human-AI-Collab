Exit code: 0
Wall time: 1.8 seconds
Output:
# CausalAssembly 12-Pair Falsification Pilot

## Instructions for the implementing Codex

Treat this document as the execution specification. Work from the repository root,
preserve unrelated user changes, and implement the pilot end to end. Do not expand
the scientific scope without an explicit user request. When a detail conflicts with
the repository, keep the scientific contract below and adapt only the software
interface. Record every deviation and its reason in `pilot_12pair/DEVIATIONS.md`.

The pilot is complete only when the artifact checklist in Section 16 passes and a
go/no-go report is produced. A rendered demonstration is not enough. Physical labels,
leakage controls, model predictions, and paired statistics must all be present.

## 1. Objective and falsifiable claim

This is a small falsification pilot, not the final benchmark.

Structural claim:

> Successful reproduction of a demonstrated assembly procedure does not establish
> that an agent has identified the physical constraints that make the procedure
> valid.

Empirical hypothesis:

> Current direct video-language models reproduce the observed action order more
> accurately than they identify whether that order is physically necessary.

Benchmark claim:

> Matched physical interventions distinguish procedure imitation from physical-rule
> identification.

The pilot must not claim that all state-of-the-art models lack physical understanding.
It asks whether different method classes can be separated by one controlled diagnostic.
A physics-assisted method doing well is a positive result because it shows that the
diagnostic rewards physically grounded reasoning.

## 2. Scope freeze

### Required now

- 12 matched pairs, comprising 24 domains and 24 successful A-then-B videos.
- Three relation families: access, closure/fastening, and function-before-closure.
- Three questions per domain: demonstrated order, necessity, and feasible procedures.
- Sequence replay, direct VLM, structured VLM, and physics-assisted baselines.
- Physics/geometry oracle validation, visibility audit, leakage controls, metrics,
  confidence intervals, and a registered go/no-go decision.

### Explicitly excluded from this pilot

- RL training.
- Active query or learned evidence selection.
- Full VLA or low-level grasp learning.
- Real-robot deployment.
- Human-robot collaboration metrics.
- Cross-product or cross-task generalization claims.
- Preference and underdetermined benchmark labels. If a domain is ambiguous to expert
  viewers, redesign it; do not force it into HARD or COMMUTABLE.
- Training or reproducing large methods such as SIMPACT, NeSyCR, or PhysMind in full.

These exclusions are deliberate. The first experiment must establish that the central
phenomenon exists before additional methods or tracks are justified.

## 3. Experimental unit

Each matched pair contains two visually grounded domains:

- `HARD`: the demonstrated order A -> B is valid, but B -> A is invalid.
- `COMMUTABLE`: both A -> B and B -> A are valid.

Within a pair, hold constant:

- part identities and semantic action names;
- demonstrated order A -> B;
- camera trajectory, lighting, duration, hand/robot motion, and prompt;
- final task goal and required actions.

Change exactly one task-relevant physical mechanism. The changed mechanism must be
visible from permitted sensor observations. The two videos need not be pixel-identical;
only the controlled physical intervention may differ.

Every candidate procedure is terminal. `A only` means action B is omitted permanently,
and `B only` means action A is omitted permanently. There is no automatic completion.
Therefore the ground-truth feasible sets are:

```text
HARD:        {A -> B}
COMMUTABLE:  {A -> B, B -> A}
```

`A only` and `B only` are infeasible in all 24 domains because both actions are part of
the declared two-action task goal.

## 4. The exact 12 matched pairs

Use existing gearbox CAD where practical. Simple low-poly inserts, ports, captive
fasteners, occluders, or collision proxies may be authored when the current CAD cannot
express a clean intervention. Such proxy geometry must look mechanically plausible and
must be rendered as part of the scene, not exposed as an annotation.

Parameter values below are mechanisms, not final dimensions. For each pair, find the
critical feasibility threshold by calibration, then select HARD and COMMUTABLE values
on opposite sides with a margin at least twice the simulator/evaluator tolerance.

| ID | A then B shown in both domains | HARD intervention | COMMUTABLE intervention | Required visible evidence | Oracle signal |
|---|---|---|---|---|---|
| ACC-01 | Insert input-shaft module; insert transfer-shaft module | The transfer module occupies the input module's only swept insertion corridor | A widened relief preserves the input corridor after transfer insertion | Relief width and overlapping swept volumes | Collision-free insertion path |
| ACC-02 | Place thin washer on output seat; seat output gear | The seated gear covers the washer seat, preventing later placement | A radial washer slot permits side insertion after the gear is seated | Washer seat, gear footprint, and side slot | Reachable placement plus final seat contact |
| ACC-03 | Insert output key; mount output gear | The mounted gear encloses the keyway entrance | An open-ended keyway remains accessible after gear mounting | Keyway opening before and after mounting | Collision-free key insertion and keyed relation |
| ACC-04 | Insert output-shaft module; attach hub cover | The cover seals the only shaft insertion port | A second base-side insertion port remains open | Both possible ports and cover footprint | Collision-free shaft insertion path |
| FAS-01 | Close casing top; insert casing through-bolt | The bolt bore becomes continuous only after the casing halves align | A captive bolt can be retained in the top before closure without interference | Bore alignment and captive-bolt recess | Closure and bolt-seat constraints |
| FAS-02 | Close casing top; install casing nut | The nut must bridge aligned lugs that do not form a seat before closure | A captive-nut pocket retains the nut before closure | Lug geometry versus captive pocket | Nut retention and collision-free closure |
| FAS-03 | Seat casing top; insert locating dowel | A continuous dowel bore exists only in the seated assembly | An open-slot/captive dowel can be preloaded without blocking seating | Closed bore versus open retaining slot | Dowel seating and closure feasibility |
| FAS-04 | Seat hub cover; install hub bolt | The bolt cannot be retained or aligned before the cover is seated | A captive hub bolt can be preloaded in the cover | Bolt-head recess and captive feature | Cover seating and bolt retention |
| TST-01 | Visually inspect gear mesh; close casing | Opaque closure removes every view of the mesh | A transparent inspection window preserves the same inspection after closure | Mesh and window/opaque surface | Camera/raycast visibility of mesh target |
| TST-02 | Rotate input and verify output response; close casing | Closure removes access to both actuation and output observation points | External input/output shaft features remain accessible after closure | Exposed versus covered shaft features | Actuation reachability plus measured output rotation |
| TST-03 | Measure gear backlash; close casing | The gauge can reach the gear only while the casing is open | A dedicated gauge port remains accessible after closure | Gauge corridor and port | Collision-free gauge path plus valid reading |
| TST-04 | Verify internal timing/alignment marks; close casing | Closure occludes all timing marks | A sight window/external witness marks preserve the check | Internal marks and alternate visible marks | Visibility plus alignment-state readout |

### Pair-design constraints

1. Do not change object names, narration, demonstrated order, or task goal between the
   two domains in a pair.
2. Do not encode `hard`, `commutable`, `blocked`, `wide`, or similar answers in filenames,
   visible labels, colors, prompts, or model-facing metadata.
3. Counterbalance variant file order, neutral object colors, and anonymous domain IDs.
4. The intervention must alter B -> A feasibility while preserving A -> B feasibility.
5. A model must be able to see the relevant geometry. A hidden mutation is an
   underdetermined task and is invalid for this pilot.
6. Use one successful A -> B demonstration in every domain. Never show B -> A in the
   canonical model input.

## 5. Repository layout to create

Create the following self-contained package at the repository root:

```text
pilot_12pair/
  README.md
  DEVIATIONS.md
  pyproject.toml                 # or requirements file consistent with the repo
  config/
    pairs.yaml
    models.example.yaml
    prompts.yaml
  schemas/
    domain.schema.json
    prediction.schema.json
    oracle_result.schema.json
  domains/
    public_manifest.jsonl
    evaluator_manifest.jsonl    # evaluator-only; never passed to a model runner
  scenes/
    source/
    generated/
  media/
    canonical/
    controls/
  oracle/
    build_domains.py
    execute_candidate.py
    calibrate_pairs.py
    validate_pairs.py
  runners/
    sequence_replay.py
    direct_vlm.py
    structured_vlm.py
    physics_assisted.py
    run_matrix.py
  prompts/
    direct_system.txt
    direct_user.txt
    graph_system.txt
    graph_user.txt
  evaluation/
    validate_predictions.py
    score.py
    statistics.py
    make_figures.py
  tests/
    test_schemas.py
    test_metric_formulas.py
    test_manifest_separation.py
    test_pair_invariants.py
    test_no_label_leakage.py
  outputs/
    raw/
    normalized/
    metrics/
    figures/
  reports/
    oracle_validation.md
    visibility_audit.csv
    leakage_audit.md
    go_no_go.md
```

Do not modify `runtime/magic_assembly.py` into the physical oracle. It may be used to
place parts for rendering or generate the shown A -> B demonstration. The oracle must
be independent and must evaluate candidate feasibility from collision, reachability,
contact, closure, visibility, or function checks.

## 6. Data contracts

### 6.1 Public domain manifest

One line per domain in `public_manifest.jsonl`:

```json
{
  "domain_id": "D-7Q2K",
  "pair_id": "P-04",
  "family": "access",
  "media": {"canonical_video": "media/canonical/D-7Q2K.mp4"},
  "action_a": {"id": "A", "text": "insert the input-shaft module"},
  "action_b": {"id": "B", "text": "insert the transfer-shaft module"},
  "candidate_procedures": [["A", "B"], ["B", "A"], ["A"], ["B"]]
}
```

Public IDs must be randomly assigned. `pair_id` may be used for paired scoring, but it
must not reveal family, source pair name, or label. During model calls, domains are
presented individually and `pair_id` is omitted.

### 6.2 Evaluator-only manifest

One line per domain in `evaluator_manifest.jsonl`:

```json
{
  "domain_id": "D-7Q2K",
  "source_pair": "ACC-01",
  "variant": "HARD",
  "demonstrated_order": ["A", "B"],
  "can_b_before_a": false,
  "feasible_procedures": {"A>B": true, "B>A": false, "A": false, "B": false},
  "intervention_parameter": {"name": "corridor_clearance_mm", "value": 3.5},
  "oracle_evidence_path": "outputs/raw/oracle/D-7Q2K.json"
}
```

This file may contain simulator paths, thresholds, and labels. It must never be loaded
by direct or structured model runners. Add a test that fails if any evaluator-only key
appears in a serialized model request.

### 6.3 Normalized prediction

Every runner must emit the same JSONL schema:

```json
{
  "run_id": "2026-08-10T120000Z-model-condition-seed",
  "model_id": "provider/model-version",
  "method": "direct",
  "condition": "canonical_video",
  "prompt_version": "v1",
  "seed": 0,
  "domain_id": "D-7Q2K",
  "sequence_order": ["A", "B"],
  "can_b_before_a": false,
  "confidence_b_before_a": 0.12,
  "procedure_feasibility": {"A>B": true, "B>A": false, "A": false, "B": false},
  "constraint_graph": [],
  "rationale": "optional unscored text",
  "raw_response_path": "outputs/raw/..."
}
```

`confidence_b_before_a` is the model's probability that B -> A is feasible, in [0, 1].
Parse failures count as incorrect; never silently retry until a correct format appears.
Record one format-repair retry separately and score the original failure in a strict
analysis plus the repaired output in a secondary analysis.

## 7. Scene implementation and oracle validation

### 7.1 Build domains

1. Audit existing CAD assets and current Isaac/Isaac Lab runtime.
2. Build one parameterized template per row in Section 4.
3. Generate two variants per template while changing only the registered mechanism.
4. Keep exact simulator parameters in the evaluator manifest.
5. Assign randomized public IDs only after all labels are frozen.

### 7.2 Candidate execution

For each domain, reset the scene and execute all four terminal procedures independently:

```text
A -> B
B -> A
A only
B only
```

Candidate actions may use deterministic high-level scripted trajectories. Low-level
policy learning is outside scope. A procedure succeeds only if:

- every commanded action reaches its relation-specific goal;
- collision/penetration and force thresholds remain within registered limits;
- the declared final two-action task goal is satisfied;
- task-specific access, fastening, inspection, or functional checks pass.

Store trajectories, collision events, final relations, and failure reasons. Do not
infer feasibility merely from an authored ordering rule.

For insertion pairs, use swept-volume collision and contact checks. For fastening
pairs, check retention, alignment, closure, and interference. For inspection pairs,
use camera/tool raycasts and access paths. For TST-02, additionally check that input
actuation yields the expected output rotation above a registered tolerance.

### 7.3 Robustness validation

Run each procedure at least 20 times with small, registered perturbations of initial
pose, actuation timing, and simulator seed. A pair passes only if:

- A -> B succeeds in both variants in at least 95% of trials;
- B -> A succeeds in COMMUTABLE in at least 95% of trials;
- B -> A succeeds in HARD in at most 5% of trials;
- the outcome is caused by the intended mechanism, confirmed from failure logs;
- A only and B only fail the declared terminal goal in 100% of trials.

If a pair misses a threshold, calibrate geometry and rerun it. Do not weaken the label
or hand-edit the result. Freeze a pair only after validation and save simulator/runtime
versions plus all random seeds.

## 8. Demonstration media

Render one successful A -> B demonstration for every domain.

Recommended canonical media:

- MP4, H.264, 720p, 10-15 FPS, 10-20 seconds;
- a fixed two-panel layout: full assembly view and task-critical close view;
- two seconds of the initial geometry before action A;
- identical timing and camera paths within each pair;
- no text overlay except neutral `Step A` and `Step B` markers if needed;
- no simulator UI, prim paths, collision shapes, socket markers, labels, or narration.

Generate these controls from the same source render:

- `text_only`: action names and demonstrated order only; no geometry description;
- `final_frame`: final RGB frame only;
- `static_multiview`: pre-action geometry views without the demonstration;
- `shuffled_video`: the same frames in a fixed shuffled order.

Interpret controls carefully. Text-only should be near chance for HARD versus
COMMUTABLE within matched pairs. Static geometry may support necessity reasoning.
Shuffling primarily tests temporal order recall; it need not destroy geometry-based
necessity reasoning.

## 9. Visibility and leakage audits

### 9.1 Visibility sufficiency

Before running expensive models, ask three independent mechanically literate reviewers
to answer Q2 and Q3 from canonical media. They must not see labels or paired videos
side-by-side. Record confidence and a short evidence description.

A domain passes if:

- majority necessity accuracy is at least 85%;
- at least two reviewers identify the intended visible mechanism;
- no reviewer reports that the answer requires hidden simulator state.

Report raw agreement and Fleiss' kappa where defined. This is a dataset quality audit,
not a claim about general human performance. Redesign ambiguous domains.

### 9.2 Leakage audit

Automatically scan all model-facing artifacts for:

- label words and variant names;
- source pair IDs or informative filenames;
- simulator prim paths, sockets, transforms, and collision metadata;
- different video lengths, codecs, frame counts, or camera timings within a pair;
- colors or object names perfectly correlated with labels.

Manually verify that a text-only classifier cannot identify variant labels above chance.
Randomize public IDs and file ordering. Keep SHA-256 hashes of frozen manifests/media.

## 10. Questions and prompts

Use one canonical prompt for the primary result. Prompts must request machine-readable
JSON and must not mention HARD or COMMUTABLE as the only choices.

### Q1: sequence recall

```text
What was the demonstrated order of the two named actions? Return A>B or B>A.
```

### Q2: physical necessity

```text
Can action B be completed before action A and still successfully complete the full
two-action assembly goal in the shown domain? Return yes/no and your probability that
B>A is feasible.
```

### Q3: feasible procedure set

```text
For each terminal candidate A>B, B>A, A only, and B only, decide whether it can
successfully complete the declared full two-action assembly goal in the shown domain.
```

Require the model to use only visible evidence and to mark uncertainty in its confidence,
but do not expose the counterpart domain. Store rationales for diagnosis; do not grade
rationale style.

Use three semantically equivalent prompt versions as a robustness analysis, with the
canonical version registered before inspecting results. The statistical unit remains
the matched pair, not the prompt repetition.

## 11. Baseline matrix

Model identifiers must be configuration values because provider names and availability
change. Record the exact model snapshot/date, API parameters, preprocessing, and cost.

### Required methods

1. **Sequence replay**
   - Returns the observed A -> B order.
   - Predicts only A -> B as feasible.
   - Demonstrates that order memorization cannot solve matched mutations.

2. **Strong open video-language model**
   - Select the strongest reproducible video-capable open model available on the
     execution date.
   - Run the canonical direct prompt with fixed decoding and documented frame sampling.

3. **Frontier closed video-language model A**
   - Select a current frontier model with native video or documented multi-frame input.

4. **Frontier closed video-language model B**
   - Use a different provider/model family to avoid correlated conclusions.

5. **VLM + explicit constraint graph**
   - Use the same base VLM as one direct condition.
   - First produce typed relations such as `blocks_access`, `requires_alignment`,
     `seals_view`, `retains`, `test_available`, and action preconditions/effects.
   - A deterministic graph reasoner enumerates the four candidate procedures.
   - This tests whether explicit representation helps without external physical trials.

6. **VLM + intervention oracle**
   - The VLM proposes/evaluates the fixed candidates.
   - Execute each candidate with the independent oracle and return only observable
     outcomes such as `collision`, `cannot_reach`, `closure_failed`, `test_unavailable`,
     or `goal_satisfied`.
   - The VLM revises its answer once. Never return the hidden label or constraint graph.
   - Treat this as a physics-grounded upper/reference baseline, not a deployable claim.

### Input controls

Run `canonical_video`, `text_only`, `final_frame`, `static_multiview`, and
`shuffled_video` for at least the open model and one frontier closed model. Other model
classes need only canonical video in the first pass.

Do not add CoT as a separate scientific method unless it changes more than prompt text.
It may be a secondary decoding ablation.

## 12. Metrics and statistics

Let there be 24 domains grouped into 12 matched pairs.

### Primary metrics

```text
SequenceRecall = exact Q1 accuracy over domains
NecessityAcc   = Q2 accuracy over domains
ImitationGap  = SequenceRecall - NecessityAcc
PMC           = fraction of pairs where both HARD and COMMUTABLE Q2 labels are correct
FSEM          = fraction of domains where the full predicted feasible set is exact
```

### Error metrics

Let `F` be the true feasible set and `F_hat` the predicted feasible set:

```text
UnsafePermission = |F_hat minus F| / max(|F_hat|, 1)
FalseRestriction = |F minus F_hat| / |F|
Brier             = mean (p(B>A feasible) - y)^2, where y=1 for COMMUTABLE
```

Here, `UnsafePermission` is the fraction of procedures permitted by the model that are
actually infeasible, and `FalseRestriction` is the fraction of truly feasible procedures
that the model rejects. Also report raw false-positive and false-negative counts so an
empty predicted set cannot look safe without showing its severe restriction error.

Report all metrics overall, by relation family, by method, and by input condition. Do
not overinterpret family-level percentages with only four pairs.

### Statistical protocol

- Compute 95% confidence intervals by resampling the 12 pair IDs with replacement and
  retaining both domains in every resampled pair.
- Compare SequenceRecall and NecessityAcc with an exact paired test over domain-level
  correctness, and report the effect size even when power is limited.
- Compare direct versus structured and physics-assisted variants with paired tests on
  the same domains.
- Treat prompt variants and repeated API calls as robustness repeats, not independent
  benchmark samples.
- Report every parse failure and failed API call. Never drop a domain selectively.
- Mark all analyses exploratory because this is a 12-pair pilot.

The main figure must show, for each method, the progression:

```text
Observed Order Recall -> Physical Necessity -> Paired Mutation Consistency
```

The main table must include SequenceRecall, NecessityAcc, ImitationGap, PMC, FSEM,
UnsafePermission, and FalseRestriction with paired-bootstrap intervals.

## 13. Pre-registered go/no-go decision

### Data-quality gate: all conditions required

- All 12 pairs pass the oracle thresholds in Section 7.3.
- Expert visibility-audit necessity accuracy is at least 85% overall and no retained
  domain is judged to require hidden state.
- No label or metadata leakage is found.
- Text-only label prediction remains near the balanced 50% baseline within uncertainty.

Failure here means repair the scenes or protocol; it says nothing about model ability.

### GO: scale the benchmark

Scale only if both are observed:

1. At least two distinct direct-model families achieve SequenceRecall >= 80% and an
   ImitationGap >= 15 percentage points on canonical video.
2. At least one physically grounded or structured method changes the result in the
   predicted direction, or rule quality significantly predicts correct feasible-set
   behavior. A particularly strong signal is a physics-assisted improvement of at least
   15 points in NecessityAcc or 20 points in PMC over its matched direct VLM.

These thresholds are pilot decision rules, not population claims.

### NO-GO or REDESIGN

- If all strong direct models achieve NecessityAcc >= 85% and PMC >= 80%, the proposed
  difficulty does not expose the central gap. Stop or design harder, still-visible cases.
- If SequenceRecall is high, NecessityAcc is low, but canonical video improves less than
  five points over text-only and reviewers cannot identify scene evidence, redesign the
  visual observations.
- If experts perform poorly, relabel as underdetermined for future work or redesign; do
  not count model errors.
- If the oracle is unstable, repair physics/evaluator implementation before any model
  conclusion.
- If direct and physics-assisted methods perform equally poorly, debug the observation
  and oracle interfaces; do not claim a universal reasoning failure.

Regardless of outcome, 12 pairs are insufficient for a full ICRA benchmark claim. A GO
decision authorizes expansion to more pairs, a second product, behavioral execution,
and only then optional active/RL methods.

## 14. Implementation phases and acceptance tests

### Phase 0: audit and freeze

- Read repository documentation, current scene code, CAD inventory, and environment
  launch scripts.
- Confirm Isaac Sim/Isaac Lab and rendering work on the target computer.
- Create the package layout and `DEVIATIONS.md`.
- Freeze this hypothesis, scope, metrics, and decision rule in `README.md`.

Acceptance: package imports, tests run, and no unrelated code is modified.

### Phase 1: manifests and metric unit tests

- Implement schemas, pair configuration, anonymous ID assignment, and metric functions.
- Construct synthetic predictions with known scores and test every formula.
- Enforce evaluator/public manifest separation.

Acceptance: all schema and metric tests pass before scene work begins.

### Phase 2: build and calibrate 12 pairs

- Implement one pair at a time, starting with ACC-01, FAS-01, and TST-01.
- Run all four candidate procedures and calibrate feasibility margins.
- Expand to the other nine pairs only after one pair per family passes.

Acceptance: `oracle_validation.md` contains 24-domain results and all Section 7.3
thresholds pass.

### Phase 3: render and audit media

- Render canonical videos and all control conditions from frozen scenes.
- Run automated leakage checks and the three-reviewer visibility audit.
- Fix and rerender failed domains, then refreeze hashes.

Acceptance: all data-quality gates pass and media hashes are recorded.

### Phase 4: implement baselines

- Implement sequence replay first as an evaluation sanity check.
- Implement one direct model runner and validate on two non-evaluation smoke scenes.
- Add the remaining direct models, structured graph baseline, and physics-assisted
  baseline with exactly the same public manifest and prompts.

Acceptance: every method produces schema-valid predictions for all 24 domains; missing
or failed outputs are explicit.

### Phase 5: locked evaluation

- Freeze prompts, model versions, and evaluator manifest checksum.
- Run the canonical matrix, then controls.
- Score once using the registered primary script; place any later analysis in a clearly
  marked exploratory section.

Acceptance: raw responses, normalized predictions, metrics, confidence intervals, and
the main table/figure are reproducible from one documented command.

### Phase 6: decision report

Write `reports/go_no_go.md` with:

1. data-quality gate outcomes;
2. primary table and figure;
3. direct evidence for or against the imitation gap;
4. structured and physics-assisted comparisons;
5. control-condition findings;
6. failed cases with screenshots and oracle traces;
7. an explicit `GO`, `REDESIGN`, or `STOP` decision using Section 13;
8. the smallest justified next experiment.

Do not rewrite the hypothesis after seeing results.

## 15. Reproducibility requirements

- Pin Python packages, simulator/container versions, model IDs, and media codecs.
- Record git revision when possible, but do not alter unrelated dirty-worktree changes.
- Save seeds, prompts, API parameters, frame-sampling policy, timestamps, latency, and
  approximate API cost.
- Keep raw model responses immutable.
- Store evaluator-only state separately from model-facing data.
- Provide one command each for validation, rendering, baseline execution, scoring, and
  report generation. Adapt command syntax to the repository, but document it in the
  package README.
- Never publish credentials, `.env` files, provider keys, private CAD, or confidential
  project material.

## 16. Minimum artifact checklist

The implementing Codex must not declare completion until all applicable boxes are true:

- [ ] 12 pair configurations and 24 anonymous domains exist.
- [ ] All four candidate procedures were independently evaluated in every domain.
- [ ] All oracle robustness thresholds pass.
- [ ] 24 canonical videos and four control conditions are generated.
- [ ] Visibility and leakage audits pass.
- [ ] Public and evaluator manifests validate and remain isolated.
- [ ] Sequence replay sanity-check scores match its expected logical behavior.
- [ ] At least one open and two closed direct model families are evaluated, or unavailable
      providers are explicitly documented without fabricating results.
- [ ] Explicit-graph and intervention-oracle baselines are evaluated.
- [ ] Primary metrics, paired confidence intervals, main table, and main figure exist.
- [ ] Every parse/API/oracle failure remains in the released logs.
- [ ] `go_no_go.md` applies the pre-registered rule without moving thresholds.
- [ ] The report states that this is a pilot and does not claim final benchmark scale.

## 17. Expected logical sanity checks (not expected empirical results)

Before calling external models, verify these deterministic properties:

- Sequence replay has 100% SequenceRecall.
- Because it always treats the demonstrated order as necessary, sequence replay has
  50% NecessityAcc on the balanced HARD/COMMUTABLE set.
- Sequence replay has 0% PMC because it cannot give different labels to both members of
  any matched pair.
- A perfect oracle has 100% on SequenceRecall, NecessityAcc, PMC, and FSEM, with zero
  UnsafePermission and zero FalseRestriction.
- A model returning the same necessity answer for both members of every pair cannot
  exceed 50% NecessityAcc and has 0% PMC.

If the scoring code violates any of these properties, fix the evaluator before running
the experiment.

## 18. Recommended execution order under limited time

1. Implement schemas, metrics, and sequence replay.
2. Build ACC-01, FAS-01, and TST-01 as one proof case per family.
3. Validate physics and visibility on those three pairs.
4. Run one strong direct model and the physics-assisted reference on six domains.
5. If the mini-pilot is discriminative, build the remaining nine pairs.
6. Run the full locked matrix and apply the go/no-go rule.

This staged order minimizes wasted simulation and API work while preserving the
registered 12-pair endpoint.

