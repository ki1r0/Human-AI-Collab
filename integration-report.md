# Qianli integration report

Status: `CODEBASE_INTEGRATION: PASS`

## Provenance and strategy

- Integration target: `/home/sunsiliang/Human-AI-Collab`, branch `main`, starting HEAD `631e7ed53e208b63779fa0a19518c4b620afab60`.
- Qianli source: `/home/sunsiliang/Downloads/Human-AI-Collab-qianli`.
- The Qianli source is a filesystem export with no `.git` directory. Therefore it has no source commit, merge base, branch graph, or unique-commit set to report. Its source commit is `UNAVAILABLE`.
- Strategy: semantic file-level union. Runtime-relevant Qianli files were compared by path and purpose, then imported selectively. This was safer than replacing the target tree and was the only history-aware option available for a source without Git metadata.

The target was already dirty. Existing changes to `.gitignore`, `memory/long_term.py`, `roco_single_run/`, `PILOT_12_PAIR_EXECUTION_PLAN.md`, `pilot_12pair/`, and `roco_single_run/artifacts/` were preserved. No merge tool, blanket `ours`/`theirs` resolution, reset, or cleanup was used.

## Integrated Qianli functionality

The semantic union added or updated:

- DAG/instance assembly: `assembly/asset_registry.yaml`, `assembly/gearbox_dag.yaml`, `assembly/gearbox_sequence.yaml`, `assembly/instantiate.py`, `assembly/sequence_runner.py`, three generated files under `assembly/instances/`, and supporting layout/pose files.
- Runtime behavior: `runtime/config.py`, `runtime/state.py`, `runtime/ui.py`, and `runtime/magic_assembly.py`.
- Launch/runtime packaging: `docker-compose.yml`, `docker/run_demo.sh`, and `tools/run_tool.sh`.
- Validation and playback tools: `tools/test_instantiate.py`, `tools/test_instance_playback.py`, `tools/test_pose_check.py`, `tools/test_seat_undo.py`, `tools/test_hover.py`, `tools/play_instance.py`, and the updated scatter/authoring utilities.
- Qianli's canonical `assets/simple_room_scene.usd` and its registered 57-part assembly inventory.

Historical Qianli status reports, backup layouts, redundant versioned scene snapshots, and an unreachable duplicate pallet asset were not imported. They add no runtime behavior and would duplicate current evidence.

## Conflict decisions

- `roco_single_run/` was left untouched by this integration. Geometry work uses repository-local gearbox assets and does not alter the RoCo environment, policy, camera, observation, or action contracts, so no learned-checkpoint run was repeated.
- Existing pilot work remains in place. The new `pilot-plan.md` maps its scientific design to the integrated real-part DAG rather than replacing the pilot package.
- Qianli's generated 73-step instance files are retained alongside the older 45-step `assembly/gearbox_sequence.yaml`. The DAG plus `assembly/instances/canonical_grouped.yaml` is the current generated-instance path; the older file remains a supported manual sequence and provenance record.
- Qianli creates some repeated fastener/socket instances at runtime. These were not flattened into the source USD because doing so would duplicate that implementation and risk changing its naming/alias behavior.

## Acceptance evidence

- `python tools/test_instantiate.py`: 22/22 pass; DAG generation, socket binding, dependency validation, and all variants pass.
- `docker compose run --rm hac tools/run_tool.sh tools/test_instance_playback.py`: 20/20 pass; all variants complete, grouped undo works, reset clears state, staging resolves sockets, and the canonical file contains 73 steps.
- `docker compose run --rm hac tools/run_tool.sh tools/test_seat_undo.py`: pass for flip/upright/put-down seating and exact undo within the stated tolerance.
- `docker compose run --rm hac tools/run_tool.sh tools/release_scene_audit.py --scene assets/simple_room_scene.usd --wait-updates 10`: zero unresolved asset paths and zero nested enabled rigid bodies.
- Live Isaac/PhysX asset tests load the upgraded USDs, cook SDF collision, step physics, and shut down cleanly; see `part-fidelity-matrix.md`.
- No conflict markers remain in the integrated implementation.

The earlier RoCo reproduction remains evidence for the independent ACT path. It was not rerun because no shared RoCo dependency was modified.

## Efficiency record

Ponytail 4.9.0 and Stop That Shit 0.1.0 were installed from their Codex plugin marketplaces. Their Node-based hooks cannot activate in this shell because Node is unavailable, so their change-mode constraints were applied manually: one primary thread, no new dependencies, no hashes, targeted tests, no repeated ACT trials or seed sweeps, and only the three permitted core deliverables.
