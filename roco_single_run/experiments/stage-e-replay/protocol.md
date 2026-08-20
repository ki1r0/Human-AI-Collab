# Stage E — official demonstration replay/action-interface test

Classification: **CONFIRMATORY**.

## Pinned input

- Official dataset repository: `rocochallenge2025/rocochallenge2025`.
- Immutable revision: `c7ae062e9d627966451f1ff0cbcf7443b78f0992`.
- File: `gearbox_assembly_demos_updated/1.hdf5`.
- Expected size: 951,664,104 bytes.
- Expected SHA-256: `dd7ade042484b4fc5a868df6f17df0a90ccdebbd6b1e309d916bbd2e2450d434`.
- Reference implementation: upstream `DataReplayPolicyWrapper` from the pinned
  `gearboxAssembly` checkout.

The episode has 590 samples. Independent inspection established that its action
at index `t` is exactly its robot qpos at `t+1` (zero aggregate absolute error)
in policy order. `current_time` is all zero, so timing is taken from the live
environment contract (0.05 s/control step), not this defective metadata field.

## Locked method

1. Launch the unchanged prepared R1 learned-agent environment with one instance.
2. Use seed 1 only as a documented heuristic because the public HDF5 has no
   reset seed or initial object-state metadata. Do not tune the scene to the
   trajectory.
3. Instantiate the upstream `DataReplayPolicyWrapper` on the pinned HDF5.
4. Independently reconstruct file actions in environment order
   `[L arm 6, R arm 6, L grip, R grip]` and require exact equality with the
   wrapper tensor.
5. Feed the wrapper fresh live qpos and RGB on every iteration even though the
   replay implementation deliberately ignores them.
6. Execute all 590 commands unless the official score reaches 6. Record live
   qpos before/after every command, score, done flags, reset/final RGB, and a
   machine-readable trace.
7. Do not change policy weights, controller gains, trajectory samples, object
   states, reward code, or physics for this test.

## Pass criteria

- Dataset size/hash match and the wrapper exposes exactly 590 finite 14-D
  actions.
- Wrapper/file action maximum absolute difference is exactly zero.
- Dataset action-to-next-qpos maximum absolute difference is exactly zero.
- The live reset qpos is within 0.01 mean absolute error of recorded qpos 0.
- All executed commands and observed live qpos are finite.
- Mean absolute live-after-step error from the commanded absolute target is at
  most 0.10 across executed non-reset steps.
- For demanded joint motions over 0.001, at least 80% of observed motions have
  the commanded direction.
- The replay reaches its final sample or genuine score 6 without premature
  termination/truncation.

Task success is reported separately. It is not an action-interface pass
criterion because the released file omits the random reset seed/object poses;
therefore a replayed open-loop robot trajectory cannot be asserted to match its
source scene.

## Compatibility note after attempt 1

Attempt 1 failed before environment creation because the upstream subclass
assigns `self.device` while its base class exposes `device` as a read-only
property. The retained compatibility subclass adds only a setter backed by
`torch.device`; all upstream HDF5 loading, concatenation, step indexing, and
`predict()` behavior remain inherited and unchanged. No pass criterion changed.
