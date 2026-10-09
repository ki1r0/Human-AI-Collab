# Bolt Collision Geometry Audit

Updated 2026-10-09. This is a read-only geometry/API interpretation. No GPU run, environment edit, or source USD edit was performed for this audit.

## Finding

The 640-resolution SDF run is **not a verified collision remedy**. Its saved runtime manifest confirms SDF was actually active, yet the unheld nominal-seat trace still reports contacts against both cover and casing in every measurement sample. Separately, the source bolt is closed, consistently wound, and globally outward after its authored transforms; global mesh inversion is therefore not a supported explanation for the SDF result. Local self-intersection, SDF cooking/query behavior, and contact-report semantics remain untested.

## Verified SDF trace

- Capture: [`summary.json`](contact_calibration/gpu2-seat-20261008T185654-883856Z/summary.json) and [`raw_contacts.jsonl`](contact_calibration/gpu2-seat-20261008T185654-883856Z/raw_contacts.jsonl). Complete 361-record trace: post-reset, 120 warmup, and 240 measurement samples at `dt=1/120 s`; no contact-buffer truncation. It is an unheld nominal-seat calibration, not a task episode. Configured bolt mass is `0.045 kg`.
- Runtime manifest: one bolt mesh at `/World/envs/env_0/Bolt/node_/mesh_`, approximation `sdf`, resolution `640`, margin `0.0`; current authored configuration agrees. The raw trace has zero robot contacts.
- Cover, over 240/240 measurement samples: 12,500 reported points (52.083/sample), mean per-point force-vector norm `55.7607 N`, peak `583.909 N`; signed separation `[-1.325136, +0.084771] mm`; bolt-axis normal cosine `[-0.997191, +0.083371]`.
- Casing, over 240/240 samples: 4,687 reported points (19.529/sample), mean per-point force-vector norm `14.2638 N`, peak `141.649 N`; signed separation `[-0.355093, +0.099903] mm`; bolt-axis normal cosine `[-0.104112, +0.997190]`.
- The trace marks normal direction, penetration tolerance, and force thresholds uncalibrated. These force numbers are per-contact-point norms, not a net reaction; signed separation is a raw observation, not a calibrated physical penetration. Cover contact is expected for seating, but this trace does not establish correct cap support or load balance. Casing contacts remain despite the independently measured positive CAD radial clearance.

## Global winding audit

- Source `assets/parts/M6 Hub Bolt.usd`, mesh `/World/node_/mesh_`: USD orientation token `rightHanded`; 17,906 triangles. After welding coincident triangle-soup positions at `1e-4` source-unit tolerance: 8,955 vertices, 26,859 edges, zero boundary edges, zero non-manifold edges, and zero shared-edge direction conflicts.
- Oriented signed volume after mesh-to-root transform: `+833.216273` source-unit^3. Orientation-adjusted volume at the task instance scale (`0.002 m/source-unit`) is `+6.665730e-6 m^3`, classified outward. Mesh-to-root and root-to-stage linear determinants are both `+1`; uniform task-scale determinant is positive (`8e-9`). No global reflection or inward orientation was found.
- This rules against whole-mesh winding inversion, not local self-intersections or every SDF cooker failure mode. The offline auditor is [`diagnose_bolt_collision.py`](../../tools/diagnose_bolt_collision.py); its self-test passes, and its full offline inspection applies SDF schema only to an in-memory stage.

## SDF margin interpretation

The installed schema source is `/isaac-sim/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/plugins/PhysxSchema/resources/schema.usda:1043`. Its default `sdfMargin=0.01` expands the sampled SDF domain relative to the mesh bounds; it is **not evidence of physical surface inflation**. For this bolt's roughly `60.569 mm` bounds diagonal, that is about `0.606 mm` of domain margin. The failed trace used `0.0`, which may reduce valid external-field sampling near the bounds. Neither value is established as better here. Do not change margin by assumption; a paired `0.0` / default `0.01` comparison would need the same calibration conditions, runtime manifest, and authorization.

The schema's `sdfResolution` spacing is longest mesh AABB extent divided by resolution. At task scale the longest extent is `52.300 mm`: resolution `640` gives `0.08172 mm` spacing; the approximately `0.33793 mm` CAD casing radial clearance spans about 4.14 intervals. Resolution is not by itself proof that the cooked SDF preserves the clearance. The installed schema documents optional remeshing for inconsistent winding/self-intersections, with a geometry-accuracy tradeoff; this audit did not find global winding problems or test self-intersections. See NVIDIA's [PhysX collision documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html) and [SDF mesh collision API](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_s_d_f_mesh_collision_a_p_i.html).

## Two-piece CAD-derived proxy audit

The task-layer helper is [`collision.py`](../../runtime/bolt_harness/collision.py). It was run against the real source USD on an in-memory PXR stage only; no source layer was saved or edited. It transforms the source mesh into bolt-root coordinates, clips at `root-local Y=8.925001144`, assigns the coplanar cut face to the cap, drops degenerate clipped side polygons, and builds one SciPy convex hull per half. SciPy `1.15.3` is already installed in Isaac Python; no dependency was installed.

- Shaft hull: 4,506 source-derived hull vertices; bounds X `[-2.942000, +2.942000]`, Y `[-13.074999, 8.925001]`, Z `[-2.941998, +2.941999]` source units. Maximum diameter is `11.768004 mm`, matching the measured `11.768 mm` CAD shaft envelope; the helper does not scale it.
- Hex-cap hull: 68 source-derived hull vertices; bounds X `[-5.773503, +5.773503]`, Y `[8.925001, 13.074999]`, Z `[-5.0, +5.0]`. Overall maximum radius remains `5.773503` source units. At the shared split plane, the shaft boundary reaches radius `2.600001` and the cap underside reaches radius `5.000001`; each has a planar cut face. Separate prims terminate/start at the same Y plane, so there is no single convex hull bridging cap to shaft.
- Enclosure check on the actual clipped source triangles: shaft half produced 8,863 unique clipped points and cap half 190; evaluating every point against all corresponding SciPy hull half-spaces gave maximum positive excess `2.43e-12` and `1.78e-15` source units, respectively (floating-point zero). Thus each authored input hull encloses its complete clipped source surface, including interpolated split edges. This proves the authored hull input, not PhysX's later cooked collider.
- The source cap underside radius is `5.000001`, not the overall head radius `5.773503`. With the measured cover aperture radius `3.647927`, actual annular radial overlap at the bearing plane is `2.704148 mm`. The previously sampled ring at radius `4.357752` is on that underside; the outer ring at `5.653503` is outside the underside and cannot be counted as cap-bearing support.
- The threaded-shaft surface has sampled root radii near `2.3707` versus crest/envelope radius `2.942001`; the convex shaft therefore fills thread grooves by up to about `1.143 mm` radially in those sections, as intended for this no-thread task. Sampled shoulder fill is `0.316 mm` at Y `8.0` and `0.043 mm` at Y `8.8`; both remain within the global shaft envelope. Sampled cap-bevel fill is `0.774 mm` radially at Y `9.0`, only `0.15 mm` above the bearing plane; the cut-plane cap radius itself matches the source. These are local section comparisons, not evidence of dynamic contact behavior.
- The shaft hull maximum radial extent is `2.942001` source units. Against the measured minimum casing radial surface `3.110965`, the conservative projected casing gap remains `0.337927 mm`; against the cover aperture radius `3.647927`, shaft radial clearance is `1.411852 mm`. This uses the same axes and task scale as the static geometry audit. It does not test contact cooking or the cap bevel against every cover triangle.

The installed `PhysxConvexHullCollisionAPI` exposes `hullVertexLimit` (default `64`); the helper explicitly authors `64`, the GPU-compatible PhysX limit. PhysX 5.1 documentation describes its default Quickhull path as selecting a limited vertex set and expanding the cooked hull to enclose input points; the actual composed/cooked child bounds still require runtime verification. See [PhysX 5.1.3 convex mesh cooking](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/docs/Geometry.html) and [GPU rigid-body limits](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/GPURigidBodies.html).

The helper disables collision on only the original visual mesh, leaves its render mesh/material bindings untouched, copies direct material bindings and PhysX contact/rest offsets onto the new shapes, and keeps the existing root `RigidBodyAPI`. It creates `collision_shaft` and `collision_cap` with collision enabled, `purpose=guide`, invisible render visibility, and `approximation=convexHull`. It refuses a root without an existing `RigidBodyAPI`.

Caller API for Avicenna, after the existing bolt authoring has applied the root rigid body and mesh material/contact settings:

```python
from runtime.bolt_harness.collision import author_split_bolt_convex_proxies

bolt_proxy_info = author_split_bolt_convex_proxies(
    stage,
    bolt_root_path="/World/envs/env_0/Bolt",
    source_mesh_path="/World/envs/env_0/Bolt/node_/mesh_",
)
```

The helper returns the two child paths, source-hull bounds, hull counts, offsets, and collision flags. The caller should use that result instead of reporting the disabled visual mesh's old SDF collider as active.

Offline tests: `tests/test_bolt_harness_collision.py` passes in the installed Isaac Python. No environment/runner integration or GPU/physics calibration was run; parent review and cooked-collider verification remain required.

## Verification and next step

The geometry audit, in-memory proxy authoring, and tests used installed PXR/PhysX schema bindings in `relaxed_spence`; no Kit session or physics stepping was used. The 640-SDF physical calibration was run separately by the environment owner and is recorded above. The new two-hull input geometry preserves the CAD envelope, but neither its PhysX cooking nor its seat/support behavior is certified until the owner integrates it and inspects runtime collider bounds/contact data.
