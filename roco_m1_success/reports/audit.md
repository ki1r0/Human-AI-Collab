# Official source audit

The official `GalaxeaRulePolicy.get_action()` sequence is planet gears 1–3,
the fourth/center gear, ring gear, and reducer. The carrier is the stationary
assembly base in that implementation. Pick and mount motions use the official
absolute-pose `DifferentialIKController` and the R1 arm/gripper joints.

The unchanged official evaluator awards three pin relations, one
carrier–ring relation, one center–ring relation, and one center–reducer
relation. Consequently center insertion alone does not change the score. The
expected successful score path is `0 -> 1 -> 2 -> 3 -> 5 -> 6`; the ring step
simultaneously makes the carrier–ring and center–ring relations observable.

Prior full scripted runs reached a transient 5/6. Their first late-stage
blocker was reducer release: at the lower insertion command the held reducer
compressed the stack, while at the clearance command the gripper opened under
load and dragged the stack during retreat. The local controller therefore
keeps the official grasp/transport/IK behavior but opens the reducer in axial
clearance and lets gravity perform the final seating before retreat.

The upstream center mount also commands a 60-degree wrist rotation. That
leaves the center quaternion incompatible with the later center–ring score
relation. The M1 controller uses the same official center target and a plain
axial insertion without that spin.
