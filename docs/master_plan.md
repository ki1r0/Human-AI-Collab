Master Plan — Scene-Agnostic Perception & Stateful Belief
Problem Statement
Build a reusable agent pipeline in Isaac Sim where the AI:
Initializes Context: autonomously analyzes a scene to separate "Interactable Objects" from "Static Background."
Maintains State: holds a persistent belief state inside a graph architecture (LangGraph), reducing reliance on external memory modules.
Reasons Temporally: updates beliefs based on visual changes (deltas) rather than re-describing the world from scratch.
Visualizes Thoughts: maps internal belief states to Ghost USD prims for debugging.
Core Design Principles
Stateful Reasoning: The Agent holds the state; the Model computes the update.
Event-Driven: Compute is expensive. Only think when the world changes (GT Trigger) or the user asks (Manual Trigger).
Semantic Split: Memory is divided into Active Beliefs (tracking trajectory/occlusion) and Static Context (room layout/walls).
Architecture: The LangGraph Loop
Instead of external memory classes, the LangGraph State (AgentState) is the single source of truth.
AgentState Schema (TypedDict):
visual_buffer: List[Images] (Last K frames)
belief_state: Dict (Active objects: {'orange': {'status': 'moving', 'container': 'basket'}})
static_context: Dict/String (Background info: {'walls': 'wood', 'table': 'white'})
user_history: List (Chat logs)
Runtime Phases
Phase A — Initialization (The "Grounding" Step)
Trigger: User clicks "Initiate Cosmos" (Before Physics/Animation starts).
Capture: System grabs 1 Static Frame.
Inquiry: Call VLM with a specific "Initialization Prompt":
"Analyze this scene. Classify objects into Interactables (tools, fruits, containers) and Structural (walls, floor, table). Initialize the belief state for Interactables using available Ground Truth as a baseline guide."
Graph Action:
Populate AgentState['static_context'] with the structural description.
Populate AgentState['belief_state'] with the interactables.
UI Feedback: Show "Initialization Complete. Ready for Play."
Phase B — Runtime Loop (Event-Based)
Trigger: Ground Truth Monitor detects displacement > Threshold OR User Input.
Input:
Retrieve last K frames from Ring Buffer.
Retrieve AgentState['belief_state'] and AgentState['static_context'].
Reasoning Node (VLM):
Prompt: "Here is what you believed before: {belief}. Here is the background: {context}. Look at the video. What changed? Update the belief state (position, occlusion, containment)."
Task: The model outputs a Delta (e.g., "Orange fell into Basket"). It ignores the static table/walls unless they break.
Merge Node:
LangGraph performs a Dictionary Merge: Current_Belief.update(VLM_Delta).
Benefit: If the model focuses on the orange and ignores the basket, the basket remains in memory (it isn't deleted).
Visualization Node:
Read the new AgentState.
Update USD Ghost Prims to match the belief.
Phase C — Manual Chat
Trigger: User types in UI.
Input: User text + Current AgentState + Current Frame.
Reasoning: Model answers user question using the Belief as its knowledge base (e.g., "I believe the orange is in the basket because I saw it fall, even though I can't see it now").
"Done" Criteria
Initialization: Clicking "Initiate" correctly separates the Orange (Active) from the Table (Static) in the logs.
Persistence: If the camera looks away or the VLM output is brief, the Belief State retains previously known objects (no flickering/amnesia).
Efficiency: The VLM is NOT called continuously. It fires only when the Orange moves or the User speaks.
Visualization: The Ghost Orange snaps to the belief location (e.g., inside the basket) even if the real orange is occluded.