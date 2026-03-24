# Human-AI-Collab: Isaac Sim + Cosmos Reason2-8B Cognitive Robot Demo

A cognitive robotics demonstration integrating NVIDIA Isaac Sim 5.1.0 with Cosmos Reason2-8B Vision Language Model (VLM) for autonomous object manipulation and reasoning.

## Overview

This project implements a two-phase grounding architecture where a VLM agent:
- Perceives the environment through multi-frame temporal vision
- Reasons about object states and dynamics
- Plans and executes manipulation tasks
- Maintains short-term and long-term memory

## Features

- **Vision-Language Integration**: Cosmos Reason2-8B VLM for scene understanding
- **Temporal Perception**: Multi-frame buffer for motion analysis
- **Belief Tracking**: Ground-truth state monitoring with confidence scores
- **Robot Control**: Franka Panda arm with IK-based motion planning
- **Memory System**: Short-term (working memory) + Long-term (Mem0 API)
- **Interactive UI**: Real-time visualization and manual control

## Requirements

- NVIDIA Isaac Sim 5.1.0
- IsaacLab framework
- Python 3.10+
- NVIDIA RTX GPU (tested on A5000)
- Cosmos Reason2-8B model server (vLLM or compatible)

## Installation

1. Install [Isaac Sim 5.1.0](https://developer.nvidia.com/isaac-sim)
2. Install IsaacLab:
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   ./isaaclab.sh --install
   ```
3. Clone this repo anywhere convenient:
   ```bash
   git clone https://github.com/ki1r0/Human-AI-Collab.git
   cd Human-AI-Collab
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Set runtime environment in `config/runtime_env.env` / `config/runtime_env.local.*`.

Key environment variables:
- `COMMANDER_API_KEY`: required higher-level LLM key
- `COSMOS_BASE_URL` or `COSMOS_CHAT_COMPLETIONS_URL`: optional Cosmos endpoint
- `COSMOS_MODEL`: Model path (default: /models/Cosmos-Reason2-8B)
- `MEM0_API_KEY`: Long-term memory API key

## Usage

### Launch the Demo

```bash
./run_main.sh
```

### UI Controls

- **PLAY/PAUSE**: Start/stop physics simulation
- **Auto Mode**: Enable autonomous VLM decision-making
- **Manual Query**: Send custom prompts with current frames
- **Teleport Objects**: Reset scene configuration
- **Ghost Mode**: Visualize planned trajectories

## Project Structure

```
Human-AI-Collab/
|-- main.py                  # Entry point
|-- rc_ui.py                  # Entry point (GUI, Kit event loop)
|-- rc_config.py              # All configuration constants
|-- rc_state.py               # Global runtime state
|-- rc_log.py                 # Logging utilities
|
|-- agent/                    # VLM reasoning + decision pipeline
|   |-- reason2.py            #   Cosmos Reason2-8B VLM calls + token budget
|   |-- graph.py              #   LangGraph agent state machine
|   |-- worker.py             #   Background worker thread
|   +-- parser.py             #   JSON response parsing
|
|-- sensor/                   # Perception + camera
|   |-- camera.py             #   Replicator RGB capture (sync + async)
|   |-- perception.py         #   Ground-truth state monitor
|   +-- vlm.py                #   VLM connection testing utilities
|
|-- belief/                   # World state tracking
|   |-- manager.py            #   Belief state manager (object hypotheses)
|   +-- ghost_visualizer.py   #   USD ghost prim visualization
|
|-- memory/                   # Short-term + long-term memory
|   |-- short_term.py         #   In-memory working memory (TTL-based)
|   +-- long_term.py          #   SQLite + Mem0 episodic memory
|
|-- control/                  # Robot motion control
|   |-- franka.py             #   Franka Panda IK controller
|   +-- robot.py              #   High-level robot commands
|
|-- assets/                   # USD scenes and robot models
|   |-- simple_room_scene.usd
|   |-- AI_robot.usd
|   +-- mock_robot.usd
|
|-- docs/                     # Architecture and debugging notes
|   |-- TWO_PHASE_GROUNDING.md
|   |-- CONTEXT_LENGTH_FIX.md
|   |-- FIXES_APPLIED.md
|   +-- ROBOT_CONTROL_TEST.md
|
+-- tests/                    # Verification scripts
    |-- verify_sim_tick.py
    +-- verify_to_torch.py
```

## Architecture

**Two-Phase Grounding:**
1. **Initialization Phase**: VLM builds scene graph from multi-view images
2. **Runtime Phase**: VLM reasons with temporal frames + ground-truth belief state

**Token Budget Management:**
- Context limit: 2048 tokens (Reason2-8B)
- Dynamic max_tokens adjustment based on input estimation
- Image encoding: ~350 tokens/image
- Calibrated text estimation: chars/5

## Documentation

- [docs/TWO_PHASE_GROUNDING.md](docs/TWO_PHASE_GROUNDING.md) - Two-phase perception architecture
- [docs/CONTEXT_LENGTH_FIX.md](docs/CONTEXT_LENGTH_FIX.md) - Token budget management
- [docs/FIXES_APPLIED.md](docs/FIXES_APPLIED.md) - Bug fix history
- [docs/ROBOT_CONTROL_TEST.md](docs/ROBOT_CONTROL_TEST.md) - Robot controller testing

## Known Issues

- **Python Module Caching**: After editing IsaacLab source files, restart Isaac Sim
- **Token Estimation**: Slightly conservative (safety margin 20 tokens)
- **Mem0 API**: Requires valid API key for long-term memory persistence

## Performance

Tested on:
- GPU: NVIDIA RTX A5000
- Isaac Sim 5.1.0 (headless/GUI modes)
- VLM inference: ~4s per query (3 frames)
- Frame capture: 30 FPS -> 3 FPS temporal sampling
