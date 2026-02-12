# Isaac Sim + Cosmos Reason2-8B Cognitive Robot Demo

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
3. Clone this demo into IsaacLab:
   ```bash
   cd IsaacLab
   git clone <your-repo-url> nvidia_tutorial
   ```
4. Install dependencies:
   ```bash
   pip install langgraph langchain_core mem0ai pillow
   ```

## CPU Compatibility Patches

This demo includes patches for IsaacLab to support CPU device mode (fixes PhysX tensor API numpy/torch compatibility). Patched files:
- `source/isaaclab/isaaclab/assets/articulation/articulation.py`
- `source/isaaclab/isaaclab/assets/articulation/articulation_data.py`
- `source/isaaclab/isaaclab/assets/rigid_object/rigid_object.py`
- `source/isaaclab/isaaclab/assets/rigid_object/rigid_object_data.py`
- `source/isaaclab/isaaclab/assets/rigid_object_collection/rigid_object_collection.py`
- `source/isaaclab/isaaclab/assets/rigid_object_collection/rigid_object_collection_data.py`

See [FIXES_APPLIED.md](FIXES_APPLIED.md) for details.

## Configuration

Edit `rc_config.py` to configure:
- VLM endpoint and model
- Camera settings
- Frame capture parameters
- Memory retention policies

Key environment variables:
- `COSMOS_ENDPOINT`: VLM API endpoint (default: http://172.17.0.1:8000)
- `COSMOS_MODEL`: Model path (default: /models/Cosmos-Reason2-8B)
- `MEM0_API_KEY`: Long-term memory API key

## Usage

### Launch the Demo

```bash
cd IsaacLab
./isaaclab.sh -p nvidia_tutorial/cognitive_main.py
```

### UI Controls

- **PLAY/PAUSE**: Start/stop physics simulation
- **Auto Mode**: Enable autonomous VLM decision-making
- **Manual Query**: Send custom prompts with current frames
- **Teleport Objects**: Reset scene configuration
- **Ghost Mode**: Visualize planned trajectories

### File Structure

```
nvidia_tutorial/
├── cognitive_main.py          # Main entry point
├── rc_ui.py                   # UI and event loop
├── rc_reason2_agent.py        # VLM integration + LangGraph agent
├── rc_perception.py           # Frame capture and preprocessing
├── rc_belief_manager.py       # Object state tracking
├── rc_franka_control.py       # Robot motion controller
├── rc_config.py               # Configuration constants
├── rc_*_memory.py             # Memory systems
├── simple_room_scene.usd      # Demo scene (table + objects)
└── tests/                     # Verification scripts
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

## Known Issues

- **Python Module Caching**: After editing source files, restart Isaac Sim to reload changes
- **Token Estimation**: Slightly conservative (safety margin 20 tokens)
- **Mem0 API**: Requires valid API key for long-term memory persistence

## Documentation

- [FIXES_APPLIED.md](FIXES_APPLIED.md) - Complete fix history
- [TWO_PHASE_GROUNDING.md](TWO_PHASE_GROUNDING.md) - Architecture details
- [CONTEXT_LENGTH_FIX.md](CONTEXT_LENGTH_FIX.md) - Token management solution
- [ROBOT_CONTROL_TEST.md](ROBOT_CONTROL_TEST.md) - Motion control testing

## Performance

Tested on:
- GPU: NVIDIA RTX A5000
- Isaac Sim 5.1.0 (headless/GUI modes)
- VLM inference: ~4s per query (3 frames)
- Frame capture: 30 FPS → 3 FPS temporal sampling

## License

This demo is built on top of NVIDIA Isaac Sim and IsaacLab, which have their own licenses. Please refer to their respective repositories for licensing information.

## Acknowledgments

- NVIDIA Isaac Sim team
- Cosmos Reason2-8B model developers
- IsaacLab framework contributors

## Contact

For issues or questions, please open an issue on GitHub.

---

🤖 **Generated with Claude Code + Isaac Sim**
