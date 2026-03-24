# Docker run path

This project is packaged as a layer on top of an official Isaac Sim / Isaac Lab runtime image.

## Prerequisites
- Linux host with NVIDIA GPU drivers working with Docker
- Docker with GPU support (`nvidia-container-toolkit`)
- Access to the chosen Isaac Sim / Isaac Lab container image
- `COMMANDER_API_KEY` configured at runtime

## Setup
0. Authenticate to NGC if your base image is hosted on `nvcr.io`:
```bash
docker login nvcr.io
```
1. Copy `.env.example` to `.env`
2. Set `ISAACLAB_BASE_IMAGE` to the exact official Isaac Lab / Isaac Sim image tag you can access
3. Set `COMMANDER_API_KEY`
4. If you use Cosmos, set `COSMOS_BASE_URL` or `COSMOS_CHAT_COMPLETIONS_URL`
5. If your Isaac Lab launcher is not `/isaac-sim/isaaclab.sh`, override `ISAACLAB_LAUNCHER`

## Build
```bash
docker compose build
```

## Run the demo
```bash
docker compose up hac
```

For X11 GUI forwarding on Linux, you may also need:
```bash
xhost +local:root
```

## Open a shell inside the runtime
```bash
docker compose run --rm hac bash
```

## Notes
- `COMMANDER_API_KEY` is required.
- Cosmos is optional. If unset, the app bypasses Cosmos cleanly.
- The project runs from `/workspace/Human-AI-Collab` inside the container.
- The compose file mounts persistent Isaac Sim / Omniverse caches using named volumes.
