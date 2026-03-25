# Docker Quickstart

Human-AI-Collab runs inside a container built on top of an official Isaac Sim / Isaac Lab image.

## Prerequisites

- Linux
- NVIDIA GPU with working drivers
- Docker
- NVIDIA Container Toolkit
- Access to a valid Isaac Sim / Isaac Lab base image
- `COMMANDER_API_KEY`

## Setup

1. Log in to NGC if needed:
   ```bash
   docker login nvcr.io
   ```
2. Copy the env template:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env`:
   - Set `ISAACLAB_BASE_IMAGE`
   - Set `COMMANDER_API_KEY`
   - Optionally set `COSMOS_BASE_URL` or `COSMOS_CHAT_COMPLETIONS_URL`
   - Only set `ISAACLAB_LAUNCHER` if auto-detection does not match your image

## Build And Run

```bash
docker compose build
docker compose up hac
```

The repo is mounted at `/workspace/Human-AI-Collab` inside the container.

To open an interactive shell in the same runtime:

```bash
docker compose run --rm hac bash
```

## Notes

- `COMMANDER_API_KEY` is required.
- Cosmos is optional.
- If Cosmos is unset, startup logs that Cosmos is bypassed.
- The compose stack uses named volumes for Isaac / Omniverse caches.
- On X11 hosts, `xhost +local:root` may be required before launch.

## Troubleshooting

- If the GUI fails to appear, verify `DISPLAY` and `/tmp/.X11-unix` forwarding first.
- If the container cannot access the GPU, validate Docker GPU support with a known-good NVIDIA container before debugging this repo.
- If the base image tag cannot be pulled, confirm your NGC access and the exact image tag in `.env`.
