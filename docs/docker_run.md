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
   - Set `DISPLAY` to the active host X11 display, for example `:1`
   - Set `XAUTHORITY_HOST_PATH` to the absolute path reported by `echo $XAUTHORITY`, for example `/run/user/1000/gdm/Xauthority`
   - Keep live keys out of tracked files such as `config/runtime_env.env`
   - Only set `ISAACLAB_LAUNCHER` if auto-detection does not match your image
4. Run the host preflight checker:
   ```bash
   python3 tools/check_setup.py
   ```
   This writes a validation log under `validation_logs/` and catches missing Docker daemon access, GPU issues, and unset required env vars.

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
- The compose stack mounts the host Xauthority file at `/root/.Xauthority` inside the container for X11 authentication.
- On X11 hosts, `xhost +local:root` may still be required before launch.
- The launch scripts prepend `/isaac-sim/python_packages` and `/opt/hac-python` onto `PYTHONPATH` so container-installed Python deps remain importable at runtime.
- Store local secrets in `.env` or `config/runtime_env.local.env`; keep `config/runtime_env.env` as a checked-in template only.

## Troubleshooting

- If the GUI fails to appear, verify `DISPLAY`, `XAUTHORITY_HOST_PATH`, and `/tmp/.X11-unix` forwarding first.
- If the container cannot access the GPU, validate Docker GPU support with a known-good NVIDIA container before debugging this repo.
- If the base image tag cannot be pulled, confirm your NGC access and the exact image tag in `.env`.
- If `docker ps` fails with a permissions error, resolve access to `/var/run/docker.sock` or use a shell with working Docker privileges before building.
