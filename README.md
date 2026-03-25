# Human-AI-Collab

Human-AI-Collab is a Docker-first project layer on top of NVIDIA Isaac Sim / Isaac Lab.
It provides the demo scene, UI, agent pipeline, and robot-task logic for the Human-AI collaboration demo.

This repo is not a standalone simulator or runtime.
You still need access to an official Isaac Sim / Isaac Lab base image and a Linux host with an NVIDIA GPU.

## What This Repo Contains

- A Franka-based demo scene and startup path
- The custom UI and runtime control flow
- Commander-driven reasoning / control integration
- Optional Cosmos integration
- Local scene hygiene fixes so the demo starts without broken decorative-material references

## Prerequisites

- Linux host
- NVIDIA GPU and working NVIDIA driver stack
- Docker
- NVIDIA Container Toolkit / Docker GPU runtime
- Access to a valid Isaac Sim / Isaac Lab base image, for example an NGC image you can pull locally

## Required And Optional Config

Required:
- `ISAACLAB_BASE_IMAGE`
- `COMMANDER_API_KEY`

Optional:
- `COSMOS_BASE_URL` or `COSMOS_CHAT_COMPLETIONS_URL`
- `COSMOS_API_KEY`
- `MEM0_API_KEY`
- `ISAACLAB_LAUNCHER` if your base image does not expose the usual launcher locations

The repo includes:
- [`.env.example`](.env.example)
- [`config/runtime_env.example`](config/runtime_env.example)

## Quickstart

1. Clone the repo and enter it.
   ```bash
   git clone <your-fork-or-local-copy> Human-AI-Collab
   cd Human-AI-Collab
   ```
2. Copy the Docker env template.
   ```bash
   cp .env.example .env
   ```
3. Edit `.env`:
   - Set `ISAACLAB_BASE_IMAGE` to the exact image tag you can pull
   - Set `COMMANDER_API_KEY`
   - Optionally set a Cosmos endpoint
4. Log in to NGC if the base image is hosted on `nvcr.io`.
   ```bash
   docker login nvcr.io
   ```
5. Build the container image.
   ```bash
   docker compose build
   ```
6. Launch the demo.
   ```bash
   docker compose up hac
   ```
7. Stop the stack when finished.
   ```bash
   docker compose down
   ```

Cosmos is optional.
If no Cosmos endpoint is configured, the app logs that it is bypassing Cosmos and continues on the commander-only path.

## GUI / X11 Notes

- This repo expects the base Isaac runtime to provide the GUI stack.
- On Linux with X11 forwarding, you may need:
  ```bash
  xhost +local:root
  ```
- If the app starts but no window appears, verify `DISPLAY`, `/tmp/.X11-unix`, NVIDIA Container Toolkit, and host GPU access first.

## Known Limitations

- The project depends on an external Isaac Sim / Isaac Lab runtime image; this repo does not vendor that runtime.
- GUI startup still depends on host X11 / display configuration.
- Cosmos-enabled validation requires a real endpoint; if unset, that path is intentionally skipped.

## Additional Docs

- Docker quickstart: [docs/docker_run.md](docs/docker_run.md)
- AI-assistant setup prompt: [docs/setup_with_ai_assistant.md](docs/setup_with_ai_assistant.md)

## Maintainer Note Before Public Release

- Any provider keys that were ever exposed during development must be rotated outside this repo before publication.
- If git history contains secret material, clean that history before any public GitHub release.
- Do not commit `.env`, `config/runtime_env.local.*`, or any live credentials.
