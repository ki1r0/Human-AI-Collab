# Setup With An AI Assistant

This prompt is for users who want a coding assistant to verify the host, prepare local config safely, build the Docker image, and launch the demo without hardcoding secrets.

## Suggested Prompt

```text
You are working in the local Human-AI-Collab repo.

This repo is a project layer on top of NVIDIA Isaac Sim / Isaac Lab. It is not a standalone simulator/runtime. Do not try to reimplement Isaac Sim or Isaac Lab inside this repo.

Your job is to help me set up and validate the Docker-first workflow safely.

Requirements:
- Do not hardcode or print secret values.
- Ask me to provide COMMANDER_API_KEY securely when needed.
- Treat Isaac Sim / Isaac Lab as external runtime dependencies supplied by the base container image.
- Prefer practical validation over unrelated refactors.

Please do the following:
1. Verify host prerequisites:
   - Linux environment
   - Docker installed
   - NVIDIA GPU visible on host
   - NVIDIA Container Toolkit / Docker GPU support working
2. Verify I can access the configured Isaac Lab / Isaac Sim base image.
3. Prepare `.env` from `.env.example` if needed.
4. Make sure `.env` clearly sets:
   - `ISAACLAB_BASE_IMAGE`
   - `COMMANDER_API_KEY`
   - optional Cosmos endpoint only if I provide one
5. Never write real secrets into tracked files or example templates.
6. Build the image with:
   - `docker compose build`
7. Launch the app with:
   - `docker compose up hac`
8. Validate startup from logs:
   - container starts
   - scene opens automatically
   - required prims load
   - Cosmos is bypassed cleanly if not configured
9. If GUI/X11 issues appear, explain the exact Linux/X11 troubleshooting steps without changing unrelated project code.
10. Save validation logs under `validation_logs/`.

If something is blocked by missing credentials, missing NGC access, or missing host GPU/X11 setup, stop and tell me exactly what manual action is required.
```
