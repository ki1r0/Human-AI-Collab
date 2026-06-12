# Cloud GPU Deployment

This deployment runs Human-AI-Collab and Isaac Lab on a Linux cloud GPU host.
The local computer only displays and controls the app through a Chromium browser.

## Recommended Host

The simplest supported path is an NVIDIA Brev VM with:

- 1x NVIDIA L40S GPU
- Ubuntu Linux
- At least 32 GB RAM
- At least 150 GB free disk space

The cloud GPU must support NVIDIA NVENC. Do not use A100 for livestreaming.

## Security

Isaac Sim WebRTC and the browser viewer do not provide authentication or encryption.
At the cloud-provider firewall, allow these ports only from your current public IP:

- TCP `8210`: browser viewer
- TCP `49100`: WebRTC signaling
- UDP `47998`: WebRTC media
- TCP `22`: SSH

Do not open the streaming ports to `0.0.0.0/0`.

## Brev Setup

1. Create an NVIDIA Brev VM in VM Mode with one L40S GPU.
2. Restrict ports `22`, `8210`, `49100`, and `47998` to your current public IP.
3. Open the Brev terminal or connect over SSH.
4. Install Docker and NVIDIA Container Toolkit if the image does not already include them.
5. Verify GPU containers:

   ```bash
   docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
   ```

## Project Setup

Clone the repository on the cloud host:

```bash
git clone https://github.com/ki1r0/Human-AI-Collab.git
cd Human-AI-Collab
cp .env.cloud.example .env.cloud
```

Edit `.env.cloud` and set:

```dotenv
PUBLIC_IP=<cloud-vm-public-ip>
COMMANDER_API_KEY=<your-key>
ISAACLAB_BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.0
```

If NGC authentication is required:

```bash
docker login nvcr.io
```

Use username `$oauthtoken` and an NGC API key as the password.

## Launch

```bash
bash tools/cloud_up.sh
```

Follow app logs:

```bash
docker compose --env-file .env.cloud -f docker-compose.cloud.yml logs -f hac-cloud
```

Open the viewer locally:

```text
http://<cloud-vm-public-ip>:8210
```

Only one browser tab or streaming client can connect to an Isaac Sim instance at a time.

If the browser viewer is incompatible with the Isaac Lab image, use the native Isaac Sim
WebRTC Streaming Client and connect it to the cloud VM public IP. The native client uses
TCP `49100` and UDP `47998`.

## Stop

```bash
bash tools/cloud_down.sh
```

Stop or delete the cloud VM when it is not in use to avoid ongoing GPU charges.
