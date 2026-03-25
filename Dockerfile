ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.0
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV HAC_REPO_ROOT=/workspace/Human-AI-Collab \
    ISAACLAB_ROOT=/isaac-sim \
    PYTHONUNBUFFERED=1

WORKDIR ${HAC_REPO_ROOT}

COPY requirements.txt /tmp/hac-requirements.txt
RUN if [[ -x /isaac-sim/python.sh ]]; then \
        mkdir -p /isaac-sim/python_packages; \
        /isaac-sim/python.sh -m pip install --no-cache-dir --target /isaac-sim/python_packages -r /tmp/hac-requirements.txt; \
    elif command -v python3 >/dev/null 2>&1; then \
        mkdir -p /opt/hac-python; \
        python3 -m pip install --no-cache-dir --target /opt/hac-python -r /tmp/hac-requirements.txt; \
    else \
        echo "[ERROR] No usable Python runtime found in base image." >&2; exit 1; \
    fi

COPY . ${HAC_REPO_ROOT}

RUN chmod +x ${HAC_REPO_ROOT}/run_main.sh \
    ${HAC_REPO_ROOT}/docker/entrypoint.sh \
    ${HAC_REPO_ROOT}/docker/run_demo.sh

ENTRYPOINT ["/workspace/Human-AI-Collab/docker/entrypoint.sh"]
CMD ["/workspace/Human-AI-Collab/docker/run_demo.sh"]
