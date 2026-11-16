FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/efficient-multimodal-inference-venv
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${VIRTUAL_ENV}/bin:${CUDA_HOME}/bin:${PATH}"

# Persistent model/data cache lives on the RunPod Network Volume.
ENV HF_HOME=/workspace/hf-cache
ENV HF_HUB_CACHE=/workspace/hf-cache/hub
ENV HF_DATASETS_CACHE=/workspace/hf-cache/datasets

# Compilation caches remain on the Pod's local filesystem.
ENV XDG_CACHE_HOME=/root/.cache
ENV VLLM_CACHE_ROOT=/root/.cache/vllm
ENV TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
ENV TRITON_CACHE_DIR=/root/.cache/triton
ENV TMPDIR=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    openssh-server \
    ca-certificates \
    build-essential \
    cuda-nvcc-13-2 \
    && rm -rf /var/lib/apt/lists/*

RUN test -x "${CUDA_HOME}/bin/nvcc" \
    && "${CUDA_HOME}/bin/nvcc" --version

RUN python3.12 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --upgrade pip

WORKDIR /opt/environment-build

COPY docker/environment.lock.txt /opt/environment-build/environment.lock.txt

# Install the exact CUDA-enabled PyTorch and TorchVision builds.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu132 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.14.0+cu132" \
    "torchvision==0.29.0+cu132"

# Preserve the CPU-only TorchAudio build recorded in the environment.
RUN python -m pip install --no-cache-dir --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    "torchaudio==2.11.0+cpu"

# TorchCodec's CUDA wheel is hosted on the PyTorch CUDA wheel index,
# not the default PyPI index.
RUN python -m pip install --no-cache-dir --no-deps \
    --index-url https://download.pytorch.org/whl/cu132 \
    --extra-index-url https://pypi.org/simple \
    "torchcodec==0.17.0+cu132"

# Install every other pinned package from the captured environment.
# --no-deps prevents pip from replacing the already-installed framework builds.
RUN grep -viE '^(torch|torchvision|torchaudio|torchcodec)==' \
    environment.lock.txt > requirements-pinned.txt \
    && python -m pip install --no-cache-dir --no-deps \
       -r requirements-pinned.txt

# Verify the key packages at image-build time.
RUN python - <<'PY'
import torch
import torchvision
import torchaudio
import transformers
import vllm

print("Python environment verification")
print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("TorchAudio:", torchaudio.__version__)
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)
print("CUDA available at runtime:", torch.cuda.is_available())
PY

WORKDIR /workspace/projects/efficient-multimodal-inference

# Keep the RunPod container alive and expose its SSH/web-terminal access.
COPY docker/runpod-start.sh /start.sh
RUN chmod 0755 /start.sh
CMD ["/start.sh"]
