# Runtime image with the locked Python environment. The repository is mounted
# at /workspace/namvis at run time; see README.md for the commands.
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# torch.compile (Triton) builds a small C launcher at run time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/namvis
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project && chmod -R a+rX /opt/venv

# Caches live in the mounted repository so a non-root --user can write them.
# Setting the TorchInductor cache also keeps torch from looking up a user name,
# which fails for a --user uid that has no passwd entry.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace/namvis \
    HOME=/workspace/namvis/.cache/home \
    TORCH_HOME=/workspace/namvis/.cache/torch \
    HF_HOME=/workspace/namvis/.cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/workspace/namvis/.cache/torchinductor
