ARG VLLM_IMAGE=vllm/vllm-openai:v0.34.0@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1
FROM ${VLLM_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/latentport
COPY . .
RUN uv pip install --system . ./integrations/vllm \
    && lib="$(python3 -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm").origin).parent.parent)')" \
    && cd "$lib" \
    && patch="$(python3 -c 'from importlib.resources import files; print(files("latentport_vllm") / "patches" / "vllm-0.34.0.patch")')" \
    && git apply --check "$patch" \
    && git apply "$patch"

ENTRYPOINT ["latentport-vllm"]
