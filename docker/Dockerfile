# syntax=docker/dockerfile:1
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS wheel
WORKDIR /src
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src
RUN python -m pip install --no-cache-dir build && python -m build --wheel --outdir /wheels

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" LATENTPORT_HOME=/opt/latentport
COPY docker/constraints.txt /constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
COPY --from=wheel /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /constraints.txt "latentport[gguf] @ file:///wheels/latentport-0.2.1-py3-none-any.whl" \
    && latentport install-native --no-cuda \
    && python -c "from latentport._native.engine import library_path; import shutil; shutil.copytree(library_path().parent, '/opt/native-lib')"

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867 AS gguf
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 latentport && mkdir /cache && chown latentport /cache
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=dependencies /opt/native-lib /opt/native-lib
ENV PATH="/opt/venv/bin:$PATH" LATENTPORT_HOME=/cache \
    LATENTPORT_NATIVE_LIB=/opt/native-lib/libllama.so PYTHONUNBUFFERED=1
USER latentport
EXPOSE 8768
ENTRYPOINT ["latentport"]
CMD ["demo", "--host", "0.0.0.0", "--no-browser"]

# BF16 kernels are compiled during the image build, not by the end user.
FROM dependencies AS bf16
ENV TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=2 CAUSAL_CONV1D_FORCE_BUILD=TRUE
RUN pip install --no-cache-dir packaging ninja 'setuptools<82' wheel \
    && pip install --no-cache-dir --no-build-isolation -c /constraints.txt "latentport[bf16]==0.2.1" \
    && pip check \
    && useradd --create-home --uid 10001 latentport && mkdir /cache && chown latentport /cache
ENV LATENTPORT_HOME=/cache LATENTPORT_NATIVE_LIB=/opt/native-lib/libllama.so PYTHONUNBUFFERED=1
USER latentport
ENTRYPOINT ["latentport"]
CMD ["doctor", "--backend", "bf16"]
