FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir build

COPY pyproject.toml .
COPY consensoor/ consensoor/

RUN python -m build --wheel

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    libgmp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 consensoor

# Install py-libp2p from PR #1109 (fixes MPLEX race condition with rust-libp2p)
# See: https://github.com/libp2p/py-libp2p/pull/1109
# Also fix gossipsub protobuf bug: messageIDs should be bytes, not string
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler && \
    git clone --depth 1 https://github.com/libp2p/py-libp2p.git /tmp/py-libp2p && \
    cd /tmp/py-libp2p && \
    git fetch origin pull/1109/head:pr-1109 && \
    git checkout pr-1109 && \
    sed -i 's/repeated string messageIDs/repeated bytes messageIDs/g' libp2p/pubsub/pb/rpc.proto && \
    protoc --python_out=. libp2p/pubsub/pb/rpc.proto && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/py-libp2p && \
    apt-get purge -y protobuf-compiler && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/dist/*.whl /tmp/

# Install consensoor (--no-deps to avoid overwriting patched libp2p)
# Pin py_ecc>=7.0.0 and remerkleable>=0.1.28 to avoid deprecation warnings from old eth-utils
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && \
    pip install --no-cache-dir "remerkleable>=0.1.28" "py_ecc>=7.0.0" pycryptodome aiohttp click python-snappy coincurve rlp trio varint pyjwt pyyaml && \
    rm /tmp/*.whl

USER consensoor

EXPOSE 9000/udp
EXPOSE 5052/tcp

ENTRYPOINT ["consensoor"]
CMD ["run", "--help"]
