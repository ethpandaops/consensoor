FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir build setuptools_scm

COPY .git/ .git/
COPY pyproject.toml .
COPY consensoor/ consensoor/

RUN python -m build --wheel

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    libgmp-dev \
    libleveldb-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone and patch py-libp2p (fix messageIDs type from string to bytes for gossipsub)
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler && \
    git clone --depth 1 https://github.com/libp2p/py-libp2p.git /tmp/py-libp2p && \
    cd /tmp/py-libp2p && \
    sed -i 's/repeated string messageIDs/repeated bytes messageIDs/g' libp2p/pubsub/pb/rpc.proto && \
    protoc --python_out=. libp2p/pubsub/pb/rpc.proto && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/py-libp2p && \
    apt-get purge -y protobuf-compiler && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/dist/*.whl /tmp/

# Install consensoor (--no-deps to avoid overwriting patched libp2p)
# Pin py_ecc>=7.0.0 and remerkleable>=0.1.28 to avoid deprecation warnings from old eth-utils
# blspy provides fast BLS cryptography (C/assembly instead of pure Python py_ecc)
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && \
    pip install --no-cache-dir "remerkleable>=0.1.28" "blspy>=2.0.0" "py_ecc>=7.0.0" "plyvel>=1.5.0" "prometheus_client>=0.20.0" pycryptodome aiohttp click python-snappy coincurve rlp trio varint pyjwt pyyaml && \
    rm /tmp/*.whl

EXPOSE 9000/udp
EXPOSE 5052/tcp
EXPOSE 8008/tcp

ENTRYPOINT ["consensoor"]
CMD ["run", "--help"]
