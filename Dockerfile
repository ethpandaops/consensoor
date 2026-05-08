# --- Stage 1: build consensoor wheel -----------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir build setuptools_scm

COPY .git/ .git/
COPY pyproject.toml .
COPY consensoor/ consensoor/

RUN python -m build --wheel

# --- Stage 2: build the Rust p2p wheel (consensoor-p2p-rs) -------------------
FROM rust:1.83-slim AS rust-builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-dev pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages "maturin>=1.7,<2.0"

COPY consensoor-p2p-rs/ consensoor-p2p-rs/

RUN cd consensoor-p2p-rs && python3 -m maturin build --release --out /tmp/wheels

# --- Stage 3: runtime --------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libleveldb-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/dist/*.whl /tmp/
COPY --from=rust-builder /tmp/wheels/*.whl /tmp/

# Install consensoor + its runtime deps + the native Rust p2p wheel.
# We deliberately do NOT install py-libp2p anymore — consensoor's host.py
# now talks to consensoor_p2p (rust) directly.
RUN pip install --no-cache-dir --no-deps /tmp/consensoor*.whl /tmp/consensoor_p2p*.whl && \
    pip install --no-cache-dir \
        "remerkleable>=0.1.28" \
        "blspy>=2.0.0" \
        "py_ecc>=7.0.0" \
        "plyvel>=1.5.0" \
        "prometheus_client>=0.20.0" \
        pycryptodome aiohttp click python-snappy coincurve rlp pyjwt pyyaml && \
    rm /tmp/*.whl

EXPOSE 9000/tcp
EXPOSE 5052/tcp
EXPOSE 8008/tcp

ENTRYPOINT ["consensoor"]
CMD ["run", "--help"]
