"""KZG / PeerDAS helpers backed by c-kzg-4844 (``ckzg``).

The trusted setup shipped with consensus-specs is vendored in
``trusted_setup_4096.json`` (identical for mainnet and minimal presets) and
converted to the text layout ``ckzg.load_trusted_setup`` expects on first use.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Sequence

import ckzg

BYTES_PER_BLOB = 131072
BYTES_PER_CELL = 2048
CELLS_PER_EXT_BLOB = 128
BYTES_PER_COMMITMENT = 48
BYTES_PER_PROOF = 48

_settings = None
_lock = threading.Lock()


def _setup_path() -> Path:
    override = os.environ.get("CONSENSOOR_TRUSTED_SETUP")
    if override:
        return Path(override)
    return Path(__file__).with_name("trusted_setup_4096.json")


def load_trusted_setup(precompute: int = 0):
    """Load (once) and return the ckzg settings object."""
    global _settings
    if _settings is not None:
        return _settings
    with _lock:
        if _settings is not None:
            return _settings
        with _setup_path().open() as f:
            data = json.load(f)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
            tf.write(f"{len(data['g1_lagrange'])}\n")
            tf.write(f"{len(data['g2_monomial'])}\n")
            for point in data["g1_lagrange"]:
                tf.write(point[2:] + "\n")
            for point in data["g2_monomial"]:
                tf.write(point[2:] + "\n")
            for point in data["g1_monomial"]:
                tf.write(point[2:] + "\n")
            path = tf.name
        try:
            _settings = ckzg.load_trusted_setup(path, precompute)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    return _settings


def compute_cells_and_kzg_proofs(blob: bytes) -> tuple[list[bytes], list[bytes]]:
    """Return (cells, proofs) for one blob: 128 cells of 2048 bytes and 128 proofs."""
    if len(blob) != BYTES_PER_BLOB:
        raise ValueError(f"blob must be {BYTES_PER_BLOB} bytes, got {len(blob)}")
    cells, proofs = ckzg.compute_cells_and_kzg_proofs(blob, load_trusted_setup())
    return [bytes(c) for c in cells], [bytes(p) for p in proofs]


def verify_cell_kzg_proof_batch(
    commitments: Sequence[bytes],
    cell_indices: Sequence[int],
    cells: Sequence[bytes],
    proofs: Sequence[bytes],
) -> bool:
    """Batch-verify cells against commitments (one entry per cell)."""
    if not cells:
        return True
    try:
        return bool(
            ckzg.verify_cell_kzg_proof_batch(
                [bytes(c) for c in commitments],
                [int(i) for i in cell_indices],
                [bytes(c) for c in cells],
                [bytes(p) for p in proofs],
                load_trusted_setup(),
            )
        )
    except Exception:
        # c-kzg raises on malformed inputs (e.g. a proof that is not a valid
        # G1 point); for validation purposes that's simply "invalid".
        return False


def blob_to_kzg_commitment(blob: bytes) -> bytes:
    return bytes(ckzg.blob_to_kzg_commitment(blob, load_trusted_setup()))
