"""EIP-2335 keystore loading and decryption."""

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt, PBKDF2
from Crypto.Hash import SHA256

from ..crypto import pubkey_from_privkey
from .types import ValidatorKey

logger = logging.getLogger(__name__)


def _decrypt_keystore(keystore: dict, password: str) -> int:
    """Decrypt an EIP-2335 keystore and return the private key as int."""
    crypto = keystore["crypto"]
    kdf = crypto["kdf"]
    cipher = crypto["cipher"]
    checksum = crypto["checksum"]

    kdf_params = kdf["params"]
    if kdf["function"] == "scrypt":
        decryption_key = scrypt(
            password.encode("utf-8"),
            bytes.fromhex(kdf_params["salt"]),
            key_len=32,
            N=kdf_params["n"],
            r=kdf_params["r"],
            p=kdf_params["p"],
        )
    elif kdf["function"] == "pbkdf2":
        decryption_key = PBKDF2(
            password.encode("utf-8"),
            bytes.fromhex(kdf_params["salt"]),
            dkLen=32,
            count=kdf_params["c"],
            hmac_hash_module=SHA256,
        )
    else:
        raise ValueError(f"Unsupported KDF: {kdf['function']}")

    dk_slice = decryption_key[16:32]
    cipher_message = bytes.fromhex(cipher["message"])
    checksum_message = bytes.fromhex(checksum["message"])

    pre_image = dk_slice + cipher_message
    computed_checksum = hashlib.sha256(pre_image).digest()
    if computed_checksum != checksum_message:
        raise ValueError("Invalid password or corrupted keystore")

    cipher_params = cipher["params"]
    if cipher["function"] == "aes-128-ctr":
        iv = bytes.fromhex(cipher_params["iv"])
        aes_key = decryption_key[:16]
        aes = AES.new(aes_key, AES.MODE_CTR, nonce=b"", initial_value=iv)
        secret = aes.decrypt(cipher_message)
    else:
        raise ValueError(f"Unsupported cipher: {cipher['function']}")

    return int.from_bytes(secret, "big")


def load_keystore(keystore_path: Union[str, Path], password: str) -> ValidatorKey:
    """Load a single keystore file."""
    with open(keystore_path, "r") as f:
        keystore = json.load(f)

    privkey = _decrypt_keystore(keystore, password)
    pubkey = pubkey_from_privkey(privkey)
    expected_pubkey = bytes.fromhex(keystore["pubkey"])

    if pubkey != expected_pubkey:
        raise ValueError(
            f"Public key mismatch: derived {pubkey.hex()}, expected {expected_pubkey.hex()}"
        )

    logger.debug(f"Loaded validator key: {pubkey.hex()[:16]}...")
    return ValidatorKey(pubkey=pubkey, privkey=privkey)


def load_keystores_from_dir(
    keystores_dir: Union[str, Path],
    secrets_dir: Union[str, Path],
) -> list[ValidatorKey]:
    """Load keystores from a directory with separate secrets directory.

    Supports multiple directory structures:
    1. Lighthouse/Nimbus: keystores_dir/keystore-*.json
    2. Teku: keystores_dir/<pubkey>/keystore.json or keystores_dir/<pubkey>/keystore-*.json

    Secrets directory contains files named by pubkey (0x...) containing the password.

    EIP-2335 keystores typically use scrypt with N=262144, which is intentionally
    slow (~1–3s per key). With 128 keystores that's minutes of serial work. We
    fan out across all CPU cores via a ThreadPoolExecutor — pycryptodome's
    scrypt releases the GIL during the inner loop so threads scale.
    """
    keystores_path = Path(keystores_dir)
    secrets_path = Path(secrets_dir)
    keys: list[ValidatorKey] = []

    logger.debug(f"Looking for keystores in: {keystores_path}")
    logger.debug(f"Secrets directory: {secrets_path}")

    if not keystores_path.exists():
        logger.error(f"Keystores directory does not exist: {keystores_path}")
        return keys

    if not secrets_path.exists():
        logger.error(f"Secrets directory does not exist: {secrets_path}")
        return keys

    keystore_files: list[Path] = []
    for pattern in ["keystore*.json", "*/keystore*.json", "*/keystore.json", "0x*.json"]:
        found = list(keystores_path.glob(pattern))
        logger.debug(f"Pattern '{pattern}' found {len(found)} files")
        keystore_files.extend(found)

    keystore_files = sorted(set(keystore_files))
    logger.info(f"Found {len(keystore_files)} keystore files")

    # First pass: read JSON, locate password, build (path, password) tuples.
    # We do this serially since it's cheap (a few JSON parses + path lookups).
    work_items: list[tuple[Path, str]] = []
    for keystore_file in keystore_files:
        try:
            with open(keystore_file, "r") as f:
                keystore = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {keystore_file}: {e}")
            continue

        pubkey_hex = keystore.get("pubkey", "")
        if not pubkey_hex:
            logger.warning(f"No pubkey in keystore {keystore_file}, skipping")
            continue

        if not pubkey_hex.startswith("0x"):
            pubkey_hex = "0x" + pubkey_hex

        password_file = None
        for candidate in [
            secrets_path / pubkey_hex,
            secrets_path / pubkey_hex[2:],
            secrets_path / (pubkey_hex + ".txt"),
            secrets_path / (pubkey_hex[2:] + ".txt"),
        ]:
            if candidate.exists():
                password_file = candidate
                break

        if password_file is None:
            logger.warning(f"No password found for {pubkey_hex[:18]}..., skipping")
            continue

        password = password_file.read_text().strip()
        work_items.append((keystore_file, password))

    if not work_items:
        return keys

    # Second pass: parallel decrypt. Cap the worker count so we don't allocate
    # a giant scrypt RAM footprint × cpu_count (each scrypt invocation is
    # ~128 MiB at N=262144).
    cpu = os.cpu_count() or 4
    max_workers = max(2, min(cpu, len(work_items)))
    logger.info(
        f"Decrypting {len(work_items)} keystores in parallel with {max_workers} workers"
    )

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="keystore") as pool:
        futures = {
            pool.submit(load_keystore, path, pw): path for path, pw in work_items
        }
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                keys.append(fut.result())
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")

    elapsed = time.monotonic() - t0
    logger.info(f"Decrypted {len(keys)} keystores in {elapsed:.1f}s")
    return keys


def load_keystores_teku_style(keys_spec: str) -> list[ValidatorKey]:
    """Load keystores using Teku-style specification.

    Format: keystores_dir:secrets_dir or keystore_file:password_file
    Multiple specs can be separated by commas.
    """
    keys = []

    logger.info(f"Loading keystores from spec: {keys_spec}")

    for spec in keys_spec.split(","):
        spec = spec.strip()
        if not spec:
            continue

        if ":" not in spec:
            logger.error(f"Invalid key spec (missing ':'): {spec}")
            continue

        keystore_part, secret_part = spec.split(":", 1)
        keystore_path = Path(keystore_part)
        secret_path = Path(secret_part)

        logger.info(f"Keystore path: {keystore_path} (exists: {keystore_path.exists()}, is_dir: {keystore_path.is_dir() if keystore_path.exists() else 'N/A'})")
        logger.info(f"Secret path: {secret_path} (exists: {secret_path.exists()}, is_dir: {secret_path.is_dir() if secret_path.exists() else 'N/A'})")

        if keystore_path.is_dir():
            loaded = load_keystores_from_dir(keystore_path, secret_path)
            logger.info(f"Loaded {len(loaded)} keys from directory {keystore_path}")
            keys.extend(loaded)
        elif keystore_path.is_file():
            try:
                password = secret_path.read_text().strip()
                key = load_keystore(keystore_path, password)
                keys.append(key)
            except Exception as e:
                logger.error(f"Failed to load keystore {keystore_path}: {e}")
        else:
            logger.error(f"Keystore path not found: {keystore_path}")

    return keys
