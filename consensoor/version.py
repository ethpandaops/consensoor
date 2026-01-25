"""Version info and graffiti builder for consensoor."""

import os
import re
from typing import Optional

CL_CLIENT_CODE = "CO"
CL_CLIENT_NAME = "consensoor"

EL_CLIENT_CODES = {
    "geth": "GE",
    "go-ethereum": "GE",
    "nethermind": "NM",
    "besu": "BU",
    "erigon": "ER",
    "reth": "RH",
    "ethereumjs": "EJ",
}

CL_CLIENT_CODES = {
    "consensoor": "CO",
    "lighthouse": "LH",
    "prysm": "PR",
    "teku": "TK",
    "nimbus": "NB",
    "lodestar": "LS",
    "grandine": "GR",
}


def _get_scm_version() -> tuple[str, str]:
    """Get version and commit from setuptools_scm generated _version.py."""
    try:
        from ._version import __version__, __version_tuple__
        version = __version__
        if __version_tuple__ and len(__version_tuple__) >= 4:
            commit = str(__version_tuple__[3]) if __version_tuple__[3] else ""
            if commit.startswith("g"):
                commit = commit[1:]
        else:
            match = re.search(r'\+g([a-f0-9]+)', version)
            commit = match.group(1) if match else ""
        return version, commit
    except ImportError:
        return os.environ.get("CONSENSOOR_VERSION", "0.1.0"), os.environ.get("CONSENSOOR_COMMIT", "")


def get_cl_version() -> str:
    """Get consensoor version string."""
    version, _ = _get_scm_version()
    return version


def get_cl_commit() -> str:
    """Get consensoor's commit hash."""
    _, commit = _get_scm_version()
    return commit


def get_cl_client_version_info() -> dict:
    """Get CL client version info for engine_getClientVersionV1."""
    version, commit = _get_scm_version()
    return {
        "code": CL_CLIENT_CODE,
        "name": CL_CLIENT_NAME,
        "version": version,
        "commit": f"0x{commit[:8]}" if commit else "0x00000000",
    }


def get_el_code(client_name: str) -> str:
    """Get 2-char code for an EL client name."""
    name_lower = client_name.lower()
    for key, code in EL_CLIENT_CODES.items():
        if key in name_lower:
            return code
    return name_lower[:2].upper()


def build_graffiti(user_graffiti: str, el_client_info: Optional[dict] = None) -> bytes:
    """Build graffiti bytes with EL+CL version info prefix.

    Format (when space allows):
    - 13+ bytes free: EL(2)+commit(4)+CL(2)+commit(4)+space+user = "GEabcdCOxxxx user"
    - 9-12 bytes free: EL(2)+commit(2)+CL(2)+commit(2)+space+user = "GEabCOxx user"
    - 5-8 bytes free: EL(2)+CL(2)+space+user = "GECO user"
    - 3-4 bytes free: CL(2)+space+user = "CO user"
    - <3 bytes free: user graffiti only
    """
    user_bytes = user_graffiti.encode("utf-8")
    max_graffiti = 32

    cl_commit = get_cl_commit()
    cl_code = CL_CLIENT_CODE

    el_code = ""
    el_commit = ""
    if el_client_info:
        el_name = el_client_info.get("name", "")
        el_code = el_client_info.get("code", "") or get_el_code(el_name)
        el_commit_raw = el_client_info.get("commit", "")
        if el_commit_raw.startswith("0x"):
            el_commit = el_commit_raw[2:]
        else:
            el_commit = el_commit_raw

    available = max_graffiti - len(user_bytes)
    if available > 0 and user_bytes:
        available -= 1

    prefix = ""
    if available >= 13 and el_code:
        prefix = f"{el_code}{el_commit[:4]}{cl_code}{cl_commit[:4]}"
    elif available >= 9 and el_code:
        prefix = f"{el_code}{el_commit[:2]}{cl_code}{cl_commit[:2]}"
    elif available >= 5 and el_code:
        prefix = f"{el_code}{cl_code}"
    elif available >= 3:
        prefix = cl_code

    if prefix:
        if user_bytes:
            graffiti = f"{prefix} {user_graffiti}"
        else:
            graffiti = prefix
    else:
        graffiti = user_graffiti

    graffiti_bytes = graffiti.encode("utf-8")[:max_graffiti]
    return graffiti_bytes + b"\x00" * (max_graffiti - len(graffiti_bytes))
