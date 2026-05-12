"""Field-by-field beacon-state diff between two HTTP endpoints.

Use this when our state_root diverges from a peer's (the usual symptom:
`STATE_ROOT_MISMATCH at slot=N` warning in consensoor logs, or reorg
replay failing on parent_root). It pulls full state SSZ from both
endpoints, decodes with consensoor's Gloas types, and reports which
top-level field roots differ. One run usually pinpoints the bug.

Run from the repo root so the `consensoor` package resolves:

    python debug_state_diff.py 5 http://127.0.0.1:32997 http://127.0.0.1:32985

(consensoor first, lighthouse second — port numbers come from
`kurtosis enclave inspect <enclave>`).

Past finds: slot-5 divergence narrowed to `next_withdrawal_validator_index`
(0 vs 256), which pointed at the Electra/Gloas branch in
`process_withdrawals` using `processed_validators_sweep_count` instead of
`MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP`. The diff alone made the bug
findable; without it we'd have been bisecting the whole state transition.

Notes:
- Assumes Gloas state shape. For a state from a pre-Gloas fork, swap the
  import to the appropriate `consensoor.spec.types.<fork>` module.
- Some primitive-typed fields fall back to a left-padded little-endian
  encoding for comparison; that's just a stable representation so the
  diff catches them, not a true SSZ root.
"""

import sys
import urllib.request

from consensoor.spec.types.gloas import BeaconState
from consensoor.crypto import hash_tree_root


def fetch_state(base_url: str, state_id: str) -> bytes:
    url = f"{base_url.rstrip('/')}/eth/v2/debug/beacon/states/{state_id}"
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def field_root(value) -> bytes:
    if hasattr(value, "hash_tree_root"):
        return bytes(value.hash_tree_root())
    if isinstance(value, bool):
        return (b"\x01" if value else b"\x00") + b"\x00" * 31
    if isinstance(value, int):
        return int(value).to_bytes(32, "little", signed=False)
    if isinstance(value, (bytes, bytearray)):
        b = bytes(value)
        return b if len(b) == 32 else b.ljust(32, b"\x00")[:32]
    raise TypeError(f"unrooted type: {type(value)}")


def main():
    if len(sys.argv) != 4:
        print("usage: debug_state_diff.py <slot_or_root> <url_a> <url_b>")
        print("       (a = consensoor, b = reference peer; order is just labels)")
        sys.exit(2)

    state_id, url_a, url_b = sys.argv[1], sys.argv[2], sys.argv[3]

    print(f"fetching state {state_id} from {url_a} ...", flush=True)
    ssz_a = fetch_state(url_a, state_id)
    print(f"  got {len(ssz_a)} bytes", flush=True)

    print(f"fetching state {state_id} from {url_b} ...", flush=True)
    ssz_b = fetch_state(url_b, state_id)
    print(f"  got {len(ssz_b)} bytes", flush=True)

    state_a = BeaconState.decode_bytes(ssz_a)
    state_b = BeaconState.decode_bytes(ssz_b)

    root_a = bytes(hash_tree_root(state_a))
    root_b = bytes(hash_tree_root(state_b))
    print(f"\nstate_root A = {root_a.hex()}")
    print(f"state_root B = {root_b.hex()}")
    if root_a == root_b:
        print("STATES MATCH — nothing to diff.")
        return

    diffs = []
    for name in BeaconState.fields().keys():
        va = getattr(state_a, name)
        vb = getattr(state_b, name)
        try:
            ra = field_root(va)
            rb = field_root(vb)
        except TypeError as exc:
            print(f"  {name}: skipped ({exc})")
            continue
        if ra == rb:
            continue
        diffs.append((name, ra, rb, va, vb))

    print(f"\n=== {len(diffs)} differing field(s) ===")
    for name, ra, rb, va, vb in diffs:
        print(f"\n--- {name} ---")
        print(f"  A: {ra.hex()}")
        print(f"  B: {rb.hex()}")
        if hasattr(va, "__len__") and hasattr(vb, "__len__"):
            try:
                print(f"  len A={len(va)}  len B={len(vb)}")
            except TypeError:
                pass
        if isinstance(va, int) and isinstance(vb, int):
            print(f"  val A={va}  val B={vb}")


if __name__ == "__main__":
    main()
