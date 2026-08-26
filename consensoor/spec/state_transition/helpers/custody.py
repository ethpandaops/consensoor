"""PeerDAS custody helpers.

Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/fulu/das-core.md
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/gloas/fork-choice.md
"""

from typing import Sequence

from ...constants import NUMBER_OF_COLUMNS, NUMBER_OF_CUSTODY_GROUPS
from ....crypto import sha256

UINT256_MAX = 2**256 - 1


def get_custody_groups(node_id: int, custody_group_count: int) -> Sequence[int]:
    """``get_custody_groups`` from fulu/das-core.md."""
    number_of_custody_groups = NUMBER_OF_CUSTODY_GROUPS()
    assert custody_group_count <= number_of_custody_groups

    # Skip computation if all groups are custodied
    if custody_group_count == number_of_custody_groups:
        return list(range(number_of_custody_groups))

    current_id = int(node_id)
    custody_groups: list[int] = []
    while len(custody_groups) < custody_group_count:
        digest = sha256(current_id.to_bytes(32, "little"))
        custody_group = int.from_bytes(digest[0:8], "little") % number_of_custody_groups
        if custody_group not in custody_groups:
            custody_groups.append(custody_group)
        if current_id == UINT256_MAX:
            # Overflow prevention
            current_id = 0
        else:
            current_id += 1

    assert len(custody_groups) == len(set(custody_groups))
    return sorted(custody_groups)


def compute_columns_for_custody_group(custody_group: int) -> Sequence[int]:
    """``compute_columns_for_custody_group`` from fulu/das-core.md."""
    number_of_custody_groups = NUMBER_OF_CUSTODY_GROUPS()
    assert custody_group < number_of_custody_groups
    columns_per_group = NUMBER_OF_COLUMNS // number_of_custody_groups
    return [number_of_custody_groups * i + custody_group for i in range(columns_per_group)]


def get_custody_column_bits(node_id: int, custody_group_count: int) -> bytes:
    """``get_custody_column_bits`` from gloas/fork-choice.md (specs #5549).

    Returns the SSZ-serialised ``CustodyColumnBits = Bitvector[NUMBER_OF_COLUMNS]``
    (16 bytes, little-endian bit order) marking the columns this node
    custodies. This is what ``notify_forkchoice_updated`` forwards to the
    execution engine as its blob-transaction sampling set (EIP-8070).
    """
    bits = bytearray(NUMBER_OF_COLUMNS // 8)
    for custody_group in get_custody_groups(node_id, custody_group_count):
        for column in compute_columns_for_custody_group(custody_group):
            bits[column // 8] |= 1 << (column % 8)
    return bytes(bits)
