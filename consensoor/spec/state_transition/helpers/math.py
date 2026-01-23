"""Math utility functions for state transition.

Implements: integer_squareroot, xor, saturating_sub
Reference: https://github.com/ethereum/consensus-specs/blob/master/specs/phase0/beacon-chain.md
"""

from typing import Union


def integer_squareroot(n: int) -> int:
    """Return the largest integer x such that x**2 <= n.

    Args:
        n: Non-negative integer

    Returns:
        Floor of the square root of n

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError(f"integer_squareroot requires non-negative input, got {n}")
    if n == 0:
        return 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def xor(bytes_1: bytes, bytes_2: bytes) -> bytes:
    """Return the XOR of two byte sequences of equal length.

    Args:
        bytes_1: First byte sequence (32 bytes expected)
        bytes_2: Second byte sequence (32 bytes expected)

    Returns:
        XOR of the two sequences

    Raises:
        ValueError: If sequences have different lengths
    """
    if len(bytes_1) != len(bytes_2):
        raise ValueError(
            f"xor requires equal length byte sequences, "
            f"got {len(bytes_1)} and {len(bytes_2)}"
        )
    return bytes(a ^ b for a, b in zip(bytes_1, bytes_2))


def saturating_sub(a: int, b: int) -> int:
    """Return max(0, a - b) - subtraction that saturates at zero.

    Args:
        a: Minuend
        b: Subtrahend

    Returns:
        a - b if a >= b, else 0
    """
    if a > b:
        return a - b
    return 0
