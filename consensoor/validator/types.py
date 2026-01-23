"""Validator data types."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidatorKey:
    """A validator's key pair."""

    pubkey: bytes
    privkey: int
    validator_index: Optional[int] = None


@dataclass
class ProposerDuty:
    """Proposer duty for a slot."""

    validator_index: int
    slot: int
    pubkey: bytes


@dataclass
class AttesterDuty:
    """Attester duty for a slot."""

    validator_index: int
    slot: int
    committee_index: int
    committee_length: int
    committees_at_slot: int
    validator_committee_index: int
    pubkey: bytes
