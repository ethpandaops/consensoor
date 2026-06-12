"""State synchronization manager for syncing from upstream beacon node."""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Union

from ..spec.constants import SLOTS_PER_EPOCH
from ..spec.types import FuluBeaconState, ElectraBeaconState, BeaconState, BeaconBlockHeader
from ..crypto import hash_tree_root
from .client import RemoteBeaconClient

if TYPE_CHECKING:
    from ..node import BeaconNode

logger = logging.getLogger(__name__)

AnyBeaconState = Union[BeaconState, FuluBeaconState, ElectraBeaconState]


class StateSyncManager:
    """Manages state synchronization from an upstream beacon node."""

    def __init__(self, client: RemoteBeaconClient, node: "BeaconNode"):
        self.client = client
        self.node = node
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_synced_epoch: int = -1

    async def start(self) -> None:
        """Start the state sync manager."""
        self._running = True
        version = await self.client.get_version()
        logger.info(f"Connected to upstream beacon node: {version}")

        await self.client.subscribe_to_events(
            ["block", "finalized_checkpoint"],
            self._on_event,
        )

        # Pull the finalized state BEFORE returning. node.start() prepares the
        # first payload right after this, and with the state still at genesis
        # that walks process_slots from slot 0 to the wall-clock slot — an
        # hours-long replay that starves the event loop and the sync itself.
        await self._initial_sync()

        self._sync_task = asyncio.create_task(self._periodic_sync())
        logger.info("State sync manager started")

    async def stop(self) -> None:
        """Stop the state sync manager."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        await self.client.close()

    async def _on_event(self, event_type: str, data: dict) -> None:
        """Handle SSE events from upstream."""
        if event_type == "block":
            await self._on_block_event(data)
        elif event_type == "finalized_checkpoint":
            await self._on_finalized_event(data)

    async def _on_block_event(self, data: dict) -> None:
        """Handle a block event from SSE."""
        slot = int(data.get("slot", 0))
        block_root = data.get("block", "")

        logger.info(f"Received block event: slot={slot}, root={block_root[:18]}...")

        slots_per_epoch = SLOTS_PER_EPOCH()
        current_epoch = slot // slots_per_epoch
        slot_in_epoch = slot % slots_per_epoch

        # Determine if we need to sync state
        need_sync = False
        state_slot = int(self.node.state.slot) if self.node.state else 0

        # Sync at epoch boundaries
        if slot_in_epoch == 0 and current_epoch > self._last_synced_epoch:
            logger.info(f"Epoch boundary at slot {slot}, triggering state sync")
            need_sync = True
            self._last_synced_epoch = current_epoch

        # Sync if we're more than 2 slots behind (to keep state fresh for block building)
        elif slot > state_slot + 2:
            logger.info(f"State behind by {slot - state_slot} slots, triggering sync")
            need_sync = True

        if need_sync:
            await self._sync_state_at_slot(slot)

        if slot > self.node.head_slot:
            self.node.head_slot = slot
            if block_root:
                self.node.head_root = bytes.fromhex(block_root.replace("0x", ""))
                await self._update_block_header(block_root)

    async def _update_block_header(self, block_root: str) -> None:
        """Fetch the full SignedBeaconBlock for the given root and store it.

        Without this, our store only has SSZ states from /eth/v2/debug/beacon/states
        and never the actual SignedBeaconBlocks — so /eth/v2/beacon/blocks/{root}
        returns 404, /eth/v1/beacon/headers/head's state_root falls back to the
        zero placeholder, and external clients (dora, validators, peers) can't
        retrieve any block data from us.

        We DO NOT update state.latest_block_header here — that has to come from
        a full state transition (process_block) or a full state sync.
        """
        if not self.node.state:
            return

        try:
            ssz_bytes = await self.client.get_block(block_root)
        except Exception as e:
            logger.warning(f"Failed to fetch block {block_root[:18]}...: {e}")
            return

        signed_block = self._decode_signed_block(ssz_bytes)
        if signed_block is None:
            logger.warning(
                f"Failed to decode signed block {block_root[:18]}... ({len(ssz_bytes)} bytes)"
            )
            return

        try:
            root_bytes = bytes.fromhex(block_root.replace("0x", ""))
            self.node.store.save_block(root_bytes, signed_block)
        except Exception as e:
            logger.warning(f"Failed to save block {block_root[:18]}...: {e}")
            return

        slot = int(signed_block.message.slot)
        logger.debug(
            f"Stored signed block {block_root[:18]}...: slot={slot}, size={len(ssz_bytes)}B"
        )

    def _decode_signed_block(self, ssz_bytes: bytes):
        """Try decoding the SSZ block bytes as each known fork type."""
        from ..spec.types.gloas import SignedBeaconBlock as SignedGloas
        from ..spec.types import SignedElectraBeaconBlock as SignedElectra

        for cls, name in ((SignedGloas, "Gloas"), (SignedElectra, "Electra")):
            try:
                block = cls.decode_bytes(ssz_bytes)
                logger.debug(f"Decoded SignedBeaconBlock as {name}")
                return block
            except Exception:
                continue
        return None

    async def _on_finalized_event(self, data: dict) -> None:
        """Handle a finalized checkpoint event."""
        epoch = int(data.get("epoch", 0))
        block_root = data.get("block", "")
        logger.info(f"Finalized checkpoint: epoch={epoch}, root={block_root[:18]}...")

    async def _periodic_sync(self) -> None:
        """Periodically sync state to keep randao_mixes current."""
        while self._running:
            try:
                await asyncio.sleep(60)
                if not self._running:
                    break
                await self._sync_finality()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic sync error: {e}")

    async def _initial_sync(self) -> None:
        """Perform initial sync on startup.

        Try finalized first (lightweight checkpoint-sync endpoints typically
        only serve finalized) and fall back to head for full beacon nodes.
        """
        logger.info("Performing initial state sync...")
        try:
            await self._sync_state_at_slot("finalized")
            return
        except Exception as e:
            logger.warning(
                f"Initial sync at 'finalized' failed ({e}); trying 'head'"
            )
        try:
            await self._sync_state_at_slot("head")
        except Exception as e:
            logger.error(f"Initial sync failed: {e}")

    async def _sync_state_at_slot(self, slot_or_id: Union[int, str]) -> None:
        """Sync state at a specific slot or state ID. Raises on failure."""
        state_id = str(slot_or_id)
        logger.info(f"Fetching state at {state_id}...")

        state_bytes = await self.client.get_state(state_id)
        state = self._decode_state(state_bytes)

        if state is None:
            raise RuntimeError(f"Failed to decode state from upstream at {state_id}")

        await self._apply_state(state)
        logger.info(
            f"State synced: slot={state.slot}, "
            f"validators={len(state.validators)}"
        )

    def _decode_state(self, state_bytes: bytes) -> Optional[AnyBeaconState]:
        """Decode state bytes, trying different fork types."""
        for state_type, name in [
            (BeaconState, "Gloas"),
            (FuluBeaconState, "Fulu"),
            (ElectraBeaconState, "Electra"),
        ]:
            try:
                state = state_type.decode_bytes(state_bytes)
                logger.debug(f"Decoded state as {name} format")
                return state
            except Exception:
                continue

        logger.error("Failed to decode state with any known format")
        return None

    async def _apply_state(self, synced_state: AnyBeaconState) -> None:
        """Apply synced state to local node state."""
        synced_slot = int(synced_state.slot)

        # Adopt the state wholesale when there is no usable local chain yet
        # (no state, or still sitting at genesis). Merging selected fields
        # into the genesis state leaves the rest (block_roots, participation,
        # exit queue, ...) stale, and without the checkpoint block root
        # anchored in the store the reorg walk falls through the checkpoint
        # chasing pre-checkpoint blocks no peer serves anymore.
        if self.node.state is None or (
            int(self.node.state.slot) == 0 and synced_slot > 0
        ):
            await self._adopt_checkpoint_state(synced_state)
            return

        local_state = self.node.state

        for i, mix in enumerate(synced_state.randao_mixes):
            local_state.randao_mixes[i] = mix

        local_state.slot = synced_state.slot
        local_state.latest_block_header = synced_state.latest_block_header
        local_state.finalized_checkpoint = synced_state.finalized_checkpoint
        local_state.current_justified_checkpoint = synced_state.current_justified_checkpoint
        local_state.previous_justified_checkpoint = synced_state.previous_justified_checkpoint

        if hasattr(local_state, "latest_execution_payload_header") and hasattr(synced_state, "latest_execution_payload_header"):
            local_state.latest_execution_payload_header = synced_state.latest_execution_payload_header
            logger.debug("Updated latest_execution_payload_header from synced state")

        if len(synced_state.validators) != len(local_state.validators):
            logger.info(
                f"Validator count changed: {len(local_state.validators)} -> "
                f"{len(synced_state.validators)}"
            )
            local_state.validators = synced_state.validators
            local_state.balances = synced_state.balances

            if self.node.validator_client:
                self.node.validator_client.update_validator_indices(local_state)

        if hasattr(local_state, "proposer_lookahead") and hasattr(synced_state, "proposer_lookahead"):
            local_state.proposer_lookahead = synced_state.proposer_lookahead
            logger.debug("Updated proposer_lookahead from synced state")

        self.node.head_slot = synced_slot
        # Note: Don't overwrite head_root here - let block events update it with the actual block root

    async def _adopt_checkpoint_state(self, state: AnyBeaconState) -> None:
        """Adopt a checkpoint state as the node's chain anchor.

        Computes the anchor block root per spec (latest_block_header with
        state_root filled in), points head at it, and caches the state under
        that root so _reorg_to's ancestor walk terminates at the checkpoint
        instead of requiring ancestry back to genesis.
        """
        slot = int(state.slot)
        header = state.latest_block_header
        header_state_root = bytes(header.state_root)
        if header_state_root == b"\x00" * 32:
            header_state_root = bytes(hash_tree_root(state))
        anchor_header = BeaconBlockHeader(
            slot=int(header.slot),
            proposer_index=int(header.proposer_index),
            parent_root=bytes(header.parent_root),
            state_root=header_state_root,
            body_root=bytes(header.body_root),
        )
        anchor_root = bytes(hash_tree_root(anchor_header))

        self.node.state = state
        self.node.head_slot = slot
        self.node.head_root = anchor_root
        self.node.store.save_state(anchor_root, state)
        self.node.store.set_head(anchor_root)

        # Best effort: fetch the anchor block itself so req/resp and the
        # beacon API can serve it. Checkpoint providers serve finalized
        # blocks by root.
        try:
            ssz_bytes = await self.client.get_block("0x" + anchor_root.hex())
            signed_block = self._decode_signed_block(ssz_bytes)
            if signed_block is not None:
                self.node.store.save_block(anchor_root, signed_block)
        except Exception as e:
            logger.debug(f"Anchor block fetch failed: {e}")

        if self.node.validator_client:
            self.node.validator_client.update_validator_indices(state)

        logger.info(
            f"Adopted checkpoint state: slot={slot}, "
            f"anchor_block={anchor_root.hex()[:16]}, "
            f"finalized_epoch={int(state.finalized_checkpoint.epoch)}"
        )

    async def _sync_finality(self) -> None:
        """Sync finality checkpoints from upstream."""
        try:
            checkpoints = await self.client.get_finality_checkpoints("head")

            if self.node.state is None:
                return

            finalized = checkpoints.get("finalized", {})
            current_justified = checkpoints.get("current_justified", {})

            if finalized:
                epoch = int(finalized.get("epoch", 0))
                root = bytes.fromhex(finalized.get("root", "0" * 64).replace("0x", ""))
                if epoch > int(self.node.state.finalized_checkpoint.epoch):
                    logger.info(f"Finality advanced: epoch {epoch}")

        except Exception as e:
            logger.error(f"Failed to sync finality: {e}")
