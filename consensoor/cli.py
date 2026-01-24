"""CLI entry point for consensoor."""

import asyncio
import logging
import sys
from typing import Optional

import click

from .config import Config


def setup_logging(level: str) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if level.upper() != "DEBUG":
        logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


@click.group()
@click.version_option(package_name="consensoor")
def cli():
    """Consensoor - Lightweight Python consensus layer client."""
    pass


@cli.command()
@click.option(
    "--engine-api-url",
    default="http://localhost:8551",
    help="Engine API URL for execution layer communication",
    envvar="CONSENSOOR_ENGINE_API_URL",
)
@click.option(
    "--jwt-secret",
    type=click.Path(exists=True),
    help="Path to JWT secret file for Engine API authentication",
    envvar="CONSENSOOR_JWT_SECRET",
)
@click.option(
    "--genesis-state",
    type=click.Path(exists=True),
    required=True,
    help="Path to genesis state SSZ file",
    envvar="CONSENSOOR_GENESIS_STATE",
)
@click.option(
    "--network-config",
    type=click.Path(exists=True),
    help="Path to network config YAML file (fetches from upstream if not provided)",
    envvar="CONSENSOOR_NETWORK_CONFIG",
)
@click.option(
    "--preset",
    default="mainnet",
    type=click.Choice(["mainnet", "minimal"], case_sensitive=False),
    help="Preset to use (mainnet or minimal)",
    envvar="CONSENSOOR_PRESET",
)
@click.option(
    "--p2p-port",
    default=9000,
    type=int,
    help="Port for P2P gossip network",
    envvar="CONSENSOOR_P2P_PORT",
)
@click.option(
    "--p2p-host",
    default="0.0.0.0",
    help="Host to bind P2P network",
    envvar="CONSENSOOR_P2P_HOST",
)
@click.option(
    "--beacon-api-port",
    default=5052,
    type=int,
    help="Port for Beacon API HTTP server",
    envvar="CONSENSOOR_BEACON_API_PORT",
)
@click.option(
    "--peers",
    multiple=True,
    help="Peer addresses (can be specified multiple times)",
    envvar="CONSENSOOR_PEERS",
)
@click.option(
    "--data-dir",
    default="./data",
    type=click.Path(),
    help="Directory for storing data",
    envvar="CONSENSOOR_DATA_DIR",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging level",
    envvar="CONSENSOOR_LOG_LEVEL",
)
@click.option(
    "--validator-keys",
    help="Validator keys spec (Teku-style: keystores_dir:secrets_dir)",
    envvar="CONSENSOOR_VALIDATOR_KEYS",
)
@click.option(
    "--fee-recipient",
    default="0x" + "00" * 20,
    help="Fee recipient address for block proposals",
    envvar="CONSENSOOR_FEE_RECIPIENT",
)
@click.option(
    "--graffiti",
    default="consensoor",
    help="Graffiti string for block proposals",
    envvar="CONSENSOOR_GRAFFITI",
)
@click.option(
    "--checkpoint-sync-url",
    help="URL of upstream beacon node for checkpoint sync (e.g., http://localhost:5052)",
    envvar="CONSENSOOR_CHECKPOINT_SYNC_URL",
)
@click.option(
    "--bootnodes",
    multiple=True,
    help="Bootnode ENRs or multiaddrs (can be specified multiple times)",
    envvar="CONSENSOOR_BOOTNODES",
)
@click.option(
    "--supernode",
    is_flag=True,
    default=False,
    help="Run as supernode (custody all 128 data column groups for PeerDAS/Fulu)",
    envvar="CONSENSOOR_SUPERNODE",
)
def run(
    engine_api_url: str,
    jwt_secret: Optional[str],
    genesis_state: str,
    network_config: Optional[str],
    preset: str,
    p2p_port: int,
    p2p_host: str,
    beacon_api_port: int,
    peers: tuple[str, ...],
    data_dir: str,
    log_level: str,
    validator_keys: Optional[str],
    fee_recipient: str,
    graffiti: str,
    checkpoint_sync_url: Optional[str],
    bootnodes: tuple[str, ...],
    supernode: bool,
):
    """Run the consensus layer node."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    from .spec.constants import set_preset
    set_preset(preset)
    logger.info(f"Preset set to: {preset}")

    from .node import run_node

    all_peers = list(peers) + list(bootnodes)

    config = Config(
        engine_api_url=engine_api_url,
        jwt_secret_path=jwt_secret or "",
        validator_keys_spec=validator_keys or "",
        fee_recipient=fee_recipient,
        graffiti=graffiti,
        genesis_state_path=genesis_state,
        network_config_path=network_config or "",
        preset=preset,
        listen_port=p2p_port,
        listen_host=p2p_host,
        beacon_api_port=beacon_api_port,
        peers=all_peers,
        data_dir=data_dir,
        log_level=log_level,
        checkpoint_sync_url=checkpoint_sync_url or "",
        supernode=supernode,
    )

    logger.info("Starting consensoor")
    logger.info(f"  Preset: {preset}")
    logger.info(f"  Supernode: {supernode}")
    logger.info(f"  Engine API: {engine_api_url}")
    logger.info(f"  P2P: {p2p_host}:{p2p_port}")
    logger.info(f"  Beacon API: port {beacon_api_port}")
    if peers:
        logger.info(f"  Peers: {list(peers)}")
    if bootnodes:
        logger.info(f"  Bootnodes: {len(bootnodes)} configured")
        for bn in bootnodes:
            logger.info(f"    - {bn[:60]}..." if len(bn) > 60 else f"    - {bn}")
    logger.info(f"  Genesis state: {genesis_state}")
    if validator_keys:
        logger.info(f"  Validator keys: {validator_keys}")
    if checkpoint_sync_url:
        logger.info(f"  Checkpoint sync: {checkpoint_sync_url}")

    try:
        asyncio.run(run_node(config))
    except KeyboardInterrupt:
        logger.info("Shutting down")
        sys.exit(0)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
