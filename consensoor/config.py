"""Configuration for consensoor node."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Node configuration."""

    peers: list[str] = field(default_factory=list)
    engine_api_url: str = "http://localhost:8551"
    jwt_secret_path: str = ""
    validator_keys_spec: str = ""
    fee_recipient: str = "0x" + "00" * 20
    graffiti: str = "consensoor"
    genesis_state_path: str = ""
    network_config_path: str = ""
    preset: str = "mainnet"
    listen_host: str = "0.0.0.0"
    listen_port: int = 9000
    beacon_api_port: int = 5052
    metrics_port: int = 8008
    data_dir: str = "./data"
    log_level: str = "INFO"
    checkpoint_sync_url: str = ""
    supernode: bool = False
    _el_client_info: Optional[dict] = field(default=None, repr=False)

    @property
    def jwt_secret(self) -> bytes:
        if not self.jwt_secret_path:
            return b""
        with open(self.jwt_secret_path, "rb") as f:
            return bytes.fromhex(f.read().decode().strip().replace("0x", ""))

    @property
    def fee_recipient_bytes(self) -> bytes:
        return bytes.fromhex(self.fee_recipient.replace("0x", ""))

    @property
    def graffiti_bytes(self) -> bytes:
        from .version import build_graffiti
        return build_graffiti(self.graffiti, self._el_client_info)

    def set_el_client_info(self, el_info: Optional[dict]) -> None:
        """Set EL client info for graffiti generation."""
        object.__setattr__(self, "_el_client_info", el_info)
