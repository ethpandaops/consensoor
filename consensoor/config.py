"""Configuration for consensoor node."""

from dataclasses import dataclass, field


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
    data_dir: str = "./data"
    log_level: str = "INFO"
    checkpoint_sync_url: str = ""
    supernode: bool = False

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
        graffiti_bytes = self.graffiti.encode("utf-8")[:32]
        return graffiti_bytes + b"\x00" * (32 - len(graffiti_bytes))
