"""Exceptions for beacon sync module."""


class BeaconAPIError(Exception):
    """Error from Beacon API."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"Beacon API error {status}: {message}")


class BlockNotFoundError(BeaconAPIError):
    """Block not found error."""

    def __init__(self, message: str):
        super().__init__(404, message)


class StateNotFoundError(BeaconAPIError):
    """State not found error."""

    def __init__(self, message: str):
        super().__init__(404, message)
