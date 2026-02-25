from pythonosc import udp_client
from osc.mapping import OSC_ADDRESSES, USE_STRING_VALUES, VALUE_CODES


class OSCSender:
    """
    Sends OSC messages to Unreal Engine (or any OSC receiver).

    Each state change maps to one OSC message:
        address: e.g. /person/proximity
        value:   e.g. "close" (string) or 1 (int, see mapping.py)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7000):
        self.host = host
        self.port = port
        self._client = udp_client.SimpleUDPClient(host, port)
        print(f"[OSCSender] Ready — sending to {host}:{port}")

    def send_change(self, field: str, value: str) -> None:
        """
        Send an OSC message for a state change.

        Args:
            field: one of "proximity", "expression", "gesture"
            value: string value of the new state (e.g. "close", "smile")
        """
        address = OSC_ADDRESSES.get(field)
        if address is None:
            print(f"[OSCSender] Warning: no OSC address defined for field '{field}'")
            return

        # floats are always sent as-is regardless of USE_STRING_VALUES
        if isinstance(value, float):
            osc_value = value
        else:
            osc_value = value if USE_STRING_VALUES else VALUE_CODES.get(value, 0)
        self._client.send_message(address, osc_value)
        print(f"[OSCSender] Sent: {address} → {osc_value}")

    def send_all(self, state) -> None:
        """
        Broadcast all current state values (useful on startup to sync UE state).
        """
        self.send_change("proximity",  state.proximity.value)
        self.send_change("expression", state.expression.value)
        self.send_change("gesture",    state.gesture.value)
