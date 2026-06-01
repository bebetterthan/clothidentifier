"""
mqtt_publisher.py — Reusable MQTT publisher for the ClothBot servo controller
paho-mqtt 2.x API (CallbackAPIVersion.VERSION2)

Usage
-----
from mqtt_publisher import ClothbotMQTT
from servo_map import SERVO_MAP

publisher = ClothbotMQTT(host="localhost", port=1883,
                          username="clothbot", password="secret")
publisher.connect()                       # called once at app startup
publisher.publish_command("baju_lengan_panjang", SERVO_MAP["baju_lengan_panjang"]["steps"])
publisher.disconnect()                    # called at app shutdown

The ESP32 subscriber should:
  - Subscribe to `clothbot/servo/command` with QoS 1
  - Parse JSON: {"label": str, "steps": [{"servos": [...], "delay_ms": int}, ...]}
  - Iterate steps in order, drive PCA9685, then delay
"""

import json
import logging
import os
import time
from typing import Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger("clothing-classifier.mqtt")

# ---------------------------------------------------------------------------
# Default topic (can be overridden via constructor / env var)
# ---------------------------------------------------------------------------
DEFAULT_TOPIC = "clothbot/servo/command"


class ClothbotMQTT:
    """
    Thread-safe MQTT publisher for servo-command messages.

    Wraps a paho-mqtt Client with:
    - Exponential-backoff automatic reconnection (1 s → 60 s)
    - Synchronous publish (blocks until broker ACKs or timeout)
    - Structured JSON payloads with a ``steps`` sequence

    Parameters
    ----------
    host : str
        Hostname or IP of the Mosquitto broker.
    port : int
        TCP port (default 1883).
    username : str
        MQTT username (Mosquitto auth).
    password : str
        MQTT password.
    topic : str
        MQTT topic to publish to.
    publish_timeout : float
        Seconds to wait for broker ACK before giving up (per publish).
    client_id : str, optional
        Explicit MQTT client ID; auto-generated when empty.
    """

    def __init__(
        self,
        host: str,
        port: int = 1883,
        username: str = "clothbot",
        password: str = "",
        topic: str = DEFAULT_TOPIC,
        publish_timeout: float = 5.0,
        client_id: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._topic = topic
        self._publish_timeout = publish_timeout

        # Build paho client with new v2 callback API
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id or f"clothbot-gce-{os.getpid()}",
            clean_session=True,
        )

        self._client.username_pw_set(username, password)

        # Exponential backoff: retry in 1 s, doubling up to 60 s
        self._client.reconnect_delay_set(min_delay=1, max_delay=60)

        # Register callbacks
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Connect to the broker and start the background network loop.

        A failed initial connection is logged as a warning (not a crash);
        paho's background thread will keep retrying automatically.
        """
        try:
            self._client.connect(self._host, self._port, keepalive=60)
            logger.info("[MQTT] Connecting to %s:%d …", self._host, self._port)
        except OSError as exc:
            logger.warning(
                "[MQTT] Initial connect to %s:%d failed: %s — background retry active",
                self._host, self._port, exc,
            )

        # loop_start() spawns a background daemon thread that handles
        # I/O, keepalive pings, and automatic reconnections.
        self._client.loop_start()

    def disconnect(self) -> None:
        """Gracefully stop the network loop and close the connection."""
        logger.info("[MQTT] Disconnecting …")
        self._client.loop_stop()
        self._client.disconnect()

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_command(
        self,
        label: str,
        steps: list[dict],
    ) -> bool:
        """
        Publish a servo-command JSON message to the configured topic.

        Blocks until the broker acknowledges the message (QoS 1) or
        ``publish_timeout`` seconds elapse.

        Parameters
        ----------
        label : str
            Clothing label string (e.g. ``"baju_lengan_panjang"``).
        steps : list[dict]
            List of step dicts from ``SERVO_MAP``, each containing
            ``servos`` (list of ch+angle dicts) and ``delay_ms``.

        Returns
        -------
        bool
            ``True`` if the broker ACKed within the timeout, ``False`` otherwise.
        """
        payload = json.dumps({"label": label, "steps": steps}, separators=(",", ":"))

        try:
            msg_info = self._client.publish(
                self._topic,
                payload=payload,
                qos=1,
                retain=False,
            )

            # Block until PUBACK received or timeout
            msg_info.wait_for_publish(timeout=self._publish_timeout)

            if msg_info.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(
                    "[MQTT] Publish returned error rc=%d for label '%s'",
                    msg_info.rc, label,
                )
                return False

            logger.info(
                "[MQTT] Published label='%s' mid=%d steps=%d bytes=%d",
                label, msg_info.mid, len(steps), len(payload),
            )
            return True

        except ValueError as exc:
            # wait_for_publish raises ValueError on timeout
            logger.warning(
                "[MQTT] Publish timed out (%.1fs) for label '%s': %s",
                self._publish_timeout, label, exc,
            )
            return False
        except RuntimeError as exc:
            logger.warning("[MQTT] Publish runtime error for label '%s': %s", label, exc)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("[MQTT] Unexpected publish error: %s", exc, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return True if the paho client reports it is currently connected."""
        return self._client.is_connected()

    def status(self) -> dict:
        """Return a dict suitable for inclusion in /health or /metrics."""
        return {
            "enabled":      True,
            "connected":    self.is_connected,
            "broker":       f"{self._host}:{self._port}",
            "topic":        self._topic,
        }

    # ------------------------------------------------------------------
    # Internal paho callbacks
    # ------------------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        connect_flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: Optional[mqtt.Properties],
    ) -> None:
        if reason_code.is_failure:
            logger.warning("[MQTT] Connection refused: %s", reason_code)
        else:
            logger.info("[MQTT] Connected (rc=%s) to %s:%d", reason_code, self._host, self._port)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: object,
        disconnect_flags: mqtt.DisconnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: Optional[mqtt.Properties],
    ) -> None:
        if reason_code.value != 0:
            logger.warning(
                "[MQTT] Unexpected disconnect rc=%s — reconnecting …", reason_code
            )
        else:
            logger.info("[MQTT] Disconnected cleanly.")


# ---------------------------------------------------------------------------
# Module-level factory — reads config from environment variables
# ---------------------------------------------------------------------------

def from_env() -> ClothbotMQTT:
    """
    Create a :class:`ClothbotMQTT` instance from environment variables.

    Expected env vars (all optional, defaults shown)::

        MQTT_HOST      = localhost
        MQTT_PORT      = 1883
        MQTT_USER      = clothbot
        MQTT_PASSWORD  = ""
        MQTT_TOPIC     = clothbot/servo/command

    Returns
    -------
    ClothbotMQTT
        A configured (but not yet connected) publisher instance.
    """
    return ClothbotMQTT(
        host=os.getenv("MQTT_HOST", "localhost"),
        port=int(os.getenv("MQTT_PORT", "1883")),
        username=os.getenv("MQTT_USER", "clothbot"),
        password=os.getenv("MQTT_PASSWORD", ""),
        topic=os.getenv("MQTT_TOPIC", DEFAULT_TOPIC),
    )
