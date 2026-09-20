"""Runtime configuration helpers.

Hardware endpoints are intentionally supplied at runtime.  Keeping them out of
the source tree makes the repository safe to publish and prevents an omitted
configuration value from silently connecting to an unintended local service.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


def _value(explicit: Optional[str], *environment_names: str) -> Optional[str]:
    if explicit and explicit.strip():
        return explicit.strip()
    for name in environment_names:
        candidate = os.environ.get(name)
        if candidate and candidate.strip():
            return candidate.strip()
    return None


@dataclass(frozen=True)
class HardwareConfig:
    """Connection settings required for real-laser evaluation."""

    dlc_host: str
    wavemeter_url: str
    wavemeter_port: int
    oscilloscope_resource: str = "USB"

    @classmethod
    def from_values(
        cls,
        *,
        dlc_host: Optional[str] = None,
        wavemeter_url: Optional[str] = None,
        wavemeter_port: Optional[int] = None,
        oscilloscope_resource: Optional[str] = None,
    ) -> "HardwareConfig":
        resolved_host = _value(dlc_host, "DLC_HOST", "DLC_IP")
        resolved_url = _value(wavemeter_url, "WAVEMETER_URL")
        resolved_resource = (
            _value(oscilloscope_resource, "OSCILLOSCOPE_RESOURCE") or "USB"
        )

        if wavemeter_port is None:
            port_text = _value(None, "WAVEMETER_PORT")
            resolved_port = int(port_text) if port_text is not None else None
        else:
            resolved_port = wavemeter_port

        missing = []
        if resolved_host is None:
            missing.append("DLC_HOST (or --dlc-host)")
        if resolved_url is None:
            missing.append("WAVEMETER_URL (or --wavemeter-url)")
        if resolved_port is None:
            missing.append("WAVEMETER_PORT (or --wavemeter-port)")
        if missing:
            raise ConfigurationError(
                "Hardware mode requires explicit connection settings: "
                + ", ".join(missing)
            )

        parsed_url = urlparse(resolved_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise ConfigurationError(
                "WAVEMETER_URL must be an absolute http(s) URL, for example "
                "http://wavemeter-host"
            )
        if not 1 <= resolved_port <= 65535:
            raise ConfigurationError("WAVEMETER_PORT must be between 1 and 65535")

        return cls(
            dlc_host=resolved_host,
            wavemeter_url=resolved_url.rstrip("/"),
            wavemeter_port=resolved_port,
            oscilloscope_resource=resolved_resource,
        )

    @classmethod
    def from_namespace(cls, args) -> "HardwareConfig":
        """Build configuration from an argparse namespace and environment."""

        return cls.from_values(
            dlc_host=getattr(args, "dlc_host", None),
            wavemeter_url=getattr(args, "wavemeter_url", None),
            wavemeter_port=getattr(args, "wavemeter_port", None),
            oscilloscope_resource=getattr(args, "oscilloscope_resource", None),
        )
