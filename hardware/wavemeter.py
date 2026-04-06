import os
import random
import requests
from typing import Optional, Union, Dict, Any


class WavemeterAPI:
    def __init__(self, base_url: str, port: int, is_dummy: bool = False):
        """
        Initialize the API client with the base URL, port number,
        and an optional flag to use dummy data.
        """
        self.base_url = base_url
        self.port = port
        self.is_dummy = is_dummy

    def _get_url(self, endpoint: str) -> str:
        return f"{self.base_url}:{self.port}{endpoint}"

    def _fetch_data(self, url: str) -> Optional[Union[Dict[str, Any], str]]:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return (
                response.json()
                if response.headers.get("Content-Type") == "application/json"
                else response.text
            )
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def get_all_wavelengths(self) -> Optional[Dict[str, Any]]:
        if self.is_dummy:
            return {
                str(channel): f"{random.uniform(1500, 1600):.2f}"
                for channel in range(4)
            }
        else:
            url = self._get_url("/api/wave/")
            return self._fetch_data(url)

    def get_channel_wavelength(self, channel: int) -> Optional[str]:
        if self.is_dummy:
            return f"{random.uniform(1500, 1600):.2f}"
        else:
            url = self._get_url(f"/api/wave/{channel}/")
            return self._fetch_data(url)

    def get_pattern_data(self) -> Optional[str]:
        if self.is_dummy:
            pattern = [[random.randint(0, 255) for _ in range(2048)]]
            return pattern
        else:
            url = self._get_url("/api/pat/")
            return self._fetch_data(url)

    def get_frequency(self) -> Optional[str]:
        if self.is_dummy:
            return f"{random.uniform(1.9e14, 2.0e14):.2f}"
        else:
            url = self._get_url("/api/freq/")
            return self._fetch_data(url)

    def check_wlm_status(self) -> Optional[str]:
        if self.is_dummy:
            return 1
        else:
            url = self._get_url("/api/status/")
            return self._fetch_data(url)


if __name__ == "__main__":
    base_url = os.environ.get("WAVEMETER_URL", "http://localhost")
    port = int(os.environ.get("WAVEMETER_PORT", "8000"))

    api_client = WavemeterAPI(base_url, port, is_dummy=False)

    all_wavelengths = api_client.get_all_wavelengths()
    print("All Wavelengths:", all_wavelengths)

    pattern_data = api_client.get_pattern_data()
    print("Pattern Data:", pattern_data)

    channel = 0
    channel_wavelength = api_client.get_channel_wavelength(channel)
    print(f"Wavelength for Channel {channel}:", channel_wavelength)

    frequency = api_client.get_frequency()
    print("Frequency:", frequency)
