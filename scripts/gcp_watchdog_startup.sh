#!/usr/bin/env bash
set -euo pipefail

metadata_url="http://metadata.google.internal/computeMetadata/v1/instance/attributes/watchdog-script"
sudo mkdir -p /opt/ubx-watchdog
curl -fsS -H "Metadata-Flavor: Google" "${metadata_url}" \
  | sudo tee /opt/ubx-watchdog/watchdog.py >/dev/null
sudo chmod 0755 /opt/ubx-watchdog/watchdog.py
exec sudo python3 -u /opt/ubx-watchdog/watchdog.py
