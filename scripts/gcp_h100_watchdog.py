#!/usr/bin/env python3
"""Keep one capped UB-X H100 Flex-start request alive for a bounded period."""

from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


METADATA = "http://metadata.google.internal/computeMetadata/v1"
METADATA_HEADERS = {"Metadata-Flavor": "Google"}
COMPUTE = "https://compute.googleapis.com/compute/v1"
STORAGE = "https://storage.googleapis.com/storage/v1"
STATE_PATH = Path("/var/lib/ubx-h100-watchdog.json")


def metadata(path: str) -> str:
    request = urllib.request.Request(f"{METADATA}/{path}", headers=METADATA_HEADERS)
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read().decode().strip()


def access_token() -> str:
    payload = json.loads(metadata("instance/service-accounts/default/token"))
    return str(payload["access_token"])


def api(method: str, url: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {access_token()}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {error.code}: {detail}") from error


def parse_timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def list_trainers(project: str, zones: list[str]) -> list[dict]:
    trainers: list[dict] = []
    name_filter = urllib.parse.quote("name eq ubx-fara-h100-.*")
    for zone in zones:
        result = api(
            "GET",
            f"{COMPUTE}/projects/{project}/zones/{zone}/instances?filter={name_filter}",
        )
        for item in result.get("items", []):
            item["_zone"] = zone
            trainers.append(item)
    return trainers


def delete_trainer(project: str, trainer: dict) -> None:
    api(
        "DELETE",
        f"{COMPUTE}/projects/{project}/zones/{trainer['_zone']}/instances/{trainer['name']}",
    )


def training_complete(bucket: str, output_prefix: str, target_step: int) -> bool:
    object_name = f"{output_prefix}/fara_adapter/latest.json"
    encoded = urllib.parse.quote(object_name, safe="")
    try:
        payload = api("GET", f"{STORAGE}/b/{bucket}/o/{encoded}?alt=media")
    except RuntimeError as error:
        if "HTTP 404" in str(error):
            return False
        raise
    return int(payload.get("step", 0)) >= target_step


def create_trainer(
    project: str,
    zone: str,
    trainer_sa: str,
    bucket: str,
    output_prefix: str,
    attempt: int,
) -> str:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M")
    name = f"ubx-fara-h100-auto-{now}-{attempt % 100:02d}"
    source_image = (
        "projects/deeplearning-platform-release/global/images/family/"
        "common-cu129-ubuntu-2404-nvidia-580"
    )
    payload = {
        "name": name,
        "machineType": f"zones/{zone}/machineTypes/a3-highgpu-1g",
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "type": "PERSISTENT",
                "initializeParams": {
                    "sourceImage": source_image,
                    "diskSizeGb": "300",
                    "diskType": f"zones/{zone}/diskTypes/hyperdisk-balanced",
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": "global/networks/default",
                "accessConfigs": [{"name": "External NAT", "type": "ONE_TO_ONE_NAT"}],
            }
        ],
        "serviceAccounts": [
            {
                "email": trainer_sa,
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            }
        ],
        "metadata": {
            "items": [
                {
                    "key": "startup-script-url",
                    "value": f"gs://{bucket}/bootstrap/gcp_overnight_fara.sh",
                },
                {
                    "key": "gcs-output",
                    "value": f"gs://{bucket}/{output_prefix}",
                },
                {"key": "train-steps", "value": "1200"},
                {"key": "max-samples", "value": "12000"},
                {"key": "deadline-seconds", "value": "27000"},
            ]
        },
        "labels": {"ubx-role": "fara-trainer", "managed-by": "ubx-watchdog"},
        "scheduling": {
            "provisioningModel": "FLEX_START",
            "instanceTerminationAction": "DELETE",
            "maxRunDuration": {"seconds": "28800"},
            "onHostMaintenance": "TERMINATE",
            "automaticRestart": False,
        },
        "params": {"requestValidForDuration": {"seconds": "7200"}},
        "reservationAffinity": {"consumeReservationType": "NO_RESERVATION"},
    }
    api(
        "POST",
        f"{COMPUTE}/projects/{project}/zones/{zone}/instances",
        payload,
    )
    return name


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"attempt": 0}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


def main() -> int:
    project = metadata("project/project-id")
    trainer_sa = metadata("instance/attributes/trainer-service-account")
    bucket = metadata("instance/attributes/output-bucket")
    output_prefix = metadata("instance/attributes/output-prefix")
    watch_seconds = int(metadata("instance/attributes/watch-seconds"))
    poll_seconds = int(metadata("instance/attributes/poll-seconds"))
    zones = metadata("instance/attributes/h100-zones").split(";")
    deadline = time.monotonic() + watch_seconds
    state = load_state()

    print(
        json.dumps(
            {
                "event": "watchdog_started",
                "project": project,
                "zones": zones,
                "watch_seconds": watch_seconds,
            }
        ),
        flush=True,
    )

    while time.monotonic() < deadline:
        try:
            if training_complete(bucket, output_prefix, 1200):
                print('{"event":"training_complete"}', flush=True)
                return 0

            trainers = list_trainers(project, zones)
            active = [
                item
                for item in trainers
                if item.get("status")
                in {"PENDING", "PROVISIONING", "STAGING", "RUNNING"}
            ]
            now = dt.datetime.now(dt.timezone.utc)
            for trainer in list(active):
                if trainer.get("status") != "PENDING":
                    continue
                age = (now - parse_timestamp(trainer["creationTimestamp"])).total_seconds()
                if age > 7_500:
                    print(
                        json.dumps(
                            {
                                "event": "delete_stale_request",
                                "name": trainer["name"],
                                "zone": trainer["_zone"],
                                "age_seconds": round(age),
                            }
                        ),
                        flush=True,
                    )
                    delete_trainer(project, trainer)
                    active.remove(trainer)

            if active:
                print(
                    json.dumps(
                        {
                            "event": "trainer_active",
                            "trainers": [
                                {
                                    "name": item["name"],
                                    "zone": item["_zone"],
                                    "status": item["status"],
                                }
                                for item in active
                            ],
                        }
                    ),
                    flush=True,
                )
            else:
                attempt = int(state["attempt"])
                zone = zones[attempt % len(zones)]
                name = create_trainer(
                    project,
                    zone,
                    trainer_sa,
                    bucket,
                    output_prefix,
                    attempt,
                )
                state["attempt"] = attempt + 1
                save_state(state)
                print(
                    json.dumps(
                        {
                            "event": "trainer_requested",
                            "name": name,
                            "zone": zone,
                            "attempt": attempt + 1,
                        }
                    ),
                    flush=True,
                )
        except Exception as error:  # Keep the bounded watchdog alive on transient API errors.
            print(json.dumps({"event": "watchdog_error", "error": str(error)}), flush=True)
        time.sleep(poll_seconds)

    print('{"event":"watchdog_deadline_reached"}', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
