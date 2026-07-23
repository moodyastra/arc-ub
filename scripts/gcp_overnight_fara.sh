#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/moodyastra/arc-ub.git}"
WORK_ROOT="${WORK_ROOT:-/opt/arc-ub-overnight}"

metadata_value() {
  curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true
}

GCS_OUTPUT="${GCS_OUTPUT:-$(metadata_value gcs-output)}"
TRAIN_STEPS="${TRAIN_STEPS:-$(metadata_value train-steps)}"
MAX_SAMPLES="${MAX_SAMPLES:-$(metadata_value max-samples)}"
DEADLINE_SECONDS="${DEADLINE_SECONDS:-$(metadata_value deadline-seconds)}"
TRAIN_STEPS="${TRAIN_STEPS:-1200}"
MAX_SAMPLES="${MAX_SAMPLES:-12000}"
DEADLINE_SECONDS="${DEADLINE_SECONDS:-27000}"
SYNC_PID=""
mkdir -p "${WORK_ROOT}/outputs/logs"
LOG_PATH="${WORK_ROOT}/outputs/logs/$(hostname)-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_PATH}") 2>&1

sync_outputs() {
  if [[ -n "${GCS_OUTPUT}" && -d "${WORK_ROOT}/outputs" ]]; then
    gcloud storage rsync --recursive "${WORK_ROOT}/outputs" "${GCS_OUTPUT}" || true
  fi
}

finish() {
  exit_status="$?"
  if [[ -n "${SYNC_PID}" ]]; then
    kill "${SYNC_PID}" 2>/dev/null || true
  fi
  echo "UB-X Fara startup exiting with status ${exit_status}"
  sync_outputs
  sudo shutdown -h now || true
}
trap finish EXIT

sudo apt-get update
sudo apt-get install -y build-essential git python3-dev python3-venv
if [[ ! -d "${WORK_ROOT}/repo/.git" ]]; then
  sudo mkdir -p "${WORK_ROOT}"
  sudo chown -R "$(id -u):$(id -g)" "${WORK_ROOT}"
  git clone "${REPO_URL}" "${WORK_ROOT}/repo"
else
  git -C "${WORK_ROOT}/repo" pull --ff-only
fi

python3 -m venv "${WORK_ROOT}/venv"
source "${WORK_ROOT}/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${WORK_ROOT}/repo/requirements-fara-train.txt"

mkdir -p "${WORK_ROOT}/outputs/data" "${WORK_ROOT}/outputs/fara_adapter"
if [[ -n "${GCS_OUTPUT}" ]]; then
  gcloud storage rsync --recursive "${GCS_OUTPUT}" "${WORK_ROOT}/outputs" || true
  (
    while true; do
      sleep 600
      sync_outputs
    done
  ) &
  SYNC_PID="$!"
fi

cd "${WORK_ROOT}/repo"
if [[ ! -s "${WORK_ROOT}/outputs/data/fara_train.jsonl" ]]; then
  python -m ubx.fara_arc_data \
    --output-dir "${WORK_ROOT}/outputs/data" \
    --episodes 2500 \
    --max-samples "${MAX_SAMPLES}"
  sync_outputs
fi

timeout "${DEADLINE_SECONDS}" python -m ubx.fara_lora \
  --manifest "${WORK_ROOT}/outputs/data/fara_train.jsonl" \
  --output-dir "${WORK_ROOT}/outputs/fara_adapter" \
  --steps "${TRAIN_STEPS}" \
  --checkpoint-every 100
