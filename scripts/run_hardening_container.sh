#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${OMNI_TORCH_DOCKER_IMAGE:-omni-torch-validation:latest}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for container hardening runs."
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "docker image not found: $IMAGE"
  echo "Set OMNI_TORCH_DOCKER_IMAGE to an existing validation image."
  exit 1
fi

if [ -n "${OMNI_TORCH_DOCKER_CPUS:-}" ]; then
  CPU_LIMIT="$OMNI_TORCH_DOCKER_CPUS"
else
  HOST_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  CPU_LIMIT="$(awk "BEGIN { v = $HOST_CPUS * 0.30; if (v < 0.10) v = 0.10; printf \"%.2f\", v }")"
fi

if [ -n "${OMNI_TORCH_DOCKER_MEMORY:-}" ]; then
  MEM_LIMIT="$OMNI_TORCH_DOCKER_MEMORY"
else
  TOTAL_KB="$(awk '/MemTotal:/ { print $2 }' /proc/meminfo)"
  MEM_LIMIT="$(awk "BEGIN { m = int(($TOTAL_KB * 0.30) / 1024); if (m < 256) m = 256; printf \"%dm\", m }")"
fi

exec docker run --rm \
  --cpus "$CPU_LIMIT" \
  --memory "$MEM_LIMIT" \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  "$IMAGE" \
  bash -lc "make test-hardening-host"
