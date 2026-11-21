"""In-memory registry of loaded deployments with a hard cap."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from invoiceops_model_api.artifacts import ModelRef


class CapacityExceeded(Exception):
    pass


@dataclass(frozen=True)
class LoadedDeployment:
    deployment_id: str
    ref: ModelRef
    run_id: str
    slot: str  # key of the model inside the inference worker (unique per load)
    artifact_path: Path  # cached verified bytes the worker (re)loads from
    columns: tuple[str, ...]
    positive_index: int
    metrics: dict
    loaded_at: str

    def summary(self) -> dict:
        return {
            "deploymentId": self.deployment_id,
            "workspace": self.ref.workspace,
            "registeredModel": self.ref.registered_model,
            "version": self.ref.version,
            "runId": self.run_id,
            "loadedAt": self.loaded_at,
            "metrics": self.metrics,
        }


class DeploymentRegistry:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: dict[str, LoadedDeployment] = {}
        self._quarantined: dict[str, dict] = {}
        self._lock = Lock()

    def put(self, deployment: LoadedDeployment) -> LoadedDeployment | None:
        """Insert/replace atomically; returns the replaced deployment (to be unloaded)."""
        with self._lock:
            if deployment.deployment_id not in self._items and len(self._items) >= self.capacity:
                raise CapacityExceeded
            replaced = self._items.get(deployment.deployment_id)
            self._items[deployment.deployment_id] = deployment
            self._quarantined.pop(deployment.deployment_id, None)
            return replaced

    def quarantine(self, slot: str, reason: str) -> bool:
        """Remove the deployment holding ``slot`` and record why (it crashed its worker)."""
        with self._lock:
            for deployment_id, item in self._items.items():
                if item.slot == slot:
                    del self._items[deployment_id]
                    self._quarantined[deployment_id] = {
                        **item.summary(),
                        "reason": reason,
                        "quarantinedAt": now_iso(),
                    }
                    return True
        return False

    def quarantined(self) -> list[dict]:
        with self._lock:
            return [self._quarantined[k] for k in sorted(self._quarantined)]

    def has_room_for(self, deployment_id: str) -> bool:
        with self._lock:
            return deployment_id in self._items or len(self._items) < self.capacity

    def get(self, deployment_id: str) -> LoadedDeployment | None:
        with self._lock:
            return self._items.get(deployment_id)

    def remove(self, deployment_id: str) -> LoadedDeployment | None:
        with self._lock:
            self._quarantined.pop(deployment_id, None)
            return self._items.pop(deployment_id, None)

    def list(self) -> list[LoadedDeployment]:
        with self._lock:
            return sorted(self._items.values(), key=lambda d: d.deployment_id)


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
