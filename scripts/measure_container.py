"""Memory of the service's child processes inside a LINUX container (where RLIMIT_AS is enforced).

Run it in the production image, which has no MLflow server or test tools, only the runtime:

    docker run --rm --entrypoint python invoiceops-model-api:measure \
        scripts/measure_container.py            # copy the script in with -v or `docker cp`

It trains representative models (logistic regression pipeline, HistGradientBoosting, RandomForest),
dumps them with skops and measures, from /proc:

* ``VmHWM`` (peak resident) and ``VmPeak`` (peak virtual, what ``RLIMIT_AS`` bounds) of the
  inference worker after loading N deployments of each kind;
* the same peaks for a one-shot check child scoring the official-size validation set;
* the smallest ``RLIMIT_AS`` (``MODEL_CHECK_MEMORY_MB``/``MODEL_WORKER_MEMORY_MB``) with which each
  model still passes a check, to size the limit against the container ``mem_limit``.

Numbers are printed as a table; nothing is written outside a temporary directory.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
from invoiceops_ml.contract import FEATURES, TARGET
from invoiceops_ml.synthetic import generate_invoices
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skops.io import dump

from invoiceops_model_api.inference import InferenceSupervisor
from invoiceops_model_api.isolation import CheckRunner
from invoiceops_model_api.settings import Settings

ROWS = int(os.environ.get("MEASURE_ROWS", "4000"))
DEPLOYMENTS = int(os.environ.get("MEASURE_DEPLOYMENTS", "10"))


def status_kb(pid: int, field: str) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def mb(kb: int) -> str:
    return f"{kb / 1024:7.0f}"


def children_of(pid: int) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                stat = (entry / "stat").read_text()
                if int(stat.rsplit(")", 1)[1].split()[1]) == pid:
                    found.append(int(entry.name))
            except (OSError, ValueError, IndexError):
                continue
    return found


class PeakWatcher(threading.Thread):
    """Polls the children of this process and keeps their peak VmHWM / VmPeak."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.hwm = 0
        self.vmpeak = 0
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            for pid in children_of(os.getpid()):
                self.hwm = max(self.hwm, status_kb(pid, "VmHWM"))
                self.vmpeak = max(self.vmpeak, status_kb(pid, "VmPeak"))
            time.sleep(0.02)


def settings(memory_mb: int = 3072) -> Settings:
    return Settings(
        service_token="x" * 40,
        tracking_uri="http://127.0.0.1:1",
        official_validation_seed=1,
        check_memory_mb=memory_mb,
        inference_memory_mb=memory_mb,
        check_timeout_seconds=120.0,
    )


def main() -> None:
    train = generate_invoices(n_invoices=ROWS, seed=1)
    x, y = train[list(FEATURES)], train[TARGET]
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "hgb-200": HistGradientBoostingClassifier(max_iter=200, random_state=0),
        "rf-200": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0),
    }
    matrix = x.to_numpy(dtype="float64")
    columns = list(FEATURES)
    print(f"python {sys.version.split()[0]}  rows={ROWS}  deployments per kind={DEPLOYMENTS}")
    with tempfile.TemporaryDirectory(prefix="invoiceops-measure-") as tmp:
        paths: dict[str, Path] = {}
        for name, model in models.items():
            model.fit(x, y)
            paths[name] = Path(tmp) / f"{name}.skops"
            dump(model, paths[name])
            print(f"{name:8s} file {paths[name].stat().st_size / 1024 / 1024:6.1f} MiB")

        print("\n-- one-shot check child (limit 3072 MB): peak MB while scoring the validation set")
        print(f"{'model':8s} {'peak RSS':>9s} {'peak virt':>10s} {'seconds':>8s}")
        runner = CheckRunner(settings())
        for name, path in paths.items():
            watcher = PeakWatcher()
            watcher.start()
            started = time.monotonic()
            outcome = runner.run(path, matrix, columns)
            elapsed = time.monotonic() - started
            watcher.stop.set()
            watcher.join()
            print(
                f"{name:8s} {mb(watcher.hwm)} {mb(watcher.vmpeak)}  {elapsed:7.1f}  ok={outcome.ok}"
            )

        print("\n-- smallest RLIMIT_AS (MB) at which the check still passes")
        for name, path in paths.items():
            passed = None
            for limit in (256, 384, 512, 640, 768, 1024, 1536, 2048, 3072):
                if CheckRunner(settings(limit)).run(path, matrix, columns).ok:
                    passed = limit
                    break
            print(f"{name:8s} {passed}")

        print(
            f"\n-- inference worker holding {DEPLOYMENTS} deployments of each kind (limit 3072 MB)"
        )
        print(f"{'step':24s} {'RSS':>8s} {'peak RSS':>9s} {'peak virt':>10s}")
        supervisor = InferenceSupervisor(settings(), cache_dir=Path(tmp) / "cache")
        try:
            supervisor.warm()
            pid = supervisor.worker_pid()
            print(
                f"{'started':24s} {mb(status_kb(pid, 'VmRSS'))} {mb(status_kb(pid, 'VmHWM'))} "
                f"{mb(status_kb(pid, 'VmPeak'))}"
            )
            for name, path in paths.items():
                for _ in range(DEPLOYMENTS):
                    slot, cached = supervisor.stage(path)
                    supervisor.activate(slot, cached, columns)
                pid = supervisor.worker_pid()
                label = f"+{DEPLOYMENTS} x {name}"
                print(
                    f"{label:24s} {mb(status_kb(pid, 'VmRSS'))} {mb(status_kb(pid, 'VmHWM'))} "
                    f"{mb(status_kb(pid, 'VmPeak'))}"
                )
            rows = np.asarray(matrix[:100], dtype=float).tolist()
            slot, cached = supervisor.stage(paths["rf-200"])
            supervisor.activate(slot, cached, columns)
            supervisor.predict(slot, rows)
            pid = supervisor.worker_pid()
            print(
                f"{'after a 100-row predict':24s} {mb(status_kb(pid, 'VmRSS'))} "
                f"{mb(status_kb(pid, 'VmHWM'))} {mb(status_kb(pid, 'VmPeak'))}"
            )
        finally:
            supervisor.shutdown()


if __name__ == "__main__":
    main()
