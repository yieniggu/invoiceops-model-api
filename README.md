# InvoiceOps Model API

Internal, multi-model scoring service (MLOPS-04). The InvoiceOps backend calls it with a
service credential; it is **not** exposed to students or the public internet.

- Loads models by **deployment**: `(workspace, registered model name, fixed integer version)` from MLflow, using a read-only service identity. Aliases are never resolved, so moving an alias in MLflow changes nothing until the app asks for another version.
- Only **skops** models, with an explicit type allowlist plus structural validation of tree arrays; student models only ever run in resource-limited child processes, never in the HTTP process. Pickle/cloudpickle and unknown types are rejected.
- Server-side **gate**: AUC on an official validation set that only exists inside this process.
- Features are computed here with `invoiceops_ml.contract` (`invoice-features-v2`); the backend sends raw invoice fields plus the supplier's prior invoices.

## Configuration (environment)

| Variable | Required | Meaning |
| --- | --- | --- |
| `MODEL_API_SERVICE_TOKEN` | yes, >= 32 chars | Bearer secret expected from the backend. The service refuses to start without it. |
| `OFFICIAL_VALIDATION_SEED` | yes, >= 128 bits | Secret of the validation set: >= 32 hex chars (`openssl rand -hex 32`) or >= 22 base64 chars. The integer seed is derived by hashing (SHA-256). Weak or short values (`1`, `42`, ...) make the service refuse to start. Never logged or returned. |
| `MLFLOW_TRACKING_URI` | yes | `http(s)` URL of the MLflow server **or gateway**. It is the ONLY host the service ever contacts for MLflow data (see Security model). |
| `MODEL_API_MLFLOW_AUTH` | no (`gateway` if `GATEWAY_SERVICE_READ_TOKEN` is set, else `mlflow`) | `gateway`: `Authorization: Bearer <GATEWAY_SERVICE_READ_TOKEN>` + `X-InvoiceOps-Workspace: <workspace>` (MLOPS-02 gateway, read-only service identity). `mlflow`: MLflow's native credentials from the environment + `X-MLFLOW-WORKSPACE` (direct server / local tests). |
| `GATEWAY_SERVICE_READ_TOKEN` | in `gateway` mode, >= 32 chars | Read-only service token of the gateway. |
| `MLFLOW_TRACKING_TOKEN` or `MLFLOW_TRACKING_USERNAME`+`MLFLOW_TRACKING_PASSWORD` | in `mlflow` mode, as your server needs | Read natively by the MLflow client code. Ignored in `gateway` mode. |
| `MODEL_CHECK_TIMEOUT_SECONDS` / `MODEL_CHECK_MEMORY_MB` / `MODEL_CHECK_CONCURRENCY` | no (60 / 1024 / 2) | Hard wall-clock timeout, address-space rlimit (`RLIMIT_AS`, enforced on Linux) and max concurrent one-shot check children. |
| `MODEL_CHECK_WAIT_SECONDS` | no (10) | How long a deploy check waits for a free check slot; then 503 `BUSY`. Checks and loads run on their own bounded thread pool (not the one that serves `/predict`); beyond its admission bound they get 503 `BUSY` at once. A full disk/tmpfs or a failed spawn is a 503 `CHECK_UNAVAILABLE` (fail closed). |
| `MODEL_API_AUDIT_HMAC_KEY` | no, >= 32 chars | Key of the audit digest in the logs. Default: derived from `MODEL_API_SERVICE_TOKEN` (`HMAC-SHA-256(token, "invoiceops-model-api/audit-key/v1")`), never a constant. |
| `MODEL_INFERENCE_TIMEOUT_SECONDS` / `MODEL_WORKER_MEMORY_MB` | no (10 / 1024) | Per-request timeout and address-space rlimit of the inference worker. |
| `GATE_MIN_AUC` / `GATE_MIN_LIFT` | no (0.70 / 0.02) | Gate thresholds. |
| `OFFICIAL_VALIDATION_ROWS` | no (4000) | Rows generated for the validation set. |
| `MAX_LOADED_DEPLOYMENTS` | no (30) | Hard cap; loads beyond it get 409. |
| `MAX_MODEL_BYTES` | no (64 MiB) | Hard cap on `model.skops` bytes, enforced while streaming the download (listing sizes are never trusted). |
| `MAX_BODY_BYTES` | no (4 MiB) | Largest request body of any route (declared or streamed): 413 `PAYLOAD_TOO_LARGE` beyond it. |
| `MAX_TOTAL_PRIORS` | no (20000) | Prior invoices accepted in ONE predict/batch request (each item is also capped at 5000): 422 `TOO_MANY_PRIORS`. |
| `MODEL_API_MLFLOW_MAX_RETRIES` / `MODEL_API_MLFLOW_TIMEOUT_SECONDS` / `MODEL_API_MLFLOW_BUDGET_SECONDS` | no (2 / 15 / 45) | Explicit bounds on every call to MLflow or the gateway: retries (0-2, exponential backoff 0.5 s), per-request timeout (1-30 s) and the total time all calls of one check may spend (the MLflow client defaults, 7 retries and 120 s, block a thread for minutes against a failing server). |
| `MAX_CONCURRENT_DOWNLOADS` | no (2) | Artifact downloads running at once; others wait up to 10 s and then get 503 `BUSY`. |

Run: `uv run uvicorn invoiceops_model_api.app:app --host 0.0.0.0 --port 8000`.

## API

All routes except `GET /health` need `Authorization: Bearer <MODEL_API_SERVICE_TOKEN>` (constant-time compare; otherwise 401). Errors are `{"detail": {"code", "message", ...}}`. Request validation errors never echo submitted values. Inputs are bounded: `amountPesos` is an integer in `[0, 10^13]`, names are 1-200 characters, a request body is at most `MAX_BODY_BYTES` and a request at most `MAX_TOTAL_PRIORS` prior invoices (422/413); an arithmetic overflow while computing features is a 422 `INVALID_INVOICE`, never a 500.

| Route | Purpose |
| --- | --- |
| `GET /health` | Liveness only, no auth: `{"status":"ok"}`. |
| `GET /metrics` | Request counters since start: `{status, processHardened, requests:{"<METHOD> <route template> <status>": count}}` (unknown HTTP methods are counted as `OTHER` and paths matching no route as `unmatched`, so the key space is bounded; in memory; the backend or a scraper may poll it with the service token). |
| `GET /deployments` | Process health + loaded deployments: `{status, featureSchemaVersion, capacity, loaded, deployments:[{deploymentId, workspace, registeredModel, version, runId, loadedAt, metrics}], quarantined:[{deploymentId, ..., reason, quarantinedAt}]}`. |
| `POST /deploy-check` | Body `{workspace, registeredModel, version}`. Downloads, validates, runs the gate; does **not** activate. 200 `{passed, reasons:[{code,message,detail}], metrics:{auc,dummyAuc,lift,minAuc,minLift,validationRows}\|null, featureSchemaVersion, model:{workspace,registeredModel,version,runId}}`. 502 `MLFLOW_UNAVAILABLE` (MLflow or the gateway failed within the retry/timeout/budget bounds above), 503 `GATE_UNAVAILABLE`, 503 `BUSY` (download or check slots taken), 503 `CHECK_UNAVAILABLE` (storage full or spawn failed). |
| `PUT /deployments/{deploymentId}` | Same body. Re-runs the full check and loads only if it passes (422 `GATE_FAILED` with the same report). Idempotent for the same reference; a different reference replaces atomically. 409 `CAPACITY_EXCEEDED` past the cap. `deploymentId` matches `[A-Za-z0-9_-]{1,64}`. |
| `DELETE /deployments/{deploymentId}` | Unload; `{deploymentId, unloaded}` (idempotent). |
| `POST /deployments/{deploymentId}/predict` | Body `{invoice:{supplierName, amountPesos, issuedOn, dueOn, createdAt, businessStatus?}, priorInvoices:[{supplierName, amountPesos, createdAt, businessStatus}]}`. A prior carrying `decidedAt` is rejected with 422 `INVALID_REQUEST` (issue type `decided_at_not_accepted`, message «/predict acepta solo estados vigentes»): send each prior with its status as of the invoice's `createdAt`. Returns `{deploymentId, workspace, registeredModel, modelVersion, runId, featureSchemaVersion, probability}`. 404 `DEPLOYMENT_NOT_LOADED`; 503 `INFERENCE_UNAVAILABLE` (worker crashed/timed out/busy, retry; `quarantined: true` when this crash got the deployment quarantined) or `INFERENCE_FAILED`. |
| `POST /deployments/{deploymentId}/predict-batch` | Body `{items:[<predict body>...]}` (<= 100). Returns the same metadata with `probabilities:[...]`. |

Reason codes (the app translates them): `INVALID_REFERENCE`, `MODEL_VERSION_NOT_FOUND`, `ARTIFACT_SOURCE_NOT_ALLOWED`, `ARTIFACT_LAYOUT_INVALID`, `ARTIFACT_TOO_LARGE`, `PICKLE_FORMAT_NOT_ALLOWED`, `UNSUPPORTED_FLAVOR`, `UNTRUSTED_TYPE` (detail.types), `SKOPS_ARCHIVE_INVALID`, `MODEL_LOAD_FAILED`, `MODEL_STRUCTURE_INVALID`, `MODEL_CHECK_CRASHED`, `MODEL_CHECK_TIMEOUT`, `SIGNATURE_MISSING`, `SIGNATURE_MISMATCH` (detail.expected/found), `NO_PREDICT_PROBA`, `INVALID_CLASSES`, `INFERENCE_FAILED`, `AUC_BELOW_MIN` (detail.auc, minAuc), `AUC_LIFT_BELOW_MIN` (detail.auc, dummyAuc, lift, minLift).

## Gate

`generate_invoices(seed=derived from OFFICIAL_VALIDATION_SEED)` is computed in-process (the parent), cached, and never exposed or logged; only its *features* are handed to the check child, which returns scores; the parent computes AUC with the labels. Pass iff `AUC >= GATE_MIN_AUC` **and** `AUC >= dummy_auc + GATE_MIN_LIFT` (the comparison uses unrounded values); `dummy_auc` is from a `DummyClassifier(strategy="prior")` on the same set (0.5). Metrics logged by the student in MLflow are never read. Without the seed the service does not start; if a test builds the app without it, deploy-check/load answer 503 `GATE_UNAVAILABLE`.

Validation-oracle mitigation: the seed has >= 128 bits (it cannot be enumerated from metrics) and the returned `auc`, `dummyAuc` and `lift` are rounded to 2 decimals. **MLOPS-05 must still rate-limit deploy-checks per group**: every check is an evaluation on the hidden set, so unlimited checks leak information regardless of rounding.

## Security model

Threat model: students fully control their registered model versions: the version `source` URI and the bytes of every artifact (including the `.npy` arrays inside the skops archive).

**1. Only the tracking server is ever contacted (no SSRF, no credential exfiltration).** `artifacts.py` builds every URL from `MLFLOW_TRACKING_URI`; MLflow's client and `requests` never resolve a host from a version's source. The source (as resolved by the server's `get-download-uri`) is only parsed: `mlflow-artifacts:` with an EMPTY netloc (or the tracking host's own), a normalized path without `..`/`.`/empty/percent/backslash segments, no query/fragment, and, if it names `workspaces/<x>/`, `<x>` must be the deployment's workspace. `runs:/<run id>/<path>` (what the gateway allows) is resolved through the tracking server's own run record in the deployment's workspace; `models:/m-<id>` (MLflow 3 logged models, what `log_model` + register produces) arrives already resolved by `get-download-uri` to the workspace's `mlflow-artifacts:/workspaces/<ws>/...`. Everything else (`http(s)`, `s3`, `file`, foreign hosts, userinfo, unresolved `models:/name/version`, malformed URLs) is `ARTIFACT_SOURCE_NOT_ALLOWED` before any artifact request; there is no setting to re-enable other schemes/hosts. Credentials (`Authorization`) and the workspace header go only to the tracking URI; redirects are not followed. Downloads (`MLmodel` <= 1 MiB, `model.skops` <= `MAX_MODEL_BYTES`) stream with a hard byte cap enforced while reading (also on decompressed bytes), into a private temp dir.
**2. The HTTP process never touches a student model.** skops/sklearn models are loaded and called ONLY in child processes: a one-shot **check child** (deploy check) and one supervised **inference worker** (serving). Children get a minimal environment (no service token, seed or MLflow credentials: `child_env` refuses secret-looking variables; what they need travels on the inherited socket), no stdio, their own session, an `RLIMIT_AS` memory limit (Linux/container; macOS ignores it) and no core dumps, and talk over one inherited socket with bounded JSON messages. `tests/test_inference_worker.py` asserts the HTTP modules never import the loader.
The HTTP process itself calls `prctl(PR_SET_DUMPABLE, 0)` at start-up (Linux) and, in the container, IS PID 1 (no docker-init), so a child running as the same user cannot read its secrets from `/proc/1/environ`, `/proc/<pid>/environ` or `/proc/<pid>/mem` or ptrace it (`tests/test_hardening.py` proves the mechanism and `tests/test_container_pid1.py` the container, both in CI). Whatever a child reports back is reduced by the parent to a closed set of codes with FIXED messages and a fixed detail schema (`isolation.sanitize_failure`): a compromised child cannot turn the API response into a channel for the validation features, and the returned metrics stay rounded to 2 decimals. The only free-form detail left is the list of disallowed type names (at most 10, each a dotted identifier of at most 120 characters). Deploy checks are rate-limited by the backend (not here): it allows a bounded number of gate runs per scope per hour (10 by default, `RATE_LIMITED` with `Retry-After`) and reloads after a Model API restart do not consume that budget; this service only bounds concurrency (`MODEL_CHECK_CONCURRENCY`, `MAX_CONCURRENT_DOWNLOADS`).
**3. Layered model validation.** (a) `skops_policy.py`: the archive's schema is read before any import, every node type must be on the allowlist, only known node loaders are accepted, `skops.io.load` gets an explicit `trusted` subset (never `True`), the zip may only contain `.npy` and `schema.json` and must expand to <= 256 MiB. (b) `structure.py`: after loading and before any predict, every `sklearn.tree._tree.Tree` (also inside forests, gradient boosting, calibrated wrappers; the object graph is walked) must have consistent `node_count`, children either `-1` (leaf, both) or strictly greater than the parent and `< node_count` (acyclic, in range), split features in `[0, n_features)`, a value array of the right shape, and `n_features` equal to the owning estimator's `n_features_in_`; HistGradientBoosting `TreePredictor.nodes` get the same child/feature/`is_leaf` checks, and categorical splits are refused. Violations are `MODEL_STRUCTURE_INVALID`. sklearn's Cython code indexes these arrays without bounds checks, so a crafted array would otherwise crash the process (SIGBUS) or loop forever. (c) Process isolation as above is the backstop if (b) ever misses something: the check child has a hard timeout and rlimit (`MODEL_CHECK_CRASHED` / `MODEL_CHECK_TIMEOUT`, API stays up) and the inference worker is supervised.
**4. Inference worker supervision (`inference.py`).** One worker holds all active deployments (a pool would multiply the ~300 MB interpreter+sklearn cost; predictions are microseconds, so requests are serialized with a bounded wait). Every request has a timeout. If the worker crashes, hangs or runs out of memory it is killed, the request gets 503 `INFERENCE_UNAVAILABLE`, and the next request starts a fresh worker that reloads the needed deployments from the cached, already-verified artifact bytes (lazily, per deployment, so one bad model cannot block the others after a restart). A deployment that was in flight when the worker died twice (cumulative) is quarantined: unloaded, cached bytes deleted, reported under `quarantined` in `GET /deployments`; a new explicit deploy clears it.
**5. Format rules.** `MLmodel` must have `sklearn.serialization_format == skops`, `pickled_model == model.skops` and no other flavor besides the matching `python_function`; any `.pkl/.pickle/.cloudpickle/.joblib` in the artifact directory is rejected.
**6. No data in logs.** Feature values and invoice data are never logged (only deployment id, model name, version, reason codes). Mutating requests (`PUT`/`DELETE`) log one audit line `audit principal=<16 hex> method route status`: the principal is a keyed, truncated HMAC of the service credential (`principal_hash`), not the secret, and `anonymous` for refused requests. Counters by route and status are served by `GET /metrics`. The service logger writes INFO lines to **stdout** (configured in `create_app`, so the real container CMD emits them) as `ts=<UTC> level=INFO logger=invoiceops_model_api msg="<key=value ...>"`: `audit ...`, `deploy_check ...`, `deployment_loaded/unloaded ...`; read them with `docker logs`. If the process cannot be made non-dumpable on Linux (`prctl(PR_SET_DUMPABLE)` failed) it logs `process_hardening_failed` at ERROR and `GET /metrics` reports `"processHardened": false`.

Known limits: `RLIMIT_AS` is not enforced on macOS (development). A deploy check costs ~3-5 s (a fresh interpreter imports sklearn/skops per check); this is also why concurrent checks are capped (`MODEL_CHECK_CONCURRENCY`).

## State: the registry lives in memory only

The set of loaded deployments (and the quarantine list) exists only in this process. After a restart or a crash of the container `GET /deployments` is empty and `POST /deployments/{id}/predict` answers 404 `DEPLOYMENT_NOT_LOADED` until somebody sends `PUT /deployments/{id}` again; nothing is persisted on purpose (the cached model bytes live in a private temp directory that is removed). The InvoiceOps backend is the source of truth: it re-sends the `PUT` (re-running the gate) when a prediction meets `DEPLOYMENT_NOT_LOADED` or when scoring is turned on (`ensureLoaded`, single flight per deployment, up to 120 s). Its periodic reconcile does **not** re-load lost deployments eagerly: it only sweeps stale checks, records quarantines and unloads orphans. So after a restart the first prediction of each deployment pays a full gate run (a few seconds); an eager re-PUT from the backend's reconcile would remove that wait.

## MLflow workspace selection (MLflow 3.16.1)

The client API has no `workspace` parameter, but the server selects the workspace from a request header: `X-MLFLOW-WORKSPACE` on a direct MLflow server (`mlflow/utils/workspace_utils.py`), `X-InvoiceOps-Workspace` on the MLOPS-02 gateway (which overwrites the MLflow header itself and only lets the read-only service token read). The service sends the right header explicitly on every call (version metadata, run lookup, artifact listing/download), so it touches **no** process-global MLflow state (`WorkspaceContext`/`MLFLOW_WORKSPACE`) and needs no lock: concurrent checks never serialize on MLflow, and nothing is held during load/predict/gate. The server/gateway decides what the service identity may read in that workspace; the app must pass the right workspace per deployment. Against a server without workspaces enabled, the workspace header is rejected.

## Container

The image needs `invoiceops-ml` (path source `../invoiceops-ml` in `pyproject.toml`/`uv.lock`):

```
docker build --build-context ml=../invoiceops-ml -t invoiceops-model-api .
```

Only `pyproject.toml`, `README.md` and `src/` of that directory are copied (to `/invoiceops-ml`, which is where `../invoiceops-ml` resolves from `/app`). In a pipeline use a clean checkout (or git URL) as the `ml` context. Secrets are runtime env only. Bind the service to the internal network only.

## Memory (measured)

Two measurements. The first (macOS, `scripts/measure_memory.py`, through the real API with a local MLflow) gave the shape; the one that counts is the second, taken **inside a Linux container of the production image** (python 3.12.16, sklearn 1.9.1, 4000-row models, `docker run --entrypoint python ... scripts/measure_container.py`; `RLIMIT_AS` is enforced there). Peaks come from `/proc/<pid>/status`: `VmHWM` (peak resident) and `VmPeak` (peak virtual, what `RLIMIT_AS` bounds).

| State (Linux container) | RSS | peak virtual |
| --- | --- | --- |
| HTTP process at rest (`uvicorn`, app imported, read-only filesystem) | ~130 MB | - |
| One-shot check child scoring the 4000-row validation set: logistic regression / HistGB (200 iter) / RandomForest (200 trees, 9.9 MiB file) | 201 / 202 / 227 MB peak, 7-8 s each | ~1.5 GB with a 3 GiB limit |
| Inference worker, started with no model | 1 MB | 9 MB |
| Worker holding 10 logreg + 10 HistGB + 10 RF deployments (30) | 300 MB | 544 MB |
| ... after a 100-row predict (OpenBLAS/OpenMP threads map their stacks) | 310 MB | 1575 MB with a 3 GiB limit |
| Smallest `RLIMIT_AS` at which a check passes (all three kinds) / the worker loads + predicts (RF) | 512 MB / <= 512 MB | |

Activating a model while others are loaded briefly runs a **candidate worker** next to the serving one (blue/green, see `inference.py`), so a worker-sized (~310 MB) peak is added during an activation.

Alignment of the limits with the container:

* `mem_limit: 2g` (cgroup) = HTTP ~300 + serving worker ~310 + candidate worker ~310 + 2 check children x ~230 = ~1.4 GB typical, with headroom for models near the 64 MiB file cap (an unconstrained forest can need several times its file size).
* `MODEL_CHECK_MEMORY_MB` and `MODEL_WORKER_MEMORY_MB` default to and are set in `compose.production.yml` to **1024** (`RLIMIT_AS`): twice the 512 MB at which every measured model still works and half of `mem_limit`, so one runaway child hits its own limit (`MODEL_CHECK_CRASHED` / worker restart) long before the cgroup would OOM-kill the HTTP process. The previous 3072 was larger than `mem_limit` and bounded nothing. Raise both together with `mem_limit` if you expect larger models; `MAX_LOADED_DEPLOYMENTS` and `MODEL_CHECK_CONCURRENCY` bound the sum.
* `pids_limit: 256` (threads count): the request thread pool (40), the event loop, up to 4 child processes and their BLAS threads.
* `read_only: true` with `/tmp` as a 768 MB **tmpfs**: the verified model cache and the downloads are the only writes (`HOME=/tmp`). Because a tmpfs lives in RAM it counts towards `mem_limit`; a full one makes a deploy fail closed. Typical models are 0.03-10 MB (30 deployments ~ 150 MB); the hard worst case (30 x 64 MiB) does not fit and is not supported without raising `mem_limit` and the tmpfs.
* `cap_drop: [ALL]`, `no-new-privileges`. **No `init: true`**: docker-init would be PID 1 carrying the full environment (service token, validation seed, gateway token), readable through `/proc/1/environ` by any child of the same user. uvicorn is PID 1 instead: it calls `prctl(PR_SET_DUMPABLE, 0)` at start-up and runs a bounded zombie reaper thread (`procutil.start_zombie_reaper`, every 2 s, only when it is PID 1) that waits for the orphans of killed children but never for the children it tracks itself. `tests/test_container_pid1.py` runs the real image with these flags in a Linux container (CI job `Container image`) and proves that a child cannot read `/proc/1/environ` or `/proc/<ppid>/environ`; `tests/test_container_config.py` fails if `init` is enabled while secrets are in the environment.

`INVOICEOPS_ML_REF` (CI repository variable): the CI guard accepts a full 40-character commit SHA or a `vX.Y.Z` tag, never a branch. **Prefer the SHA**: a tag is mutable (it can be deleted and re-created on another commit), a SHA is not. Tags stay allowed.

The build fails when the version of the `ml` context differs from the one pinned in `uv.lock` (`scripts/check_ml_version.py`, run after the last `uv sync`; `uv sync --frozen` alone would accept any checkout of that path source). `.python-version` pins the interpreter (3.12) for local runs.

Reproduce: `docker build --build-context ml=../invoiceops-ml -t invoiceops-model-api:measure .` and `docker run --rm --entrypoint python -v "$PWD/scripts/measure_container.py:/m/measure_container.py:ro" invoiceops-model-api:measure -u /m/measure_container.py`.

## Development

```
uv sync
uv run ruff format . && uv run ruff check .
uv run pytest -q
```

Tests start a real `mlflow server --enable-workspaces` on a free localhost port (sqlite backend, artifact proxy), log real skops models to it, and use local fake servers for SSRF/streaming/credential checks. Fault-injection tests use `tests/helpers/` entry points (a check child without structural validation, a worker that crashes/hangs on demand); production entry points always validate. The suite takes a few minutes because every check/worker spawns a fresh interpreter.
