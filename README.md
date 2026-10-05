# InvoiceOps Model API

Servicio HTTP mínimo para exponer modelos de InvoiceOps. Usa un catálogo local
explícito y carga sus modelos desde MLflow durante el inicio del proceso.

## Inicio rápido

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
export MLFLOW_TRACKING_URI="https://mlflow.example.internal"
export MODEL_API_ADMIN_TOKEN="..."
uv run uvicorn invoiceops_model_api.app:app --host 0.0.0.0 --port 8000
```

En otra terminal, verifica el servicio:

```bash
curl http://127.0.0.1:8000/health
```

La respuesta esperada es:

```json
{"status":"ok"}
```

## Configuración de MLflow

`MLFLOW_TRACKING_URI` es obligatoria y no puede ser vacía o contener sólo
espacios. Si falta o es inválida, el arranque falla cerrado: el proceso no queda
operacional. El servicio sólo lee la configuración desde variables de entorno
en el arranque; no acepta credenciales por HTTP ni desde archivos de configuración.

La autenticación básica es opcional, pero `MLFLOW_TRACKING_USERNAME` y
`MLFLOW_TRACKING_PASSWORD` forman un par atómico: ambas deben estar definidas o
ambas ausentes. Una configuración parcial también falla cerrado al arrancar. No
incluyas estos valores en el repositorio, imágenes, logs o respuestas HTTP.

```bash
export MLFLOW_TRACKING_URI="https://mlflow.example.internal"
export MLFLOW_TRACKING_USERNAME="..."
export MLFLOW_TRACKING_PASSWORD="..."
```

`MODEL_API_ADMIN_TOKEN` es obligatoria y protege exclusivamente el refresh
administrativo. Debe proveerse sólo mediante el entorno de ejecución; si falta,
es vacía o contiene sólo espacios, el proceso falla cerrado al iniciar. No la
incluyas en el repositorio, imágenes, logs o respuestas HTTP.

```bash
export MODEL_API_ADMIN_TOKEN="..."
```

`MODEL_REFRESH_TIMEOUT_SECONDS` es opcional y define el límite del refresh
administrativo completo; su valor por defecto es `30`. Debe ser un número finito
positivo. El refresh carga en un proceso aislado: al vencer el límite, el proceso
HTTP termina el hijo con una espera de limpieza acotada, conserva el modelo activo
y responde `503`. La terminación no confirma ni informa una cancelación al proveedor.

```bash
export MODEL_REFRESH_TIMEOUT_SECONDS="30"
```

Por cada entrada del catálogo, el inicio resuelve su alias, retiene en memoria
el objeto cargado junto con `model_version` y `run_id`, y carga la versión
concreta resuelta mediante `models:/<modelo-registrado>/<versión>` una única
vez. Las solicitudes normales no realizan llamadas a MLflow.

## Smoke de model release

El smoke es una verificación explícitamente opt-in para el operador después de
una promoción manual. Ejecuta `refresh → health → predict` para
`invoice-review`; no inicia el servicio, no cambia aliases ni contacta MLflow
directamente.

Provee exclusivamente por entorno la URL de la instancia, el token
administrativo y la metadata esperada:

```bash
export MODEL_API_BASE_URL="https://model-api.example.internal"
export MODEL_API_ADMIN_TOKEN="..."
export MODEL_RELEASE_EXPECTED_MODEL_VERSION="..."
export MODEL_RELEASE_EXPECTED_RUN_ID="..."
uv run python scripts/model_release_smoke.py
```

Each HTTP request runs in an isolated child process with a 10-second total
operational deadline for connection, transfer, and reading. A server that sends
periodic bytes therefore cannot keep the smoke waiting indefinitely. On expiry,
the parent terminates the child, waits at most one second for cleanup, and only
if it remains alive kills it and waits at most one additional second before
returning a safe failure. The child does not write to stdout or stderr; its
temporary result contains only the HTTP response and is deleted before
continuing. The command does not print the URL or token; incomplete
configuration, HTTP errors, unexpected metadata, invalid responses, or
out-of-range probabilities exit nonzero.

## Verificación local

```bash
uv run pytest
uv run ruff check .
uv build
```

## Contenedor

Construye la imagen sin credenciales ni archivos de entorno:

```bash
docker build -t invoiceops-model-api .
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI="https://mlflow.example.internal" \
  -e MODEL_API_ADMIN_TOKEN="..." \
  invoiceops-model-api
```

El contenedor no inicia sin `MLFLOW_TRACKING_URI` y `MODEL_API_ADMIN_TOKEN`. Luego consulta
`http://127.0.0.1:8000/health`.

## Alcance actual

- Incluye `GET /health`.
- Incluye `GET /models`, que lista los IDs lógicos registrados en orden.
- Incluye las rutas genéricas `GET /models/{model_id}/health` y `POST /models/{model_id}/predict`; agregar un ID al catálogo no requiere agregar rutas FastAPI.
- El catálogo local tipado mapea cada ID lógico a un modelo registrado y a un alias no vacío:

| ID lógico | Modelo registrado | Alias |
| --- | --- | --- |
| `invoice-review` | `invoice-review-production` | `champion` |

- Un ID desconocido responde `404`. Un ID registrado que no pudo cargar responde `503`; el detalle de su respuesta de health incluye el ID lógico, el modelo registrado, el alias y un mensaje de loader no disponible. Fallas del proveedor, alias, metadata o carga degradan sólo el ID afectado y no exponen el error del proveedor.
- `POST /models/{model_id}/predict` exige exactamente las ocho features del artifact entrenado: `invoice_amount_cents`, `vendor_tenure_days`, `previous_incidents_12m`, `amount_vs_vendor_median`, `has_purchase_order`, `three_way_match`, `bank_account_recently_changed` y `country_risk`. Un body inválido responde el `422` estándar de FastAPI.
- Una predicción exitosa usa únicamente el modelo activo ya cargado en memoria y devuelve `model_id`, `model_version`, `run_id` y `probability`. `probability` es la probabilidad finita de la clase positiva (`1`, revisión manual); no aplica thresholds, recomendaciones ni Business Policy. La solicitud no resuelve aliases, carga artifacts ni llama a MLflow. Una salida o falla de inferencia no válida responde `503` con un detalle genérico.
- El contrato HTTP App ↔ Model API `v1`, con request/response de ejemplo, tipos, compatibilidad, responsabilidades y fallback del consumidor, se documenta en [`../invoiceops-app/README.md`](../invoiceops-app/README.md#contrato-app--model-api-int-01-v1). `country_risk` se recibe como string estricto: `low`, `medium` y `high` son los valores entrenados que envía la aplicación, pero este endpoint no restringe el dominio a esos tres valores.
- `GET /models/{model_id}/health` responde `200` cuando el modelo está activo y devuelve `model_id`, `registered_model`, `alias`, `model_version` y `run_id` realmente servidos. Un ID registrado sin carga sigue respondiendo `503`; un ID desconocido responde `404`. `GET /health` conserva su respuesta de proceso `200` con `{"status":"ok"}`.
- `POST /admin/models/{model_id}/refresh` exige el header `X-Model-Admin-Token` que coincida con `MODEL_API_ADMIN_TOKEN`. Resuelve y carga el champion actual en un proceso aislado antes de sustituir bajo lock la instancia activa. Las solicitudes predict/health sólo obtienen una instancia completa; timeout, salida anómala del hijo, error de transferencia o falla del proveedor conservan la versión anterior y responden `503` sin detalles del proveedor. Registra eventos seguros de intento, éxito o falla sin token, credenciales ni excepción cruda del proveedor.
- `scripts/model_release_smoke.py` es una verificación opt-in operada por entorno para `refresh → health → predict`; valida la versión/run esperados y una probabilidad finita entre 0 y 1, sin imprimir URL ni token.
- Docker Compose, CI y despliegue cloud están fuera de alcance; el consumidor de la aplicación está documentado en INT-01 y el trabajo de INT-03 fue absorbido por APP-08.
