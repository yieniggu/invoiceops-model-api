# InvoiceOps Model API

Servicio HTTP mínimo para exponer modelos de InvoiceOps. Usa un catálogo local
explícito y carga sus modelos desde MLflow durante el inicio del proceso.

## Inicio rápido

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
export MLFLOW_TRACKING_URI="https://mlflow.example.internal"
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

Por cada entrada del catálogo, el inicio resuelve su alias, retiene en memoria
el objeto cargado junto con `model_version` y `run_id`, y carga la versión
concreta resuelta mediante `models:/<modelo-registrado>/<versión>` una única
vez. Las solicitudes normales no realizan llamadas a MLflow.

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
docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI="https://mlflow.example.internal" invoiceops-model-api
```

El contenedor no inicia sin `MLFLOW_TRACKING_URI`. Luego consulta
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
- `GET /models/{model_id}/health` responde `503` incluso con el modelo cargado hasta API-04; API-03 no introduce un contrato exitoso para esa ruta. `GET /health` conserva su respuesta de proceso `200` con `{"status":"ok"}`.
- API-05 no agrega refresh/swap de modelos ni endpoints administrativos. Esas extensiones pertenecen a hitos posteriores.
- Docker Compose, CI, despliegue cloud, el consumidor de la aplicación e INT-03 están fuera de alcance.
