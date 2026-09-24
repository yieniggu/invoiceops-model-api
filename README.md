# InvoiceOps Model API

Servicio HTTP mínimo para exponer modelos de InvoiceOps. Usa un catálogo local
explícito de modelos y no se conecta a MLflow ni a servicios externos.

## Inicio rápido

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
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
docker run --rm -p 8000:8000 invoiceops-model-api
```

Luego consulta `http://127.0.0.1:8000/health`.

## Alcance actual

- Incluye `GET /health`.
- Incluye `GET /models`, que lista los IDs lógicos registrados en orden.
- Incluye las rutas genéricas `GET /models/{model_id}/health` y `POST /models/{model_id}/predict`; agregar un ID al catálogo no requiere agregar rutas FastAPI.
- El catálogo local tipado mapea cada ID lógico a un modelo registrado y a un alias no vacío:

| ID lógico | Modelo registrado | Alias |
| --- | --- | --- |
| `invoice-review` | `invoice-review-production` | `champion` |

- Un ID desconocido responde `404`. Un ID registrado sin loader responde `503`; el detalle de su respuesta de health incluye el ID lógico, el modelo registrado, el alias y un mensaje de loader no disponible.
- `POST /models/{model_id}/predict` exige un body Pydantic con `payload` como objeto. Un body inválido responde el `422` estándar de FastAPI.
- No hay loader ni inferencia todavía, por lo que este hito no tiene respuesta exitosa de `predict`.
- API-03 es responsable de la integración con MLflow, la resolución de alias y la carga de modelos. Este catálogo API-02 no se conecta a MLflow, no resuelve alias, no carga versiones de modelos ni IDs de ejecución, no realiza predicciones, no refresca estado, no lee credenciales ni entorno, ni realiza llamadas de red.
- Docker Compose, CI, despliegue cloud, el consumidor de la aplicación e INT-03 están fuera de alcance.
