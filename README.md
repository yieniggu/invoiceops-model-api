# InvoiceOps Model API

Servicio HTTP mínimo para exponer modelos de InvoiceOps. Expone un registro local explícito de IDs lógicos, sin conectarse a MLflow ni a servicios externos.

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
- Incluye `GET /models`, que lista los IDs lógicos registrados. Actualmente el registro local contiene `invoice-review`.
- Incluye las rutas genéricas `GET /models/{model_id}/health` y `POST /models/{model_id}/predict`; agregar un ID al registro no requiere agregar rutas FastAPI.
- Un ID desconocido responde `404` y un ID registrado sin loader responde `503`.
- `POST /models/{model_id}/predict` exige un body Pydantic con `payload` como objeto. Un body inválido responde el `422` estándar de FastAPI.
- No hay loader ni inferencia todavía; por ello no existe respuesta exitosa de `predict` en este hito.
- No incluye integración con MLflow, Model Registry, catálogo, credenciales, Docker Compose, CI ni despliegue cloud.
