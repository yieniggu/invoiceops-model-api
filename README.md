# InvoiceOps Model API

Servicio HTTP mínimo para exponer modelos de InvoiceOps. Este primer hito sólo entrega un health check determinista; no se conecta a MLflow ni a servicios externos.

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
- No incluye endpoints de modelos o predicción.
- No incluye integración con MLflow, Model Registry, catálogo, credenciales, Docker Compose, CI ni despliegue cloud.
