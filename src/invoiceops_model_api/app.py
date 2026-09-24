from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

MODEL_REGISTRY = {"invoice-review": None}


class PredictRequest(BaseModel):
    payload: dict[str, object]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, list[str]]:
    return {"models": sorted(MODEL_REGISTRY)}


@app.get("/models/{model_id}/health")
def model_health(model_id: str) -> None:
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model not found")
    if MODEL_REGISTRY[model_id] is None:
        raise HTTPException(status_code=503, detail="Model loader is unavailable")


@app.post("/models/{model_id}/predict")
def predict(model_id: str, request: PredictRequest) -> None:
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model not found")
    if MODEL_REGISTRY[model_id] is None:
        raise HTTPException(status_code=503, detail="Model loader is unavailable")
