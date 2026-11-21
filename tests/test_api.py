from __future__ import annotations

import logging
import os
from dataclasses import replace

import pytest
from conftest import AUTH, SEED, TOKEN, WORKSPACE, make_settings, ref_body, sample_request
from fastapi.testclient import TestClient

from invoiceops_model_api.app import create_app
from invoiceops_model_api.settings import ConfigError, Settings


def codes(body: dict) -> list[str]:
    return [r["code"] for r in body["reasons"]]


# ---- auth -------------------------------------------------------------------------------


def test_liveness_is_open(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/deployments"),
        ("GET", "/metrics"),
        ("POST", "/deploy-check"),
        ("PUT", "/deployments/d1"),
        ("DELETE", "/deployments/d1"),
        ("POST", "/deployments/d1/predict"),
        ("POST", "/deployments/d1/predict-batch"),
    ],
)
def test_every_route_requires_the_service_credential(
    client: TestClient, method: str, path: str
) -> None:
    assert client.request(method, path, json={}).status_code == 401
    wrong = {"Authorization": "Bearer " + "x" * 40}
    assert client.request(method, path, json={}, headers=wrong).status_code == 401
    assert (
        client.request(method, path, json={}, headers={"Authorization": TOKEN}).status_code == 401
    )


def test_short_or_missing_token_config_fails_closed(store) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({"MLFLOW_TRACKING_URI": "http://x", "OFFICIAL_VALIDATION_SEED": "1"})
    with pytest.raises(ConfigError):
        Settings.from_env(
            {
                "MODEL_API_SERVICE_TOKEN": "short",
                "MLFLOW_TRACKING_URI": "http://x",
                "OFFICIAL_VALIDATION_SEED": "1",
            }
        )
    # Even if a short token slipped into Settings, nothing can authenticate.
    weak = TestClient(create_app(make_settings(store.uri, service_token="abc")))
    assert weak.get("/deployments", headers={"Authorization": "Bearer abc"}).status_code == 401


def test_missing_validation_seed_fails_closed(store, versions) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({"MODEL_API_SERVICE_TOKEN": TOKEN, "MLFLOW_TRACKING_URI": "http://x"})
    app = create_app(make_settings(store.uri, official_validation_seed=None))
    c = TestClient(app)
    r = c.post("/deploy-check", json=ref_body(versions, "good"), headers=AUTH)
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "GATE_UNAVAILABLE"
    r = c.put("/deployments/d1", json=ref_body(versions, "good"), headers=AUTH)
    assert r.status_code == 503


# ---- deploy check -----------------------------------------------------------------------


def test_deploy_check_passes_for_a_real_skops_model(client: TestClient, versions) -> None:
    r = client.post("/deploy-check", json=ref_body(versions, "good"), headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["passed"] is True and body["reasons"] == []
    m = body["metrics"]
    assert m["auc"] >= 0.70 and m["dummyAuc"] == pytest.approx(0.5)
    assert m["lift"] >= 0.02
    assert body["model"]["runId"] and body["model"]["version"] == versions["good"][1]
    assert body["featureSchemaVersion"] == "invoice-features-v2"
    assert str(SEED) not in r.text
    # deploy-check does not activate anything
    assert client.get("/deployments", headers=AUTH).json()["loaded"] == 0


def test_hist_gradient_boosting_also_passes(client: TestClient, versions) -> None:
    r = client.post("/deploy-check", json=ref_body(versions, "hgb"), headers=AUTH)
    assert r.json()["passed"] is True, r.text


def test_low_auc_fails_and_fake_logged_metric_is_ignored(client: TestClient, versions) -> None:
    # The run logged auc=0.99; the server evaluates its own AUC on the official data.
    r = client.post("/deploy-check", json=ref_body(versions, "dummy_fake"), headers=AUTH)
    body = r.json()
    assert body["passed"] is False
    assert "AUC_BELOW_MIN" in codes(body) and "AUC_LIFT_BELOW_MIN" in codes(body)
    assert body["metrics"]["auc"] == pytest.approx(0.5)
    detail = next(x for x in body["reasons"] if x["code"] == "AUC_BELOW_MIN")["detail"]
    assert detail["minAuc"] == 0.70


def test_lift_over_dummy_is_required_even_if_min_auc_is_met(store, versions) -> None:
    c = TestClient(create_app(make_settings(store.uri, gate_min_auc=0.5, gate_min_lift=0.02)))
    body = c.post("/deploy-check", json=ref_body(versions, "dummy_fake"), headers=AUTH).json()
    assert codes(body) == ["AUC_LIFT_BELOW_MIN"]


def test_pickle_flavor_is_rejected(client: TestClient, versions) -> None:
    body = client.post("/deploy-check", json=ref_body(versions, "pickle"), headers=AUTH).json()
    assert body["passed"] is False
    assert codes(body) == ["PICKLE_FORMAT_NOT_ALLOWED"]


def test_type_outside_allowlist_is_rejected(client: TestClient, versions) -> None:
    body = client.post("/deploy-check", json=ref_body(versions, "svc"), headers=AUTH).json()
    assert codes(body) == ["UNTRUSTED_TYPE"]
    assert any("SVC" in t for t in body["reasons"][0]["detail"]["types"])


def test_wrong_signature_is_rejected(client: TestClient, versions) -> None:
    body = client.post("/deploy-check", json=ref_body(versions, "seven"), headers=AUTH).json()
    assert codes(body) == ["SIGNATURE_MISMATCH"]
    assert body["reasons"][0]["detail"]["expected"][0] == "amount_pesos"


def test_unknown_version_and_alias_like_versions(client: TestClient, versions) -> None:
    body = client.post(
        "/deploy-check", json=ref_body(versions, "good", version="999"), headers=AUTH
    ).json()
    assert codes(body) == ["MODEL_VERSION_NOT_FOUND"]
    body = client.post(
        "/deploy-check", json=ref_body(versions, "good", version="champion"), headers=AUTH
    ).json()
    assert codes(body) == ["INVALID_REFERENCE"]
    body = client.post(
        "/deploy-check", json=ref_body(versions, "good", workspace="Bad_WS"), headers=AUTH
    ).json()
    assert codes(body) == ["INVALID_REFERENCE"]


def test_oversized_artifact_is_rejected(store, versions) -> None:
    c = TestClient(create_app(make_settings(store.uri, max_model_bytes=100)))
    body = c.post("/deploy-check", json=ref_body(versions, "good"), headers=AUTH).json()
    assert codes(body) == ["ARTIFACT_TOO_LARGE"]


# ---- load / predict / unload ------------------------------------------------------------


def test_predict_racing_a_redeploy_retries_once_on_the_current_slot(
    client: TestClient, versions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The slot a request holds is replaced and unloaded before it reaches the worker."""
    assert (
        client.put("/deployments/dep-1", json=ref_body(versions, "good"), headers=AUTH).status_code
        == 200
    )
    registry = client.app.state.registry
    supervisor = client.app.state.supervisor
    old = registry.get("dep-1")
    real_predict = supervisor.predict
    calls: list[str] = []

    def racing_predict(slot: str, rows):
        calls.append(slot)
        if len(calls) == 1:  # the redeploy lands between get_loaded() and predict()
            new_slot, staged = supervisor.stage(old.artifact_path)
            supervisor.activate(new_slot, staged, old.columns)
            replaced = registry.put(replace(old, slot=new_slot, artifact_path=staged))
            supervisor.unload(replaced.slot, replaced.artifact_path)
        return real_predict(slot, rows)

    monkeypatch.setattr(supervisor, "predict", racing_predict)
    response = client.post("/deployments/dep-1/predict", json=sample_request(), headers=AUTH)
    assert response.status_code == 200, response.text
    assert len(calls) == 2 and calls[0] == old.slot and calls[1] != old.slot


def test_predict_does_not_retry_when_the_slot_was_not_replaced(
    client: TestClient, versions, monkeypatch: pytest.MonkeyPatch
) -> None:
    from invoiceops_model_api.inference import InferenceUnavailable

    client.put("/deployments/dep-1", json=ref_body(versions, "good"), headers=AUTH)
    calls: list[str] = []

    def failing(slot: str, rows):
        calls.append(slot)
        raise InferenceUnavailable("worker crashed")

    monkeypatch.setattr(client.app.state.supervisor, "predict", failing)
    response = client.post("/deployments/dep-1/predict", json=sample_request(), headers=AUTH)
    assert response.status_code == 503 and len(calls) == 1


def test_load_predict_returns_version_and_run(client: TestClient, versions) -> None:
    r = client.put("/deployments/dep-1", json=ref_body(versions, "good"), headers=AUTH)
    assert r.status_code == 200, r.text
    run_id = r.json()["runId"]
    p = client.post("/deployments/dep-1/predict", json=sample_request(), headers=AUTH)
    assert p.status_code == 200, p.text
    body = p.json()
    assert 0 <= body["probability"] <= 1
    assert body["modelVersion"] == versions["good"][1] and body["runId"] == run_id
    assert body["registeredModel"] == versions["good"][0]
    batch = client.post(
        "/deployments/dep-1/predict-batch",
        json={"items": [sample_request(), sample_request()]},
        headers=AUTH,
    ).json()
    assert batch["probabilities"] == [body["probability"]] * 2


def test_two_models_and_two_deployments_of_different_versions_coexist(
    client: TestClient, versions
) -> None:
    assert (
        client.put("/deployments/a", json=ref_body(versions, "good"), headers=AUTH).status_code
        == 200
    )
    assert (
        client.put("/deployments/b", json=ref_body(versions, "hgb"), headers=AUTH).status_code
        == 200
    )
    health = client.get("/deployments", headers=AUTH).json()
    assert health["loaded"] == 2
    assert {d["registeredModel"] for d in health["deployments"]} == {"m-good", "m-hgb"}
    assert all(
        d["runId"] and d["version"] and d["workspace"] == WORKSPACE for d in health["deployments"]
    )


def test_failed_gate_does_not_load(client: TestClient, versions) -> None:
    r = client.put("/deployments/bad", json=ref_body(versions, "dummy_fake"), headers=AUTH)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "GATE_FAILED"
    assert client.get("/deployments", headers=AUTH).json()["loaded"] == 0


def test_load_cap_and_replacement_and_unload(store, versions) -> None:
    c = TestClient(create_app(make_settings(store.uri, max_deployments=2)))
    for dep in ("a", "b"):
        assert (
            c.put(f"/deployments/{dep}", json=ref_body(versions, "good"), headers=AUTH).status_code
            == 200
        )
    over = c.put("/deployments/c", json=ref_body(versions, "good"), headers=AUTH)
    assert over.status_code == 409 and over.json()["detail"]["code"] == "CAPACITY_EXCEEDED"
    # re-loading an existing id (replacing its model) does not need extra capacity
    assert c.put("/deployments/a", json=ref_body(versions, "hgb"), headers=AUTH).status_code == 200
    assert c.get("/deployments", headers=AUTH).json()["loaded"] == 2
    assert c.delete("/deployments/a", headers=AUTH).json() == {
        "deploymentId": "a",
        "unloaded": True,
    }
    assert c.delete("/deployments/a", headers=AUTH).json()["unloaded"] is False
    assert c.post("/deployments/a/predict", json=sample_request(), headers=AUTH).status_code == 404
    assert c.put("/deployments/c", json=ref_body(versions, "good"), headers=AUTH).status_code == 200


def test_invalid_invoice_does_not_echo_values_or_log_them(
    client: TestClient, versions, caplog: pytest.LogCaptureFixture
) -> None:
    client.put("/deployments/dep-1", json=ref_body(versions, "good"), headers=AUTH)
    bad = sample_request()
    bad["invoice"]["amountPesos"] = "SECRET-AMOUNT-123"
    with caplog.at_level(logging.DEBUG):
        r = client.post("/deployments/dep-1/predict", json=bad, headers=AUTH)
        ok = client.post("/deployments/dep-1/predict", json=sample_request(), headers=AUTH)
    assert r.status_code == 422 and "SECRET-AMOUNT-123" not in r.text
    assert ok.status_code == 200
    assert "Proveedor" not in caplog.text and "1250000" not in caplog.text


def test_priors_of_other_suppliers_are_ignored_and_future_priors_too(
    client: TestClient, versions
) -> None:
    client.put("/deployments/dep-1", json=ref_body(versions, "good"), headers=AUTH)
    base = sample_request()
    noisy = sample_request()
    noisy["priorInvoices"] += [
        {
            "supplierName": "Otro",
            "amountPesos": 1,
            "createdAt": "2025-05-02T00:00:00Z",
            "businessStatus": "REJECTED",
        },
        {
            "supplierName": "Proveedor 001",
            "amountPesos": 1,
            "createdAt": "2025-06-04T00:00:00Z",
            "businessStatus": "REJECTED",
        },
    ]
    a = client.post("/deployments/dep-1/predict", json=base, headers=AUTH).json()["probability"]
    b = client.post("/deployments/dep-1/predict", json=noisy, headers=AUTH).json()["probability"]
    assert a == b


def test_workspace_is_sent_as_a_header_without_touching_process_state(
    versions, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakeservers import FakeTracking

    monkeypatch.delenv("MLFLOW_WORKSPACE", raising=False)
    server = FakeTracking(
        [],
        {},
        {"/api/2.0/mlflow/model-versions/get": (404, {"error_code": "RESOURCE_DOES_NOT_EXIST"})},
    )
    try:
        c = TestClient(create_app(make_settings(server.url)))
        body = c.post(
            "/deploy-check", json=ref_body(versions, "good", workspace="group-b"), headers=AUTH
        ).json()
        c.app.state.supervisor.shutdown()
    finally:
        server.close()
    assert codes(body) == ["MODEL_VERSION_NOT_FOUND"]
    assert [r["headers"].get("X-MLFLOW-WORKSPACE") for r in server.requests] == ["group-b"]
    assert "MLFLOW_WORKSPACE" not in os.environ  # nothing global to restore


# ---- hardening: rounding, classes, seed -------------------------------------------------


def test_returned_metrics_are_rounded_to_two_decimals(client: TestClient, versions) -> None:
    body = client.post("/deploy-check", json=ref_body(versions, "good"), headers=AUTH).json()
    for key in ("auc", "dummyAuc", "lift"):
        assert body["metrics"][key] == round(body["metrics"][key], 2)
    failing = client.post("/deploy-check", json=ref_body(versions, "dummy_fake"), headers=AUTH)
    detail = next(r for r in failing.json()["reasons"] if r["code"] == "AUC_BELOW_MIN")["detail"]
    assert detail["auc"] == round(detail["auc"], 2)


def test_non_integer_classes_are_invalid_classes_not_a_500(client: TestClient, versions) -> None:
    r = client.post("/deploy-check", json=ref_body(versions, "strclasses"), headers=AUTH)
    assert r.status_code == 200, r.text
    assert codes(r.json()) == ["INVALID_CLASSES"]
    r = client.put("/deployments/s", json=ref_body(versions, "strclasses"), headers=AUTH)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "GATE_FAILED"


@pytest.mark.parametrize(
    "classes", [["rej", "ok"], ["0", "1"], [0.5, 1.0], [None, 1], [0, 2], [1, 1], [0, 1, 2]]
)
def test_positive_index_rejects_anything_but_binary_integer_labels(classes) -> None:
    from invoiceops_model_api.model_runtime import ModelRejected, positive_index

    class Fake:
        classes_ = classes

    with pytest.raises(ModelRejected) as info:
        positive_index(Fake())
    assert info.value.code == "INVALID_CLASSES"


def test_positive_index_accepts_integer_like_labels() -> None:
    import numpy as np

    from invoiceops_model_api.model_runtime import positive_index

    class A:
        classes_ = np.array([0, 1])

    class B:
        classes_ = np.array([1.0, 0.0])

    class C:
        classes_ = np.array([False, True])

    assert (positive_index(A()), positive_index(B()), positive_index(C())) == (1, 0, 1)


ENV = {"MODEL_API_SERVICE_TOKEN": TOKEN, "MLFLOW_TRACKING_URI": "http://mlflow.internal:5000"}


@pytest.mark.parametrize(
    "seed",
    ["1", "424242", "abc", "a" * 40, "ab" * 20, "x" * 21, "12345678901234567890123", "a!" * 20],
)
def test_weak_validation_seed_is_refused(seed: str) -> None:
    with pytest.raises(ConfigError, match="128 bits"):
        Settings.from_env({**ENV, "OFFICIAL_VALIDATION_SEED": seed})


def test_strong_seed_is_accepted_and_derived_by_hashing() -> None:
    hex_seed = "9f2c5d71e0a84b36c7d1f0958e2a6b43"  # 32 hex chars = 128 bits
    b64_seed = "QUJDREVGR0hJSktMTU5PUA"  # 22 base64 chars
    derived = []
    for raw in (hex_seed, b64_seed):
        settings = Settings.from_env({**ENV, "OFFICIAL_VALIDATION_SEED": raw})
        seed = settings.official_validation_seed
        assert seed is not None and seed.bit_length() > 128
        assert str(seed) not in repr(settings) and raw not in repr(settings)
        derived.append(seed)
    assert derived[0] != int(hex_seed, 16)  # hashed, not parsed
    assert (
        derived[0]
        == Settings.from_env({**ENV, "OFFICIAL_VALIDATION_SEED": hex_seed}).official_validation_seed
    )  # deterministic


def test_http_flow_runs_on_the_seed_derived_from_the_environment(store, versions) -> None:
    """Environment secret -> derive_validation_seed -> app -> gate, observed through HTTP."""
    from invoiceops_model_api.checks import official_frame
    from invoiceops_model_api.settings import derive_validation_seed

    secret = "9f2c5d71e0a84b36c7d1f0958e2a6b43"
    other = "0b7e4a19c3d58f62a1e9d0c7b4f35a86"
    env = {
        **ENV,
        "MLFLOW_TRACKING_URI": store.uri,
        "OFFICIAL_VALIDATION_SEED": secret,
        "OFFICIAL_VALIDATION_ROWS": "2500",
    }
    settings = Settings.from_env(env)
    assert settings.official_validation_seed == derive_validation_seed(secret)
    app = create_app(settings)
    try:
        client = TestClient(app)
        body = client.post("/deploy-check", json=ref_body(versions, "good"), headers=AUTH).json()
        # The frame the gate used is the one of THAT derived seed, not of a neighbouring secret.
        used = official_frame(settings)
        assert body["passed"] is True and body["metrics"]["validationRows"] == len(used)
        reference = official_frame(
            make_settings(store.uri, official_validation_seed=derive_validation_seed(secret))
        )
        different = official_frame(Settings.from_env({**env, "OFFICIAL_VALIDATION_SEED": other}))
        assert used.equals(reference)
        assert not used.equals(different)
        assert "OFFICIAL_VALIDATION_SEED" not in client.get("/deployments", headers=AUTH).text
        assert secret not in client.get("/deployments", headers=AUTH).text
    finally:
        app.state.supervisor.shutdown()


# ---- /predict accepts only current statuses -----------------------------------------


@pytest.mark.parametrize("value", ["2025-05-02T10:00:00Z", None, ""])
def test_decided_at_is_rejected_with_422_and_a_clear_message(client: TestClient, value) -> None:
    body = sample_request()
    body["priorInvoices"][0]["decidedAt"] = value
    for path, payload in (
        ("/deployments/d1/predict", body),
        ("/deployments/d1/predict-batch", {"items": [body]}),
    ):
        response = client.post(path, json=payload, headers=AUTH)
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "INVALID_REQUEST"
        issue = next(i for i in detail["issues"] if i["type"] == "decided_at_not_accepted")
        assert issue["message"] == "/predict acepta solo estados vigentes"
        assert "priorInvoices" in issue["loc"]
        assert "2025-05-02" not in response.text  # the submitted value is never echoed


def test_priors_without_decided_at_are_still_accepted_by_validation(client: TestClient) -> None:
    response = client.post("/deployments/d1/predict", json=sample_request(), headers=AUTH)
    assert response.status_code == 404  # valid body: it only fails because d1 is not loaded
