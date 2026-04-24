import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------
# FIXTURE
# ---------------------------
@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------------------------
# ROOT
# ---------------------------
def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "message" in data
    assert "current_model" in data


# ---------------------------
# LIST MODELS
# ---------------------------
def test_list_models(client):
    response = client.get("/models")

    assert response.status_code == 200
    data = response.json()

    assert "available_models" in data
    assert isinstance(data["available_models"], list)


# ---------------------------
# SET MODEL SUCCESS
# ---------------------------
def test_set_model_success(client):
    # prima recupero i modelli disponibili
    models = client.get("/models").json()["available_models"]

    if not models:
        pytest.skip("No models available")

    model_name = models[0]

    response = client.get(f"/set_model?model_name={model_name}")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["loaded_model"] == model_name


# ---------------------------
# SET MODEL FAIL
# ---------------------------
def test_set_model_not_found(client):
    response = client.get("/set_model?model_name=notamodel.pkl")

    assert response.status_code == 404
    assert "Model not found" in response.text


# ---------------------------
# PREDICT SUCCESS
# ---------------------------
def test_predict_success(client):
    payload = {
        "text": "This is a test news article",
        "label" : "-1"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "text" in data
    assert "label" in data
    assert isinstance(data["label"], int)


# ---------------------------
# PREDICT EMPTY TEXT
# ---------------------------
def test_predict_empty_text(client):
    payload = {
        "text": "",
        "label" : "-1"
    }

    response = client.post("/predict", json=payload)

    # dipende dal tuo schema, ma tipicamente:
    assert response.status_code in [200, 422]


# ---------------------------
# MODEL SWITCH EFFECT
# ---------------------------
def test_model_switch_effect(client):
    models = client.get("/models").json()["available_models"]

    if len(models) < 2:
        pytest.skip("Need at least 2 models")

    # switch al primo
    client.get(f"/set_model?model_name={models[0]}")
    res1 = client.post("/predict", json={"text": "Breaking news test", "label": "-1"}).json()

    # switch al secondo
    client.get(f"/set_model?model_name={models[1]}")
    res2 = client.post("/predict", json={"text": "Breaking news test", "label": "-1"}).json()

    # non garantito diverso, ma almeno validi
    assert "label" in res1
    assert "label" in res2