"""API tests. They need no CSV and no model file: a tiny fake model is injected."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import FEATURES, JsonFormatter, app, logger


class FakeModel:
    """Stands in for the champion pipeline; returns the shift hours times 5."""

    def predict(self, df):
        assert list(df.columns) == FEATURES
        return np.asarray(df["shift_hours"], dtype=float) * 5


VALID = {
    "loom_age_yr": 5,
    "thread_count": 120,
    "power_cuts_min": 60,
    "weaver_exp_yr": 10,
    "humidity_pct": 55,
    "shift_hours": 8.0,
    "fabric": "cotton",
    "shed": "Shed 1",
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        app.state.model = FakeModel()  # replace whatever lifespan loaded
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.json()["prediction"] == pytest.approx(40.0)


@pytest.mark.parametrize("power_cuts", [251, 1000])
def test_power_cuts_above_250_rejected(client, power_cuts):
    r = client.post("/predict", json={**VALID, "power_cuts_min": power_cuts})
    assert r.status_code == 422


def test_power_cuts_250_accepted(client):
    r = client.post("/predict", json={**VALID, "power_cuts_min": 250})
    assert r.status_code == 200


@pytest.mark.parametrize("age", [-0.1, -5])
def test_negative_loom_age_rejected(client, age):
    r = client.post("/predict", json={**VALID, "loom_age_yr": age})
    assert r.status_code == 422


def test_missing_field_rejected(client):
    body = dict(VALID)
    del body["shed"]
    assert client.post("/predict", json=body).status_code == 422


def test_unknown_category_rejected(client):
    assert client.post("/predict", json={**VALID, "fabric": "wool"}).status_code == 422


def test_no_model_gives_503(client):
    app.state.model = None
    assert client.post("/predict", json=VALID).status_code == 503


def test_json_formatter_one_line_of_valid_json():
    import logging

    rec = logging.LogRecord("mp212", logging.INFO, __file__, 1, "prediction", None, None)
    rec.fields = {"prediction": 12.5}
    out = JsonFormatter().format(rec)
    assert "\n" not in out
    data = json.loads(out)
    assert data["message"] == "prediction" and data["prediction"] == 12.5


def test_prediction_is_logged_as_one_json_line(client):
    import io
    import logging

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    try:
        client.post("/predict", json=VALID)
    finally:
        logger.removeHandler(handler)
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["message"] == "prediction" and data["prediction"] == pytest.approx(40.0)


def test_logger_writes_to_stdout_not_a_file():
    import logging

    assert logger.handlers
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
