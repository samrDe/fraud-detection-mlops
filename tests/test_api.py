def sample_features():
    return {
        "Time": 0,
        "Amount": 10,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_valid_payload(client):
    response = client.post(
        "/predict",
        json={"features": sample_features()}
    )

    assert response.status_code == 200, (
        f"Status: {response.status_code}, "
        f"Response: {response.text}"
    )

    assert response.json()["prediction"] == 0


def test_predict_rejects_missing_features(client):
    payload = sample_features()
    payload.pop("V1")

    response = client.post(
        "/predict",
        json={"features": payload}
    )

    assert response.status_code == 422
