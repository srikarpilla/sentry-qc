"""
test_api.py
-----------
Quick smoke test for the Sentry QC API.
If a live server is running on http://127.0.0.1:8000, it tests against that.
Otherwise, it automatically tests using FastAPI TestClient in-process!
"""
import io
import json
import urllib.request
import urllib.error
from PIL import Image
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

def is_server_running(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False

def make_sample_image(array: np.ndarray) -> bytes:
    img = Image.fromarray(array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def run_live_tests():
    print(f"Detected live server running at {BASE_URL}. Testing via HTTP...")
    # 1. Health
    with urllib.request.urlopen(f"{BASE_URL}/health") as resp:
        print("GET /health ->", json.loads(resp.read().decode()))

    # 2. Model Info
    with urllib.request.urlopen(f"{BASE_URL}/model-info") as resp:
        print("GET /model-info ->", json.loads(resp.read().decode()))

    # 3. Predict single
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    img_bytes = make_sample_image(np.ones((288, 288, 3), dtype=np.uint8) * 200)
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="sample.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        print("POST /predict ->", json.loads(resp.read().decode()))

def run_inprocess_tests():
    print("No live server detected on port 8000. Running in-process smoke test via TestClient...")
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        # 1. Health
        res = client.get("/health")
        print("GET /health ->", res.status_code, res.json())
        assert res.status_code == 200

        # 2. Model info
        res = client.get("/model-info")
        print("GET /model-info ->", res.status_code, res.json())
        assert res.status_code == 200

        # 3. Predict
        img_bytes = make_sample_image(np.ones((288, 288, 3), dtype=np.uint8) * 200)
        res = client.post("/predict", files={"file": ("clean_surface.jpg", img_bytes, "image/jpeg")})
        print("POST /predict ->", res.status_code, res.json())
        assert res.status_code == 200

        # 4. Batch
        res = client.post(
            "/predict/batch",
            files=[
                ("files", ("img1.jpg", img_bytes, "image/jpeg")),
                ("files", ("img2.jpg", make_sample_image(np.random.randint(0, 256, (288, 288, 3), dtype=np.uint8)), "image/jpeg")),
            ],
        )
        print("POST /predict/batch ->", res.status_code, res.json())
        assert res.status_code == 200

if __name__ == "__main__":
    if is_server_running(BASE_URL):
        run_live_tests()
    else:
        run_inprocess_tests()
    print("\n--- All tests completed successfully! ---")
