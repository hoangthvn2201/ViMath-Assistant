# tests/test_api.py

import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Path to a sample math image (you should place one under this path)
SAMPLE_IMAGE_PATH = "tests/sample_images/math_example_1.png"

@pytest.mark.skipif(not os.path.exists(SAMPLE_IMAGE_PATH), reason="Sample image not found.")
def test_solve_math_problem():
    question = "Tìm nghiệm của phương trình x^2 - 5x + 6 = 0"

    with open(SAMPLE_IMAGE_PATH, "rb") as img_file:
        response = client.post(
            "/solve",
            files={"image": ("math_example_1.png", img_file, "image/png")},
            data={"question": question}
        )

    assert response.status_code == 200
    json_data = response.json()

    assert "question" in json_data
    assert "ocr_text" in json_data
    assert "retrieved_examples" in json_data
    assert "answer" in json_data

    assert question in json_data["question"]
    assert isinstance(json_data["retrieved_examples"], list)
    assert isinstance(json_data["answer"], str)
    assert len(json_data["answer"]) > 0

    print("LLM Answer:", json_data["answer"])