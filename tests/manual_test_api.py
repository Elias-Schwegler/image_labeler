import os
from fastapi.testclient import TestClient
from PIL import Image
from src.api import app
import io

client = TestClient(app)


def test_api_workflow():
    print("Starting API Test Workflow...")

    # 1. Setup Test Data
    test_dir = "data/raw/TEST_IMAGES"
    output_dir = "data/api_test_output"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create dummy image
    img_path = os.path.join(test_dir, "api_test_image.jpg")
    Image.new("RGB", (100, 100), color="blue").save(img_path)
    print(f"Created test image at: {img_path}")

    # 2. Test /label-image/ Endpoint
    print("\nTesting /label-image/ endpoint...")
    with open(img_path, "rb") as f:
        files = {"file": ("api_test_image.jpg", f, "image/jpeg")}
        response = client.post("/label-image/", files=files)

    if response.status_code == 200:
        print("✅ /label-image/ Success!")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ /label-image/ Failed: {response.status_code}")
        print(response.text)
        return

    # 3. Test /split-dataset/ Endpoint
    print("\nTesting /split-dataset/ endpoint...")
    payload = {
        "input_path": test_dir,
        "output_path": output_dir,
        "split_ratio": 0.5,
    }
    response = client.post("/split-dataset/", json=payload)

    if response.status_code == 200:
        print("✅ /split-dataset/ Success!")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ /split-dataset/ Failed: {response.status_code}")
        print(response.text)
        return

    print("\nAPI Test Workflow Completed Successfully.")


if __name__ == "__main__":
    test_api_workflow()
