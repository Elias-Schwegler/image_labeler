import os
import requests
import sys
import time

# Configuration
API_URL = "http://127.0.0.1:8000"
INPUT_DIR = "data/raw/TEST_IMAGES"
OUTPUT_DIR = "data/processed/api_output"

def verify_api():
    print(f"🔬 Starting API Verification")
    print(f"   Input:  {INPUT_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    
    # Check connection
    try:
        requests.get(f"{API_URL}/docs", timeout=5)
        print("✅ API is reachable.")
    except Exception:
        print("❌ Could not connect to API. Please run `uvicorn src.api:app --reload` in a separate terminal.")
        return

    # Check images
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory {INPUT_DIR} does not exist.")
        return
        
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print(f"❌ No images found in {INPUT_DIR}.")
        return

    # Label Images
    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        print(f"   Processing {img_name}...")
        try:
            with open(img_path, "rb") as f:
                files = {"file": (img_name, f, "image/jpeg")}
                response = requests.post(f"{API_URL}/label-image/", files=files)
            if response.status_code == 200:
                print(f"     ✅ Label: {response.json().get('label')}")
            else:
                print(f"     ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"     ❌ Error: {e}")

    # Split
    print(f"✂️  Triggering Dataset Split...")
    try:
        payload = {"input_path": INPUT_DIR, "output_path": OUTPUT_DIR, "split_ratio": 0.8}
        response = requests.post(f"{API_URL}/split-dataset/", json=payload)
        if response.status_code == 200:
            print(f"   ✅ Split Successful: {response.json()}")
        else:
            print(f"   ❌ Split Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    verify_api()
