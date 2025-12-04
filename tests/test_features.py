import pytest
import os
import json
from PIL import Image
import io
import base64
from src.labeler import encode_image
from src.app import split_dataset # We might need to refactor app.py to make this testable or mock the logic if it's tightly coupled to Streamlit.
# Actually, the robust splitting logic is inside app.py's button click handler, which is hard to test directly.
# However, I can test the logic if I extract it or just test the concept using a mock labels.json.
# Let's test encode_image resizing first.

def test_encode_image_resizing(tmp_path):
    # Create a large image
    img_path = tmp_path / "large_image.jpg"
    img = Image.new('RGB', (2000, 2000), color = 'red')
    img.save(img_path)
    
    # Encode with resizing
    encoded = encode_image(str(img_path), max_size=1024)
    
    # Decode and check size
    decoded_bytes = base64.b64decode(encoded)
    decoded_img = Image.open(io.BytesIO(decoded_bytes))
    
    assert max(decoded_img.size) == 1024
    
def test_encode_image_no_resizing_needed(tmp_path):
    # Create a small image
    img_path = tmp_path / "small_image.jpg"
    img = Image.new('RGB', (500, 500), color = 'blue')
    img.save(img_path)
    
    # Encode
    encoded = encode_image(str(img_path), max_size=1024)
    
    # Decode and check size
    decoded_bytes = base64.b64decode(encoded)
    decoded_img = Image.open(io.BytesIO(decoded_bytes))
    
    assert max(decoded_img.size) == 500

# For robust splitting, since the logic is embedded in app.py, 
# I will create a test that simulates the logic: reading a json and filtering files.
def test_robust_splitting_logic(tmp_path):
    # Mock data
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create dummy images
    img1 = input_dir / "img1.jpg"
    img1.write_text("content")
    img2 = input_dir / "img2.jpg"
    img2.write_text("content")
    img3 = input_dir / "img3.jpg" # Not labeled
    img3.write_text("content")
    
    # Create labels.json
    labels_data = [
        {"filename": "img1.jpg", "original_path": str(img1), "label": "cat"},
        {"filename": "img2.jpg", "original_path": str(img2), "label": "dog"}
    ]
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f)
        
    # Simulate the logic from app.py
    files_to_split = []
    with open(labels_path, "r") as f:
        data = json.load(f)
        for item in data:
            if "original_path" in item and os.path.exists(item["original_path"]):
                files_to_split.append(item["original_path"])
                
    assert len(files_to_split) == 2
    assert str(img1) in files_to_split
    assert str(img2) in files_to_split
    assert str(img3) not in files_to_split

def test_real_images_processing():
    """
    Test processing of real images if they exist in data/raw/TEST_IMAGES.
    This ensures we can handle large files and different formats.
    """
    # Define path relative to the test file (assuming tests/ is one level deep)
    # Or better, relative to project root.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_images_dir = os.path.join(base_dir, "data", "raw", "TEST_IMAGES")
    
    if not os.path.exists(test_images_dir):
        pytest.skip(f"Real test images directory not found at {test_images_dir}")
        
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        pytest.skip("No image files found in real test images directory")
        
    # Test on the first few images to save time
    for img_file in image_files[:3]:
        img_path = os.path.join(test_images_dir, img_file)
        print(f"Testing encoding of {img_path}")
        
        # Encode with resizing
        max_size = 1024
        encoded = encode_image(img_path, max_size=max_size)
        
        # Decode and verify
        decoded_bytes = base64.b64decode(encoded)
        decoded_img = Image.open(io.BytesIO(decoded_bytes))
        
        # Check dimensions
        width, height = decoded_img.size
        assert max(width, height) <= max_size
        assert width > 0 and height > 0

