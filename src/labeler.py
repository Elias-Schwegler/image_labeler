import os
import base64
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import threading

# Load environment variables
load_dotenv()

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-vl-4b")

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
model_lock = threading.Lock()


def encode_image(image_path: str) -> str:
    """
    Encode an image file to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def label_image(
    image_path: str,
    prompt: str = "Describe this image and provide a label.",
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Send an image to the local LM Studio model and get a structured label response.
    """
    if progress_callback:
        progress_callback(0.1, "Encoding image...")
    base64_image = encode_image(image_path)

    # JSON Schema for structured output
    json_schema = {
        "name": "image_label_response",
        "strict": "true",
        "schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "A short, concise label for the image.",
                },
                "description": {
                    "type": "string",
                    "description": "A detailed description of the image content.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of relevant tags.",
                },
            },
            "required": ["label", "description", "tags"],
        },
    }

    try:
        if progress_callback:
            progress_callback(0.3, "Sending request to LM Studio...")

        with model_lock:
            response = client.chat.completions.create(
                model=LM_STUDIO_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that labels images. Always output in JSON format.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                response_format={"type": "json_schema", "json_schema": json_schema},
                temperature=0.7,
            )

        content = response.choices[0].message.content
        if progress_callback:
            progress_callback(0.9, "Processing response...")
        return json.loads(content)

    except Exception as e:
        print(f"Error labeling image {image_path}: {e}")
        return {
            "label": "error",
            "description": f"Failed to process image: {str(e)}",
            "tags": [],
        }
