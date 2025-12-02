from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import tempfile
from .labeler import label_image
from .splitter import split_dataset, organize_dataset
from .data_loader import get_image_files

app = FastAPI(title="Image Labeler API")


class SplitRequest(BaseModel):
    input_path: str
    output_path: str
    split_ratio: float = 0.8


@app.post("/label-image/")
async def api_label_image(file: UploadFile = File(...)):
    """
    Upload an image and get its label/description from LM Studio.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = label_image(tmp_path)
        return result
    finally:
        os.remove(tmp_path)


@app.post("/split-dataset/")
async def api_split_dataset(request: SplitRequest):
    """
    Split a dataset located at input_path into train/test sets at output_path.
    """
    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail="Input path not found")

    try:
        image_files = get_image_files(request.input_path)
        if not image_files:
            raise HTTPException(status_code=404, detail="No images found in input path")

        train_files, test_files = split_dataset(image_files, request.split_ratio)
        train_dir, test_dir = organize_dataset(
            train_files, test_files, request.output_path
        )

        return {
            "message": "Dataset split successfully",
            "train_count": len(train_files),
            "test_count": len(test_files),
            "train_dir": train_dir,
            "test_dir": test_dir,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
