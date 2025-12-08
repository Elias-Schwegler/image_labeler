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
    """
    Request model for the split-dataset endpoint.
    """

    input_path: str
    output_path: str
    split_ratio: float = 0.8


@app.post("/label-image/")
async def api_label_image(file: UploadFile = File(...)):
    """
    Upload an image and get its label/description from LM Studio.

    Args:
        file (UploadFile): The image file to be labeled.

    Returns:
        dict: A dictionary containing the label, description, and tags.
    """
    # Save uploaded file temporarily to disk because the labeler
    # expects a file path
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Process the image using the local LM Studio model
        result = label_image(tmp_path)
        return result
    finally:
        # Ensure the temporary file is removed after processing
        os.remove(tmp_path)


@app.post("/split-dataset/")
async def api_split_dataset(request: SplitRequest):
    """
    Split a dataset located at input_path into train/test sets at output_path.

    Args:
        request (SplitRequest): The request object containing input/output
        paths and split ratio.

    Returns:
        dict: A summary of the split operation, including counts and paths.
    """
    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail="Input path not found")

    try:
        # Retrieve all valid image files from the input directory
        image_files = get_image_files(request.input_path)
        if not image_files:
            raise HTTPException(
                status_code=404, detail="No images found in input path"
            )

        # Perform the random split based on the requested ratio
        train_files, test_files = split_dataset(
            image_files, request.split_ratio
        )

        # Copy files to their respective train/test directories
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
