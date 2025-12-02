# Image Labeler

## Project Title and Description
**Image Labeler** is a tool designed to prepare image datasets for training recognition models. It leverages a local LM Studio model (specifically Qwen VL) to automatically label images and provides utilities to split the dataset into training and testing sets. The project includes a CLI, a FastAPI backend, and a Streamlit UI.

## Use Case: Automated Personal Photo Organization
The primary goal of this project is to bootstrap a high-quality labeled dataset using a Vision Language Model (VLM). This labeled dataset will serve as the ground truth for training a lightweight, specialized image recognition model. Once trained, this specialized model can label and categorize personal photo collections significantly faster and more efficiently than a large VLM, enabling rapid, offline organization of large photo libraries into meaningful categories.

## Setup Instructions

### Prerequisites
- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running locally with an OpenAI-compatible server.
- Loaded Vision Language Model (e.g., `qwen/qwen3-vl-4b`).

### Installation
1.  Clone the repository.
2.  Create and activate a virtual environment:
    **Option A: Using Conda**
    ```bash
    conda create -n image_labeler_env python=3.10
    conda activate image_labeler_env
    ```

    **Option B: Using venv (Standard Python)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

    **Option C: Using pyenv**
    ```bash
    pyenv install 3.10.6
    pyenv virtualenv 3.10.6 image_labeler_env
    pyenv activate image_labeler_env
    ```
3.  Install dependencies:
    ```bash
    pip install -e .
    ```
4.  Configure environment variables:
    - Copy `.env` example (if provided) or create one:
      ```
      LM_STUDIO_URL=http://127.0.0.1:1234/v1
      LM_STUDIO_MODEL=qwen/qwen3-vl-4b
      ```

## How to Run

### CLI
```bash
python src/main.py --path ./data/raw --split-ratio 0.8 --output ./data/processed
```

### Streamlit UI
```bash
streamlit run src/app.py
```

### API
```bash
uvicorn src.api:app --reload
```

## Expected Outputs
- **Labeled Data**: JSON files containing image descriptions/labels.
- **Split Data**: Organized folders for `train` and `test` sets.
- **Notebooks**: Experimental code and analysis in `notebooks/`.

## Dependencies
- `streamlit`
- `openai` (for LM Studio client)
- `fastapi`
- `uvicorn`
- `python-dotenv`
- `Pillow`

## Credits
Project created for HSLU Bachelor Informatik course.
