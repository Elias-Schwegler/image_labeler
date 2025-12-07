# Image Labeler

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![License](https://img.shields.io/badge/License-MIT-green)

**Image Labeler** is a powerful tool to bootstrap high-quality labeled datasets using local Vision Language Models (VLMs). It automates the tedious process of image labeling, enabling you to build specialized recognition models faster.

## Use Case: Automated Personal Photo Organization

The primary goal of this project is to create a ground-truth dataset for training lightweight image recognition models.

1.  **Bootstrap**: Use a large, slow VLM (like Qwen VL via LM Studio) to accurately label a raw collection of images.
2.  **Train**: Use this labeled dataset to train a smaller, faster specialized model (e.g., EfficientNet, ResNet).
3.  **Deploy**: Run the specialized model on personal devices to organize photo libraries offline, privately, and rapidly.

## Architecture

The system consists of three main interfaces (CLI, Streamlit UI, FastAPI) that interact with shared core logic modules. The `Labeler` module communicates with a local LM Studio instance to generate descriptions.

```text
    +--------+
    |  User  |
    +---+----+
        |
        v
+-------+---------------------------------------------------+
|                   Interfaces                              |
|                                                           |
|  +-------------+    +-------------+    +-------------+    |
|  |     CLI     |    | Streamlit UI|    |   FastAPI   |    |
|  |  (main.py)  |    |   (app.py)  |    |   (api.py)  |    |
|  +------+------+    +------+------+    +------+------+    |
|         |                  |                  |           |
+---------+------------------+------------------+-----------+
          |                  |                  |
          v                  v                  v
+---------+------------------+------------------+-----------+
|                   Core Modules                            |
|                                                           |
|  +-------------+    +-------------+    +-------------+    |
|  | Data Loader |    |   Labeler   |    |   Splitter  |    |
|  +-------------+    +------+------+    +-------------+    |
|                            |                              |
+----------------------------+------------------------------+
                             |
                             v
                  +----------+-----------+
                  |  External Services   |
                  |                      |
                  |     LM Studio        |
                  |     (Qwen VL)        |
                  +----------------------+
```

### Components

*   **Interfaces**:
    *   **CLI (`src/main.py`)**: Command-line entry point for batch processing.
    *   **Streamlit UI (`src/app.py`)**: Web-based interface for interactive labeling and management.
    *   **FastAPI (`src/api.py`)**: REST API for programmatic access to labeling features.
*   **Core Modules**:
    *   **Data Loader (`src/data_loader.py`)**: Handles file scanning and storage operations.
    *   **Labeler (`src/labeler.py`)**: Prepares images and sends requests to the LM Studio API.
    *   **Splitter (`src/splitter.py`)**: Logic for splitting datasets and organizing files.
*   **External Services**:
    *   **LM Studio**: Local server running the Vision Language Model (e.g., Qwen VL) that performs the actual image analysis.

## Key Features

*   **🤖 Automated Labeling**: Leverage local LLMs/VLMs for privacy-first, offline image labeling.
*   **⚡ Resolution Control**: Configurable resizing (256px - 1024px) to optimize for speed or detail.
*   **📊 Smart Splitting**:
    *   Randomly split into Train/Test sets.
    *   **Resume Capability**: Pick up where you left off using existing `labels.json`.
*   **🛑 Safe Interruption**: Stop labeling at any time; progress is automatically saved.

## Setup Instructions

### Prerequisites
*   **Python 3.10+**
*   **[LM Studio](https://lmstudio.ai/)**: Must be running locally with the server started.
*   **Model**: A VLM like `qwen/qwen3-vl-4b` or similar loaded in LM Studio.

### Installation

1.  **Clone the repository**
2.  **Create Virtual Environment** (Choose one)
    *   **Conda**: `conda create -n image_labeler_env python=3.10 && conda activate image_labeler_env`
    *   **Venv**: `python -m venv venv && source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
3.  **Install Dependencies**
    ```bash
    pip install -e .
    ```
4.  **Configure Environment**
    Create a `.env` file:
    ```env
    LM_STUDIO_URL=http://127.0.0.1:1234/v1
    LM_STUDIO_MODEL=qwen/qwen3-vl-4b
    ```
    > **Note**: `LM_STUDIO_MODEL` is mandatory. Images are resized to max 1024px.

## Usage

### 🖥️ Streamlit UI (Recommended)
Interactive web interface for labeling and splitting.
```bash
streamlit run src/app.py
```

### 💻 Command Line Interface
Batch process images directly.
```bash
python src/main.py --path ./data/raw --split-ratio 0.8 --output ./data/processed
```

### 🔌 API Server
Run the backend API.
```bash
```bash
uvicorn src.api:app --reload
```
View the interactive API documentation (Swagger UI) at:
- http://127.0.0.1:8000/docs

## Expected Outputs

*   **📄 labels.json**: A JSON file containing image paths and their generated labels.
*   **📂 /train & /test**: Organized directories containing the split dataset.
*   **📓 Notebooks**: Experimental code in `notebooks/`.

## Project Structure

A detailed overview of the project's file organization and module responsibilities.

```text
image_labeler/
├── 📂 data/                   # Directory for storing raw and processed images
│   └── 📂 raw/                # Place your input images here
├── 📂 src/                    # Source code directory
│   ├── 🐍 api.py              # FastAPI backend application
│   ├── 🐍 app.py              # Streamlit frontend application
│   ├── 🐍 data_loader.py      # Utilities for loading files and saving JSON
│   ├── 🐍 labeler.py          # Logic for interacting with LM Studio API
│   ├── 🐍 main.py             # CLI entry point for batch processing
│   └── 🐍 splitter.py         # Logic for splitting datasets (Train/Test)
├── 📂 tests/                  # Test suite
│   ├── 🐍 manual_test_api.py  # Script for manual API testing
│   ├── 🐍 manual_test_split.py# Script for manual split logic testing
│   ├── 🐍 test_core.py        # Unit tests for core logic (splitting, loading)
│   ├── 🐍 test_features.py    # Feature tests (image resizing, encoding)
│   └── 🐍 verify_api.py       # Helper script for API verification
├── 🐳 Dockerfile              # Docker image definition for the app
├── 🐙 docker-compose.yml      # Orchestration for running the app container
├── ⚙️ requirements.txt        # Python dependencies
└── 📄 README.md               # Project documentation
```

### File Legend
- **`src/app.py`**: The main user interface. It handles the Streamlit session state, displays the file uploader, and visualizes progress.
- **`src/labeler.py`**: Contains `label_image` and `encode_image`. It handles image resizing (to max 1024px) for optimization and constructs the payload for the Vision Language Model.
- **`src/splitter.py`**: Implements the `split_dataset` logic to randomly divide files based on the requested ratio and `organize_dataset` to move files into `train/` vs `test/` folders.
- **`tests/test_features.py`**: Verifies that images are correctly resized before being sent to the model to avoid token limit issues and test base64 encoding.

## Testing

The project includes a robust test suite using `pytest` to ensure reliability.

### Running Tests
To run the full test suite:
```bash
pytest
```

### Test Scope
1.  **Unit Tests (`tests/test_core.py`)**:
    - Verifies that `split_dataset` respects the requested split ratio.
    - Ensures no data leakage between Train and Test sets.
2.  **Feature Tests (`tests/test_features.py`)**:
    - **Image Resizing**: Confirms that large images (>1024px) are automatically resized.
    - **Encoding**: Checks that images are correctly converted to Base64 for API transmission.
    - **Real Data**: If available, tests processing on actual image files in `data/raw`.

## Docker Deployment

You can containerize the application for consistent deployment.

### Using Docker Compose
The included `docker-compose.yml` sets up the application and maps port `8501` (Streamlit) and `8000` (FastAPI).

1.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```
2.  **Access**:
    - Streamlit UI: `http://localhost:8501`
    - API Docs: `http://localhost:8000/docs`

> **Note on Networking**: The container uses `host.docker.internal` to communicate with the LM Studio running on your host machine. Ensure LM Studio is listening on all interfaces (0.0.0.0) or specifically allows connections from the container.

## Credits
Project created for HSLU Bachelor Informatik course.
