from setuptools import setup, find_packages

setup(
    name="image_labeler",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "openai",
        "python-dotenv",
        "Pillow",
        "pytest",
        "fastapi",
        "uvicorn",
        "requests",
        "black",
        "flake8",
    ],
)
