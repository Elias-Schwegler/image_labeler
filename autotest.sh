#!/bin/bash

echo "Running tests..."
pytest tests/

echo "Checking formatting..."
black --check src/ tests/

echo "Linting..."
flake8 src/ tests/

echo "Done."
