Write-Output "Running tests..."
pytest tests/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "Checking formatting..."
black --check src/ tests/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "Linting..."
flake8 src/ tests/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "Done."
