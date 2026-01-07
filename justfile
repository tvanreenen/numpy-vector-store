# Clean build artifacts and cache directories
clean:
    rm -rf .venv .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov ssl
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
