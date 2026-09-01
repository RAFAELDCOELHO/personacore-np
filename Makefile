.PHONY: infer

# One human command: download public slim weights if missing, build public
# fixtures from them, run the suite. No private companion repo.
infer:
	python3 -c "import torch" 2>/dev/null || python3 -m pip install 'torch' --index-url https://download.pytorch.org/whl/cpu
	python3 -c "import pytest, numpy" 2>/dev/null || python3 -m pip install 'pytest>=8.0' 'numpy>=2.0'
	python3 scripts/gen_public_fixtures.py
	python3 -m pytest
