.PHONY: install test demo api clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

demo:
	python -m lending_ai_lab.cli demo --n-samples 12000 --epochs 8 --output-dir artifacts

api:
	uvicorn lending_ai_lab.serving.api:app --host 0.0.0.0 --port 8000

clean:
	rm -rf artifacts/* site/index.html .pytest_cache
