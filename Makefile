export PYTHONPATH := src:app
.PHONY: data train monitor figures test serve docker
data:
	bash scripts/download_data.sh
train:
	python -m loadforecast.train --market spain
	python -m loadforecast.train --market pjme
monitor:
	python -m loadforecast.monitoring --market pjme
figures:
	python scripts/make_figures.py
test:
	python -m pytest -q tests
serve:
	uvicorn main:app --reload --port 8080
docker:
	docker build -t load-forecast:latest .
	docker run --rm -p 8080:8080 load-forecast:latest
