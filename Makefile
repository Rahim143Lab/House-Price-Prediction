.PHONY: install eda train evaluate test api frontend clean

install:
	pip install -r requirements.txt

eda:
	python -m src.eda

train:
	python -m src.train

evaluate:
	python -m src.evaluate

test:
	pytest tests/ -v

api:
	uvicorn api.main:app --reload

frontend:
	cd frontend && python -m http.server 8080

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
