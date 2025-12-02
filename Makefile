install:
	pip install -e .

test:
	pytest tests/

run-ui:
	streamlit run src/app.py

run-api:
	uvicorn src.api:app --reload

docker-build:
	docker build -t image_labeler .

docker-run:
	docker-compose up
