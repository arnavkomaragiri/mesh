start:
	docker compose up

stop:
	docker compose down

install:
	poetry install
	wget https://github.com/milvus-io/milvus/releases/download/v2.3.1/milvus-standalone-docker-compose.yml -O docker-compose.yml