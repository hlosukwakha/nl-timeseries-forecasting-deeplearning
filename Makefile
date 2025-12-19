SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE ?= docker compose

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_.-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS=":.*## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: dev
dev: ## Start MLflow + Prometheus + Grafana (and optionally API) in background
	$(COMPOSE) up --build -d mlflow prometheus grafana

.PHONY: train
train: ## Run end-to-end pipeline: download -> preprocess -> train(4 models) -> evaluate
	$(COMPOSE) run --rm trainer python -m src.cli run-all

.PHONY: preprocess
preprocess: ## Download + preprocess only
	$(COMPOSE) run --rm trainer python -m src.cli download-data
	$(COMPOSE) run --rm trainer python -m src.cli preprocess

.PHONY: api
api: ## Build + start API container in background
	$(COMPOSE) up --build -d api

.PHONY: logs
logs: ## Tail logs for core services
	$(COMPOSE) logs -f mlflow api prometheus grafana

.PHONY: stop
stop: ## Stop all services
	$(COMPOSE) down

.PHONY: clean-data
clean-data: ## Delete local data artifacts (raw/processed/reports)
	rm -rf data/raw/knmi data/processed data/reports

.PHONY: clean-all
clean-all: ## Delete data + mlruns + models (full local reset)
	rm -rf data/raw/knmi data/processed data/reports mlruns models

.PHONY: rebuild-trainer
rebuild-trainer: ## Rebuild trainer image (uses cache)
	$(COMPOSE) build trainer

.PHONY: rebuild-trainer-nocache
rebuild-trainer-nocache: ## Rebuild trainer image without cache (use when code changes aren't picked up)
	$(COMPOSE) build --no-cache trainer

.PHONY: shell-trainer
shell-trainer: ## Open a shell in the trainer container
	$(COMPOSE) run --rm trainer bash

.PHONY: verify-code
verify-code: ## Print build_supervised_frame from inside container to confirm image has latest code
	$(COMPOSE) run --rm trainer python -c "import inspect; import src.data.features as f; print(inspect.getsource(f.build_supervised_frame))"
