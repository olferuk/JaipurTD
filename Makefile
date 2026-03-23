.PHONY: install train test lint fmt evaluate help ci

SRC = ai/ jaipur/ tests/ *.py

ifeq ($(shell command -v uv 2>/dev/null),)
  RUN = python3 -m
else
  RUN = uv run
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (uv or pip)
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra dev; \
	else \
		python3 -m pip install -e ".[dev]"; \
	fi

train: ## Train the neural agent
	$(RUN) python -m ai.trainer

test: ## Run tests
	$(RUN) pytest tests/ -v

lint: ## Lint with ruff
	$(RUN) ruff check $(SRC)

fmt: ## Format code with ruff
	$(RUN) ruff format $(SRC)
	$(RUN) ruff check --fix $(SRC)

evaluate: ## Evaluate neural agent vs baselines
	$(RUN) python -m evaluate

ci: lint test ## Run lint + tests
