.PHONY: install train test style lint evaluate help ci

PYTHON ?= python3

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (uv or pip)
	@if command -v uv >/dev/null 2>&1; then \
		uv sync; \
	else \
		$(PYTHON) -m pip install -e ".[dev]"; \
	fi

train: ## Train the neural agent (self-play TD learning)
	$(PYTHON) -m ai.trainer

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

style: ## Format code with black
	$(PYTHON) -m black ai/ jaipur/ tests/ *.py

lint: ## Lint with ruff (catches Optional, old List/Dict, etc.)
	$(PYTHON) -m ruff check ai/ jaipur/ tests/ *.py

evaluate: ## Evaluate neural agent vs baselines
	$(PYTHON) -m evaluate

ci: lint test ## Run lint + tests (CI pipeline)
