.PHONY: test vet serve demo venv experiment

test: ## go test -race + pytest
	cd watchtower && go test ./... -race
	.venv/bin/pytest tests -q

vet: ## go vet ./...
	cd watchtower && go vet ./...

serve: ## run watchtower serve on :4318
	cd watchtower && go run ./cmd/watchtower serve --addr :4318

demo: ## offline python demo: fake provider -> spans -> serve -> report
	.venv/bin/python -m harness.demo

venv: ## create .venv and install harness + experiments
	python3 -m venv .venv
	.venv/bin/pip install -e .

experiment: ## run the seeded experiment suite
	.venv/bin/python -m experiments.suite --out artifacts
