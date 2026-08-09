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

live: ## live smoke test (needs GEMINI_API_KEY in .env)
	.venv/bin/python -m harness.live_smoke

venv: ## create .venv and install harness + experiments
	python3 -m venv .venv
	.venv/bin/pip install -e .

experiment: ## offline matrix run + analysis (deterministic, free)
	.venv/bin/python -m experiments.suite --out artifacts
	.venv/bin/python -m experiments.run --grid artifacts/grid.json --out artifacts/results.json
	.venv/bin/python -m experiments.analyze --results artifacts/results.json

experiment-live: ## small live stratified sample (needs GEMINI_API_KEY in .env)
	.venv/bin/python -m experiments.run --live --limit 12 --out artifacts/results-live.json
	.venv/bin/python -m experiments.analyze --results artifacts/results-live.json
